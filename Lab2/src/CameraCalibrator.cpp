#include <algorithm>
#include <cmath>
#include <exception>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>

#include "CameraCalibrator.hpp"

using namespace lab2;

void CameraCalibrator::calibrate()
{
    if (calibrationImgs_ == nullptr)
    {
        Log::error("No calibration images provided.");
        return;
    }

    /* Clear previous computations. */
    resetData_();

    /* Compute the 2d positions of the chessboard corners. */
    Log::info("Looking for chessboard corners...");
    /* Use multiple threads only if there are enough images to process. */
    if (calibrationImgs_->size() < 2 * thread_num)
    {
        processCorners_();
    }
    else
    {
        std::vector<std::thread> threads;
        threads.reserve(thread_num);
        for (int i = 0; i < thread_num; ++i)
        {
            threads.emplace_back([this]() { this->processCorners_(); });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }
    }

    /* Check that all images were processed correctly. */
    int validImgs{ std::accumulate(validImgs_.begin(), validImgs_.end(), 0) };
    Log::info(
        "Corners found in %d/%d images.",
        validImgs,
        calibrationImgs_->size()
        );
    if (validImgs < calibrationImgs_->size())
    {
        Log::error("Not all images were processed correctly.");
        return;
    }

    Log::info("Calibrating camera...");
    double error = cv::calibrateCamera(
        corners3D_, corners2D_, (*calibrationImgs_)[0].size(), results_.camera,
        results_.distParams, results_.rotVecs, results_.tVecs
    );
    Log::info("Calibration completed with an RMS reprojecton error of %f.", error);
}

CalibrationStats CameraCalibrator::meanProjErr() const
{
    CalibrationStats stats;
    for (size_t imgIndx = 0; imgIndx < corners3D_.size(); ++imgIndx)
    {
        /* Reproject 3D points on the image plane. */
        std::vector<cv::Point2f> projectedCorners;
        cv::projectPoints(
            corners3D_[imgIndx], results_.rotVecs[imgIndx], results_.tVecs[imgIndx], 
            results_.camera, results_.distParams, projectedCorners
            );

        /* Compute the mean reprojection error. */
        double meanImgError{ 0.0 };
        for (size_t i = 0; i < projectedCorners.size(); ++i)
        {
            cv::Point2f pt = projectedCorners[i] - corners2D_[imgIndx][i];
            meanImgError += std::sqrt(std::pow(pt.x, 2) + std::pow(pt.y, 2));
        }
        meanImgError /= static_cast<int>(projectedCorners.size());

        if (meanImgError < stats.minRE.err)
        {
            stats.minRE.err = meanImgError;
            stats.minRE.indx = imgIndx;
        }
        else if (meanImgError > stats.maxRE.err)
        {
            stats.maxRE.err = meanImgError;
            stats.maxRE.indx = imgIndx;
        }

        stats.meanRE += meanImgError;
    }
    stats.meanRE /= corners3D_.size();
        
    return stats;
}

void CameraCalibrator::printResults()
{
    Log::info("%.75s", Log::asterisks);
    Log::info("%.27s CALIBRATION RESULTS %.27s", Log::asterisks, Log::asterisks);
    Log::info("*** INTRINSIC PARAMETERS %.46s ***", Log::hyphens);
    Log::info("*** -- alpha_u = %7.2f %.46s ***", results_.camera.at<double>(0, 0), Log::hyphens);
    Log::info("*** -- alpha_v = %7.2f %.46s ***", results_.camera.at<double>(1, 1), Log::hyphens);
    Log::info("*** -- u_0     = %7.2f %.46s ***", results_.camera.at<double>(0, 2), Log::hyphens);
    Log::info("*** -- v_0     = %7.2f %.46s ***", results_.camera.at<double>(1, 2), Log::hyphens);
    Log::info("*** %.67s ***", Log::hyphens);
    Log::info("*** DISTORTION PARAMETERS %.45s ***", Log::hyphens);
    Log::info(
        "*** -- Radial:     k_1 = %9.2e | k_2 = %9.2e | k_3 = %9.2e  ***",
        results_.distParams[0], results_.distParams[1], results_.distParams[4], Log::hyphens
    );
    Log::info(
        "*** -- Tangential: p_1 = %9.2e | p_2 = %9.2e %.18s ***", 
        results_.distParams[2], results_.distParams[3], Log::hyphens
    );
    Log::info("%.75s", Log::asterisks);
}

void CameraCalibrator::undistort(cv::Mat img) const
{
    cv::Mat newCamMat;
    /* Reprojection maps. */
    cv::Mat mX{ img.size(), CV_32FC1 };
    cv::Mat mY{ img.size(), CV_32FC1 };
    cv::initUndistortRectifyMap(
        results_.camera, results_.distParams, cv::Mat{}, newCamMat, img.size(), CV_32FC1, mX, mY
    );
    cv::remap(img, img, mX, mY, cv::INTER_LINEAR);
}

bool CameraCalibrator::setCalibImgs(const std::vector<cv::Mat>* imgs, Size<float> cellSize, Size<int> patSize)
{
    /* Check that imgs is valid. */
    if (imgs == nullptr)
    {
        Log::error("Null pointer to calibration images.");
        return false;
    }
    if (imgs->size() < min_imgs)
    {
        Log::error("Camera calibration requires at least %d images.", min_imgs);
        return false;
    }

    /* Change relevant calibration parameters. */
    calibrationImgs_ = imgs;
    cellSize_ = cellSize;
    patternSize_.width = patSize.w;
    patternSize_.height = patSize.h;

    /* Recompute 3D corner positions. */
    corners3D_.clear();
    corners3D_.resize(calibrationImgs_->size());

    /* Compute a single corner pose. */
    int cornerNum{ patternSize_.width * patternSize_.height };
    std::vector<cv::Point3f> samplePose;
    samplePose.resize(cornerNum);
        
    for (size_t rows = 0; rows < patternSize_.height; ++rows)
    {
        for (size_t cols = 0; cols < patternSize_.width; ++cols)
        {
            samplePose[rows * patternSize_.width + cols] = 
                cv::Point3f(cols * cellSize_.w, rows * cellSize_.h, 0.0F);
        }
    }
    /* Copy the sample pose in each pose. */
    for (auto& pose : corners3D_)
    {
        pose = samplePose;
    }

    return true;
}

void CameraCalibrator::processCorners_()
{
    for (size_t index = nextImg_++; index < calibrationImgs_->size(); index = nextImg_++)
    {
        /* Find initial estimates for the corners. */
        if (!cv::findChessboardCorners((*calibrationImgs_)[index], patternSize_, corners2D_[index]))
        {
            Log::warn("Unable to find corners in image %d.", index);
        }
        else
        {
            Log::info("Corners found in image %d.", index);
            validImgs_[index] = true;
            /* Refine corner estimates. */
            Log::info("Refining corners for image %d.", index);
            cv::cornerSubPix((*calibrationImgs_)[index], corners2D_[index], winSize_, zeroZone_, pixRefCriteria_);
        }
    }
}

void CameraCalibrator::resetData_()
{
    Log::info("Resetting calibration data...");
    corners2D_.clear();
    corners2D_.resize(calibrationImgs_->size());
    validImgs_.clear();
    validImgs_.resize(calibrationImgs_->size(), false);
    nextImg_ = 0;
    results_.reset();
    Log::info("SUCCESS.");
}