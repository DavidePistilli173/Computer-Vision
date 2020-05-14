#ifndef CAMERACALIBRATOR_HPP
#define CAMERACALIBRATOR_HPP

#include <atomic>
#include <opencv2/core.hpp>
#include <vector>

#include "lab2.hpp"

namespace lab2
{
    /* Performs camera calibration on a given dataset. */
    class CameraCalibrator
    {
    public:
        CameraCalibrator() = default;

        /********** CONSTANTS **********/
        static constexpr int min_imgs{ 2 }; // Minimum number of calibration images required.
        static constexpr int thread_num{ 4 }; // Number of threads used.
        /* Default corner refinement parameters. */
        static constexpr Size def_win_size{ 5, 5 }; // Window size.
        static constexpr Size def_zero_zone{ -1, -1 }; // Zero zone.
        static constexpr double def_epsilon{ 0.1 }; // Accuracy.
        static constexpr int def_max_iter{ 30 }; // Maximum number of iterations.

        /********** METHODS **********/
        void calibrate(); // Perform camera calibration on stored data.
        CalibrationStats meanProjErr() const; // Compute the mean reprojection error.
        void printResults(); // Print calibration results.
        void undistort(cv::Mat img) const; // Undistort a given image.
        // Set the calibration images.
        bool setCalibImgs(
            const std::vector<cv::Mat>* imgs, Size<float> cellSize, Size<int> patSize
        ); 

    private:
        /********** METHODS **********/
        // Find and refine checkerboard corners in a thread-safe manner.
        void processCorners_();
        // Reset calibration data.
        void resetData_();

        /********** VARIABLES **********/
        const std::vector<cv::Mat>* calibrationImgs_{ nullptr }; // Images used for calibration.
        std::vector<std::vector<cv::Point2f>> corners2D_; // Corners detected for each image.
        std::vector<std::vector<cv::Point3f>> corners3D_; // Coordinates of the corners in the real reference system.
        std::vector<bool> validImgs_; // Keeps track of which images have computed corners and which don't.
        std::atomic<size_t> nextImg_{ 0 }; // Keeps track of what image needs processing.
        CalibrationData results_; // Calibration results.
        /* Corner refinement parameters. */
        Size<float> cellSize_{ 0.0F, 0.0F }; // Size of each cell of the calibration pattern.
        cv::Size patternSize_{ 0, 0 }; // Number of inner corners of the chessboard.
        cv::Size winSize_{ def_win_size.w, def_win_size.h }; // Size of the window used by cornerSubPix.
        cv::Size zeroZone_{ def_zero_zone.w, def_zero_zone.h }; // Size of the zero zone used by cornerSubPix.
        /* Criteria for pixel refinement. */
        cv::TermCriteria pixRefCriteria_
        {
            cv::TermCriteria::Type::COUNT + cv::TermCriteria::Type::EPS, 
            def_max_iter, 
            def_epsilon
        };
    };
}

#endif