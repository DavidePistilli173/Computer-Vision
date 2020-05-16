#include "PanoramicImage.hpp"

#include "panoramic_utils.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <algorithm>

using namespace lab5;

bool PanoramicImage::computeMatches(double ratio)
{
    /* Input check. */
    if (descriptors_.empty())
    {
        Log::error("No available feature descriptors.");
        return false;
    }

    Log::info("Creating descriptor matcher.");
    int normType{ cv::NORM_HAMMING };
    if (currentMode_ == Mode::sift) normType = cv::NORM_L2;
    auto bfMatcher = cv::BFMatcher::create(normType);
    
    /* Reset output. */
    matches_.clear();
    matches_.resize(descriptors_.size() - 1);

    /* From the second image onwards, compute its matches with the previous one. */
    for (size_t i = 1; i < descriptors_.size(); ++i)
    {
        size_t prev{ i - 1 };
        Log::info("Computing matches for images %d and %d.", prev, i);
        bfMatcher->match(descriptors_[prev], descriptors_[i], matches_[prev]);

        /* Compute the minimum distance between matches. */
        float minDist
        {
            std::min_element(
                matches_[prev].begin(), 
                matches_[prev].end(), 
                [](const cv::DMatch& left, const cv::DMatch& right) { return left.distance < right.distance; }
            )->distance
        };
        Log::info("Minimum distance between matches: %f.", minDist);
        double distTh{ minDist * ratio };
        Log::info("Distance threshold: %f.", distTh);

        /* Remove matches that are too distant from each other. */
        int count{ 0 };
        for (auto it = matches_[prev].begin(); it != matches_[prev].end();)
        {
            if (it->distance > distTh)
            {
                it = matches_[prev].erase(it);
                ++count;
            }
            else
            {
                ++it;
            }
        }
        Log::info("Removed %d matches.", count);
        if (matches_[prev].empty())
        {
            Log::error("No matches remaining.");
            return false;
        }

        /* Check outliers. */
        Log::info("Computing homography matrix.");
        std::vector<cv::Point2f> srcPts;
        std::vector<cv::Point2f> dstPts;
        for (const auto& match : matches_[prev])
        {
            srcPts.emplace_back(keypoints_[prev][match.queryIdx].pt);
            dstPts.emplace_back(keypoints_[i][match.trainIdx].pt);
        }
        
        cv::Mat matchMask;
        cv::Mat H{ cv::findHomography(srcPts, dstPts, matchMask, cv::RANSAC) };

        Log::info("Computing average translation.");
        cv::Point2f translation{ 0.F, 0.F };
        count = 0;
        for (int i = 0; i < dstPts.size(); ++i)
        {
            if (matchMask.at<bool>(i))
            {
                ++count;
                translation += (srcPts[i] - dstPts[i]);
            }
        }

        /* Handle bad matches. */
        if (count == 0)
        {
            Log::error("No inliers found. Forcing translation to image size.");
            translation.x = cilProj_[prev].cols;
            translation.y = 0;
        }
        else
        {
            translation /= count;
            if (translation.x < 0)
            {
                Log::warn("Negative translation value. Forcing translation to image size.");
                if (translation.x < 0) translation.x = cilProj_[prev].cols;
            }
        }
        Log::info("Number of inliers: %d/%d.", count, dstPts.size());

        translations_.emplace_back(cvRound(translation.x), cvRound(translation.y));
        Log::info(
            "Average translation for images %d and %d is (%d, %d).", 
            prev, 
            i, 
            translations_[prev].x, 
            translations_[prev].y
        );
    }

    return true;
}

cv::Mat PanoramicImage::computePanorama() const
{
    /* Compute final image size. */
    Log::info("Computing final panorama size.");
    int width{ 0 }; // Overall width of the panorama.
    int above{ 0 }; // Vertical space above the initial image.
    int below{ 0 }; // Vertical space below the initial image.
    for (int i = 1; i < cilProj_.size(); ++i)
    {
        auto [transX, transY] = translations_[i - 1];
        if (transX > 0) width += transX;
        if (transY > 0 && below < transY) below = transY;
        else if (above < -transY) above = -transY;
    }
    width += cilProj_[cilProj_.size() - 1].cols;
    int height{ above + cilProj_[0].rows + below };
    Log::info("Size: (%d,%d).", width, height);

    /* Initialise the result image. */
    cv::Mat result{ cv::Mat::zeros(height, width, cilProj_[0].type()) };

    /* Copy all images to the output. */
    cilProj_[0].copyTo(result(cv::Range{ above, cilProj_[0].rows + above }, cv::Range{ 0, cilProj_[0].cols }));
    int x{ 0 };
    for (int i = 1; i < cilProj_.size(); ++i)
    {
        Log::info("Processing image %d.", i);
        auto [transX, transY] = translations_[i - 1];
        x += transX;
        cilProj_[i].copyTo(result(
            cv::Range{ above + transY, cilProj_[i].rows + above + transY }, 
            cv::Range{ x, x + cilProj_[i].cols }
        ));

        /* Blur the transition between images. */
        cv::Mat transitionArea{ result(
            cv::Range{ 0, cilProj_[i].rows }, 
            cv::Range{ x - transition_side, x + transition_side }
        ) };
        cv::GaussianBlur(
            transitionArea,
            transitionArea,
            cv::Size{ kernel_size.w, kernel_size.h },
            gauss_sigma
        );
    }

    Log::info("Equalising final histogram.");
    cv::equalizeHist(result, result);
    return result;
}

bool PanoramicImage::extractORB()
{
    /* Input check. */
    if (cilProj_.empty())
    {
        Log::error("No available projected images.");
        return false;
    }

    /* Reset output. */
    keypoints_.clear();
    descriptors_.clear();

    /* Compute keypoints and descriptors. */
    Log::info("Creating ORB object.");
    auto orb = cv::ORB::create();
    Log::info("Finding keypoints.");
    orb->detect(cilProj_, keypoints_);
    Log::info("Computing descriptors.");
    orb->compute(cilProj_, keypoints_, descriptors_);

    /* Change operating mode. */
    currentMode_ = Mode::orb;
    return true;
}

bool PanoramicImage::extractSIFT()
{
    /* Input check. */
    if (cilProj_.empty())
    {
        Log::error("No available projected images.");
        return false;
    }

    /* Reset output. */
    keypoints_.clear();
    descriptors_.clear();

    Log::info("Creating SIFT object.");
    auto sift = cv::xfeatures2d::SIFT::create();
    Log::info("Finding keypoints.");
    sift->detect(cilProj_, keypoints_);
    Log::info("Computing descriptors.");
    sift->compute(cilProj_, keypoints_, descriptors_);

    /* Change operating mode. */
    currentMode_ = Mode::sift;
    return true;
}

bool PanoramicImage::loadImages(std::string_view folder)
{
    /* Load images filenames. */
    std::vector<cv::String> fileNames;
    cv::utils::fs::glob(
        folder.data(),
        file_pattern.data(),
        fileNames
    );
    if (fileNames.size() < min_images)
    {
        Log::error("Not enough images found in folder %s.", folder.data());
        return false;
    }

    imgs_.clear();
    /* Load image files. */
    Log::info("Loading image %s.", fileNames[0].c_str());
    if (imgs_.emplace_back(cv::imread(fileNames[0])).empty())
    {
        Log::error("FAILED");
        return false;
    }

    int rows{ imgs_[0].rows };
    int cols{ imgs_[0].cols };
    for (int i = 1; i < fileNames.size(); ++i)
    {
        Log::info("Loading image %s.", fileNames[i].c_str());
        if (imgs_.emplace_back(cv::imread(fileNames[i])).empty())
        {
            Log::error("Loading failed.");
            return false;
        }
        if (imgs_[i].rows != rows || imgs_[i].cols != cols)
        {
            Log::error("The image does not have the same resolution as the others.");
            return false;
        }
    }

    return true;
}

bool PanoramicImage::projectImages(int hfov)
{
    /* Input check. */
    if (imgs_.empty())
    {
        Log::error("No available images.");
        return false;
    }

    /* Compute the cilindrical projection for all images and equalise all histograms. */
    cilProj_.clear();
    for (int i = 0; i < imgs_.size(); ++i)
    {
        Log::info("Projecting image %d.", i);
        cilProj_.emplace_back(PanoramicUtils::cylindricalProj(imgs_[i], hfov));

        Log::info("Equalising image %d.", i);
        cv::equalizeHist(cilProj_[i], cilProj_[i]);
    }
}
