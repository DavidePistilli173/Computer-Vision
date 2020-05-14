#include "PanoramicImage.hpp"

#include "lab5.hpp"
#include "panoramic_utils.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <algorithm>

using namespace lab5;

bool PanoramicImage::computeMatches(float ratio)
{
    /* Input check. */
    if (descriptors_.empty())
    {
        Log::error("No available feature descriptors.");
        return false;
    }

    Log::info("Creating descriptor matcher.");
    auto bfMatcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    
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
        float distTh{ minDist * ratio };
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
        cv::Mat H{ cv::findHomography(srcPts, dstPts, matchMask, cv::RANSAC, distTh) };

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
            Log::error("No matches found. Forcing translation to image size.");
            translation.x = cilProj_[prev].cols;
            translation.y = cilProj_[prev].rows;
        }
        else
        {
            translation /= count;
            if (translation.x < 0 || translation.y < 0)
            {
                Log::warn("Negative translation value.");
                //if (translation.x < 0) translation.x = cilProj_[prev].cols;
                //if (translation.y < 0) translation.y = cilProj_[prev].rows;
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
    int width{ 0 };
    int height{ cilProj_[0].rows };
    for (int i = 1; i < cilProj_.size(); ++i)
    {
        int prev = i - 1;
        if (translations_[prev].x > 0) width += translations_[prev].x;
    }
    width += cilProj_[cilProj_.size() - 1].cols;

    /* Initialise the result image. */
    cv::Mat result{ cv::Mat::zeros(height, width, cilProj_[0].type()) };

    /* Copy all images to the output. */
    cilProj_[0].copyTo(result(cv::Range{ 0, cilProj_[0].rows }, cv::Range{ 0, cilProj_[0].cols }));
    int x{ 0 };
    for (int i = 1; i < cilProj_.size(); ++i)
    {
        int prev = i - 1;
        x += translations_[prev].x ;
        cilProj_[i].copyTo(result(cv::Range{ 0, cilProj_[i].rows }, cv::Range{ x, x + cilProj_[i].cols }));
    }

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
    for (const auto& fileName : fileNames)
    {
        Log::info("Loading image %s.", fileName.c_str());
        if (imgs_.emplace_back(cv::imread(fileName)).empty())
        {
            Log::error("FAILED");
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
