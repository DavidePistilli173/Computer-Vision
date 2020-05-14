#ifndef PANORAMICIMAGE_HPP
#define PANORAMICIMAGE_HPP

#include "lab5.hpp"

#include <opencv2/core.hpp>

#include <string_view>
#include <vector>

namespace lab5
{
    class PanoramicImage
    {
    public:
        /********** CONSTANTS **********/
        static constexpr std::string_view file_pattern{ "i*.*" }; // Glob name pattern.
        static constexpr int min_images{ 2 }; // Minimum number of required images.
        static constexpr int transition_side{ 2 };
        static constexpr Size kernel_size{ 7, 1 };
        static constexpr double gauss_sigma{ 2.5 };

        /********** CONSTRUCTOR **********/
        PanoramicImage() = default;

        /********** METHODS **********/
        bool computeMatches(float ratio); // Compute matches between images.
        cv::Mat computePanorama() const; // Compute the final panoramic image.
        bool extractORB(); // Extract ORB features and compute their descriptors.
        bool loadImages(std::string_view folder); // Load input images.
        bool projectImages(int hfov); // Compute the cylindrical projection and equalise all histograms.

    private:
        /********** METHODS **********/


        /********** VARIABLES **********/
        std::vector<cv::Mat> imgs_; // Source images.
        std::vector<cv::Mat> cilProj_; // Projected and equalised images.
        std::vector<std::vector<cv::KeyPoint>> keypoints_; // Keypoints for all images.
        std::vector<cv::Mat> descriptors_; // Descriptors for all images.
        std::vector<std::vector<cv::DMatch>> matches_; // Matches between consecutive images.
        std::vector<cv::Point2i> translations_; // Average translation wrt the previous image.
    };
}

#endif