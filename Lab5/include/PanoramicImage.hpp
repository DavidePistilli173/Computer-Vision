#ifndef PANORAMICIMAGE_HPP
#define PANORAMICIMAGE_HPP

#include <opencv2/core.hpp>

#include <string_view>
#include <vector>

namespace lab5
{
    class PanoramicImage
    {
    public:
        /********** CONSTANTS **********/
        static constexpr std::string_view file_pattern{ "i*.*" };

        /********** CONSTRUCTOR **********/
        PanoramicImage() = default;

        /********** METHODS **********/
        bool computeMatches(float ratio);
        cv::Mat computePanorama();
        void extractORB();
        bool loadImages(std::string_view folder);
        void projectImages(int hfov);

    private:
        /********** METHODS **********/


        /********** VARIABLES **********/
        std::vector<cv::Mat> imgs_;
        std::vector<cv::Mat> cilProj_;
        std::vector<std::vector<cv::KeyPoint>> keypoints_;
        std::vector<cv::Mat> descriptors_;
        std::vector<std::vector<cv::DMatch>> matches_;
        std::vector<cv::Point2i> translations_;
    };
}

#endif