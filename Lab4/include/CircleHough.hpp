#ifndef CIRCLEHOUGH_HPP
#define CIRCLEHOUGH_HPP

#include "ImgProcessor.hpp"

#include <optional>
#include <utility>

namespace lab4
{
    /* Hough transform for circles. */
    class CircleHough : public ImgProcessor
    {
    public:
        /* List of parameters of type double. */
        enum class DoubleParam
        {
            res,
            dist,
            th1,
            th2,
            tot
        };
        /* List of parameters of type int. */
        enum class IntParam
        {
            min_r,
            max_r,
            tot
        };
        /* List of parameters that define a circle. */
        enum class CircleParam
        {
            x,
            y,
            r
        };

        /********** CONSTANTS **********/
        static constexpr int circle_intensity{ 255 }; // Intensity of circle drawing. 
        static constexpr int circle_thickness{ 2 }; // Thickness of circle lines.
        /* Number of parameters. */
        static constexpr int double_param_num{ static_cast<int>(DoubleParam::tot) };
        static constexpr int int_param_num{ static_cast<int>(IntParam::tot) };
        /* Maximum trackbar values. */
        static constexpr std::array double_param_max{ 10, 20, 50, 50 };
        static constexpr std::array int_param_max{ 20, 20 };
        /* Starting trackbar values. */
        static constexpr std::array double_param_start_vals{ 0, 0, 5, 11 };
        static constexpr std::array int_param_start_vals{ 1, 4 };
        /* Normalisation ranges. */
        static constexpr std::array double_param_limits
        {
            std::pair{ 1.0, 10.0 },
            std::pair{ 1.0, 100.0 },
            std::pair{ 10.0, 50.0 },
            std::pair{ 10.0, 50.0 }
        };
        static constexpr std::array int_param_limits
        {
            std::pair{ 1, 20 },
            std::pair{ 5, 25 }
        };
        /* Trackbar names. */
        static constexpr std::array<std::string_view, double_param_num> double_param_names
        {
            "Res",
            "Min Dist",
            "Th1",
            "Th2"
        };
        static constexpr std::array<std::string_view, int_param_num> int_param_names
        {
            "MinR",
            "MaxR"
        };

        /********** CONSTRUCTOR **********/
        CircleHough(const cv::Mat& img, std::string_view winName);

        /********** METHODS **********/
        /* Return the circle with the highest score, if there is one. */
        [[nodiscard]] std::optional<cv::Vec3f> getRelevantCircle();
        bool update(bool force = false); // Update the result image and display it.

    private:
        /********** METHODS **********/
        void drawCircles_(); // Draw all detected circles.
        /* Get current trackbar values and fit them into their normalisation ranges. */
        void fetchNormalisedParams_();

        /********** VARIABLES **********/
        cv::Mat greyScaleImg_; // Greyscale input image.
        std::vector<cv::Vec3f> circles_; // List of detected circles.
        /* Parameter values. */
        std::array<double, double_param_num> double_params_
        {
            double_param_limits[static_cast<int>(DoubleParam::res)].first,
            double_param_limits[static_cast<int>(DoubleParam::dist)].first,
            double_param_limits[static_cast<int>(DoubleParam::th1)].first,
            double_param_limits[static_cast<int>(DoubleParam::th2)].first
        };
        std::array<int, int_param_num> int_params_
        {
            int_param_limits[static_cast<int>(IntParam::min_r)].first,
            int_param_limits[static_cast<int>(IntParam::max_r)].first
        };
    };
}

#endif