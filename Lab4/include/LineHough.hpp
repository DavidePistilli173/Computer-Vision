#ifndef LINEHOUGH_HPP
#define LINEHOUGH_HPP

#include "ImgProcessor.hpp"

#include <optional>

namespace lab4
{
    /* Hough transform for lines. */
    class LineHough : public ImgProcessor
    {
    public:
        /* List of parameters. */
        enum class Param
        {
            dist_res,
            ang_res,
            th,
            tot
        };
        /* Parameters that define a line. */
        enum class LineParam
        {
            rho,
            theta
        };
        /********** CONSTANTS **********/
        /* Preprocessing parameters. */
        static constexpr int dilate_width{ 3 };
        static constexpr int dilate_height{ 1 };
        /* Drawing parameters. */
        static constexpr int line_intensity{ 255 }; // Intensity of detected lines.
        static constexpr int line_thickness{ 2 }; // Thickness of line drawings.
        static constexpr int param_num{ static_cast<int>(Param::tot) }; // Number of parameters.
        static constexpr std::array param_max{ 20, 20, 100 }; // Maximum trackbar values.
        static constexpr std::array param_start_vals{ 0, 2, 60 }; // Starting trackbar values.
        /* Normalisation ranges. */
        static constexpr std::pair dist_res_limits{ 1.0, 20.0 };
        static constexpr std::pair ang_res_limits{ pi/32, pi };
        static constexpr std::pair th_limits{ 50, 200 };
        /* Trackbar names. */
        static constexpr std::array<std::string_view, param_num> param_names
        {
            "Dist Res",
            "Angle Res",
            "Th"
        };

        /********** CONSTRUCTOR **********/
        LineHough(const cv::Mat& img, std::string_view winName);

        /********** METHODS **********/
        /* Return the two most relevant lines from all detections, if present. */
        [[nodiscard]] std::optional<std::pair<cv::Vec2f, cv::Vec2f>> getRelevantLines() const;
        bool update(bool force = false); // Update the result if necessary.

    private:
        /********** METHODS **********/
        /* Draw the detected lines onto the result image. */
        void drawLines_();
        /* Normalise the parameters so that they fit into their normalisation ranges. */
        void fetchNormalisedParams_();
        /* Preprocess the input image. */
        void preprocessInput_();

        /********** VARIABLES **********/
        cv::Mat processedInputImg_; // Input image after pre-processing.
        std::vector<cv::Vec2f> lines_; // Detected lines.
        /* Parameter values. */
        double dist_res_{ dist_res_limits.first };
        double ang_res_{ ang_res_limits.first };
        int th_{ th_limits.first };
    };
}

#endif