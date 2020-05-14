#ifndef STREETANALISER_HPP
#define STREETANALISER_HPP

#include "Canny.hpp"
#include "CircleHough.hpp"
#include "LineHough.hpp"
#include "lab4.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <string_view>
#include <utility>

namespace lab4
{
    /* Detector for a street lane and a road sign. */
    class StreetAnaliser
    {
    public:
        /* Tuning phases for each shape. */
        enum class TuningPhase
        {
            canny,
            hough
        };

        /********** CONSTANTS **********/
        static constexpr int lane_win_num{ 2 };  // Number of tuning windows for lane detection.
        static constexpr int frame_delay{ 50 }; // Update delay for each frame.
        static constexpr int lane_colour_intensity{ 255 }; // Colour intensity for the lane.
        static constexpr int sign_colour_intensity{ 255 }; // Colour intensity for the sign.

        /* Preprocessing parameters. */
        static constexpr int gauss_width{ 9 };
        static constexpr int gauss_height{ 9 };
        static constexpr double gauss_sigma{ 3.5 };

        /* Name of the output window. */
        static constexpr std::string_view output_win_name{ "Result" };
        /* Names of lane tuning windows. */
        static constexpr std::array<std::string_view, lane_win_num> lane_win_names
        {
            "Lane tuning - Canny detector",
            "Lane tuning - Hough transform"
        };
        /* Name of the sign tuning window. */
        static constexpr std::string_view sign_win_name{ "Sign tuning." };

        /********** CONSTRUCTOR **********/
        explicit StreetAnaliser(cv::Mat img);

        /********** METHODS **********/
        void run(); // Run the analyser.

    private:
        /********** METHODS **********/
        void drawLane_(); // Draw the detected lane.
        void drawSign_(); // Draw the detected sign.
        void update_(); // Update the processing pipeline.

        /********** VARIABLES **********/
        const cv::Mat srcImg_; // Original image.
        cv::Mat img_; // Pre-processed image.
        cv::Mat outputImg_; // Result image.
        Window outputWin_; // Output window.
        /* Lane detector. */
        Canny lineCanny_;
        LineHough lineHough_;
        /* Sign detector. */
        CircleHough circleHough_;
    };
}

#endif