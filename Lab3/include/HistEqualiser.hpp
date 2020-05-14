#ifndef LAB3_HISTEQUALISER_HPP
#define LAB3_HISTEQUALISER_HPP

#include "lab3.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <string_view>

namespace lab3
{
    class HistEqualiser
    {
    public:
        /* Operation mode for histogram computations. */
        enum class Mode
        {
            rgb,
            hsv
        };

        /********** CONSTRUCTOR **********/
        explicit HistEqualiser(const cv::Mat& img);

        /********** CONSTANTS **********/
        static constexpr int channel_num{ 3 }; // Number of channels for the input image.
        static constexpr int hist_channel{ 0 }; // Channel to use for histogram computation.
        /* RGB specific constants. */
        static constexpr int rgb_max_val{ 255 }; // Maximum intensity value for the histogram.
        static constexpr int rgb_bins{ rgb_max_val + 1 }; //  Number of bins for the histogram.
        static constexpr std::array<float, 2> rgb_range{ 0, rgb_bins }; // Range for the histogram's bins.
        /* HSV specific constants. */
        static constexpr std::array<int, channel_num> hsv_bins{ 180, 256, 256 };
        static constexpr std::array<std::array<float, 2>, channel_num> hsv_ranges
        {
            std::array<float, 2>{ 0, hsv_bins[static_cast<int>(HSV::h)] },
            std::array<float, 2>{ 0, hsv_bins[static_cast<int>(HSV::s)] },
            std::array<float, 2>{ 0, hsv_bins[static_cast<int>(HSV::v)] }
        };

        /* Window names. */
        static constexpr std::string_view img_win_name{ "Image" }; // Source image.
        static constexpr std::array<std::string_view, channel_num> hist_win_names{ "Blue", "Green", "Red" }; // Histograms.

        /********** METHODS **********/
        void computeHists();
        void equalise(HSV channel = HSV::v);
        cv::Mat getResult() const;
        void reset();
        void setMode(Mode mode);
        void show() const;

    private:
        const cv::Mat src_; // Source image.
        cv::Mat img_; // Image that will be modified.
        std::array<cv::Mat, channel_num> channels_; // Separate channels of the image.
        std::array<cv::Mat, channel_num> histograms_; // Histograms for each channel.
        const float* rgb_ranges_{ rgb_range.data() }; // Array of ranges for RGB channels.
        /* Array of ranges for HSV channels. */
        std::array<const float*, channel_num> hsv_ranges_
        {
            hsv_ranges[static_cast<int>(HSV::h)].data(),
            hsv_ranges[static_cast<int>(HSV::s)].data(),
            hsv_ranges[static_cast<int>(HSV::v)].data()
        };
        Mode mode_{ Mode::rgb }; // Current colour mode.
    };
}

#endif