#ifndef LAB3_FILTERCOMPARATOR_HPP
#define LAB3_FILTERCOMPARATOR_HPP

#include "Filter.h"
#include "lab3.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <string_view>

namespace lab3
{
    class FilterComparator
    {
    public:
        /********** ENUMS **********/
        /* List of implemented filters. */
        enum class Filters
        {
            gaussian,
            median,
            bilateral,
            tot
        };
        /* Lists of parameters for the various filters. */
        enum class GaussParam
        {
            size,
            sigma,
            tot
        };
        enum class MedianParam
        {
            size,
            tot
        };
        enum class BilatParam
        {
            range_sigma,
            space_sigma,
            tot
        };

        /********** CONSTRUCTORS **********/
        explicit FilterComparator(const cv::Mat& img);

        /********** CONSTANTS **********/
        /* Names of the filters. */
        static constexpr std::array<std::string_view, static_cast<int>(Filters::tot)> filter_names
        {
            "Gaussian Filter",
            "Median Filter",
            "Bilateral Filter"
        };
        /* Maximum values for filter parameters. */
        static constexpr std::array filter_counts
        {
            std::array<int, 2>{ 11, 25 },
            std::array<int, 2>{ 11, -1 },
            std::array<int, 2>{ 128, 25 }
        };

        static constexpr int def_filter_size{ 3 }; // Default filter size.
        static constexpr int bilateral_filter_size{ 11 }; // Size of the bilateral filter.
        static constexpr float val_coeff{ 0.1F }; // Coefficient for spacial sigma parameters.
        /* Parameter names. */
        static constexpr std::string_view kernel_name{ "Kernel Size" };
        static constexpr std::string_view sigma_name{ "Sigma" };
        static constexpr std::string_view range_sigma_name{ "Range Sigma" };
        static constexpr std::string_view space_sigma_name{ "Space Sigma" };
        static constexpr int frame_delay{ 50 }; // Update delay for each frame.

        /********** METHODS **********/
        void run(); // Run the filter comparison.

    private:
        /********** METHODS **********/
        /* Trackbar callbacks. */
        static void gaussianCallBack_(int val, void* ptr);
        static void medianCallBack_(int val, void* ptr);
        static void bilateralCallBack_(int val, void* ptr);

        void initTrackBars_(); // Initialise the trackbars.

        /********** VARIABLES **********/
        std::array<Window, static_cast<int>(Filters::tot)> windows_; // Display windows.
        cv::Mat img_; // Input image.
        /* Filters. */
        GaussianFilter gaussianFilter_;
        MedianFilter medianFilter_;
        BilateralFilter bilateralFilter_;
        /* Current parameter values for each filter. */
        std::array<int, 2> gaussVals_{ def_filter_size, 0 };
        int medVal_{ 0 };
        std::array<int, 2> biVals_{ 0, 0 };
    };
}

#endif