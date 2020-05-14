#ifndef CANNY_HPP
#define CANNY_HPP

#include "ImgProcessor.hpp"

#include <utility>

namespace lab4
{
    /* Canny edge detector. */
    class Canny : public ImgProcessor
    {
    public:
        /* Parameter list. */
        enum class Param
        {
            th1,
            th2,
            tot
        };

        /********** CONSTANTS **********/
        /* Parameters. */
        static constexpr int param_num{ static_cast<int>(Param::tot) }; // Number of parameters.
        static constexpr int param_max{ 50 }; // Maximum trackbar value.
        static constexpr std::pair param_start_vals{ 28, 31 }; // Starting trackbar values.
        static constexpr std::pair param_limits{ 20, 200 }; // Normalisation range.
        /* Trackbar names. */
        static constexpr std::array<std::string_view, param_num> param_names
        {
            "Th 1",
            "Th 2"
        };

        /********** CONSTRUCTOR **********/
        Canny(const cv::Mat& img, std::string_view winName);

        /********** METHODS **********/
        bool update(); // Update the result image and display it.

    private:
        /********** METHODS **********/
         /* Get the current trackbar values and fit them into the range specified by param_limits. */
        void fetchNormalisedParams_();

        /********** VARIABLES **********/
        /* Parameter values. */
        int th1_{ param_limits.first };
        int th2_{ param_limits.first };
    };
}

#endif