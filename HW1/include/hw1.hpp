#ifndef HW1_HPP
#define HW1_HPP

#include <opencv2/core.hpp>

namespace hw1
{
    /********** TYPE ALIASES **********/
    using vec3uc_t = cv::Vec<uchar, 3>;
    using vec3ui_t = cv::Vec<uint, 3>;

    /********** ENUMS **********/
    /* OpenCV uses BGR as default for colour images. */
    enum class RGB
    {
        B,
        G,
        R,
        TOT
    };
    enum class HSV
    {
        H,
        S,
        V,
        TOT
    };
    /* Keycodes. */
    enum class Key
    {
        ESC = 27
    };

    /********** STRUCTS **********/
    /* Minimum and maximum values that define a range. */
    template <typename T>
    struct Range
    {
        constexpr Range() {}
        constexpr Range(const T a, const T b) :
            min{ a }, max{ b }
        {}

        T min, max;
    };

    /********** FUNCTIONS **********/
    /* Compute base^exponent at compile time. */
    template <typename T>
    constexpr T pow(T base, unsigned int exponent)
    {
        if (exponent == 0) return 1;
        return base * pow(base, exponent - 1);
    }
}

#endif