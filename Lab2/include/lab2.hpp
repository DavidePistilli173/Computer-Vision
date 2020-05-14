#ifndef LAB2_HPP
#define LAB2_HPP

#include <cstdio>
#include <limits>
#include <mutex>
#include <opencv2/core.hpp>
#include <string_view>

namespace lab2
{
    /********** STRUCTS **********/
    /* Constexpr version of cv::Size. */
    template <typename T>
    struct Size
    {
        constexpr Size() = default;
        constexpr Size(const T a, const T b) :
            w{ a }, h{ b }
        {}

        T w, h;
    };
    /* Data related to camera calibration. */
    struct CalibrationData
    {
        void reset(); // Clear all data.
        
        cv::Mat camera{ cv::Mat::eye(3, 3, CV_64F) }; // Camera matrix.
        std::vector<float> distParams; // Distortion parameters.
        std::vector<cv::Mat> rotVecs; // Rotation vectors.
        std::vector<cv::Mat> tVecs; // Translation vectors.
    };
    struct CalibrationStats
    {
        /* Reprojection error of a specific image. */
        struct ImageError
        {
            double err; // Reprojection error of the image.
            size_t indx; // Index of the image.
        };

        double meanRE{ 0.0 }; // Mean reprojection error.
        ImageError minRE{ std::numeric_limits<double>::infinity(), 0 }; // Minimum reprojection error.
        ImageError maxRE{ 0.0, 0 }; // Maximum reprojection error.
    };

    /********** CLASSES **********/
    /* Basic console logging functions. */
    class Log
    {
    public:
        /********** CONSTANTS **********/
        static constexpr const char* asterisks =
            "********************************************************************************";
        static constexpr const char* spaces =
            "                                                                                ";
        static constexpr const char* hyphens =
            "--------------------------------------------------------------------------------";

        /********** METHODS **********/
        template <typename... Args>
        static void info(std::string_view msg, Args... args)
        {
            std::scoped_lock lck(mtx_);
            std::printf("\033[1;37m[INFO] - ");
            std::printf(msg.data(), args...);
            std::printf("\n");
        }
        template <typename... Args>
        static void warn(std::string_view msg, Args... args)
        {
            std::scoped_lock lck(mtx_);
            std::printf("\033[0;33m[WARN] - ");
            std::printf(msg.data(), args...);
            std::printf("\n");
        }
        template <typename... Args>
        static void error(std::string_view msg, Args... args)
        {
            std::scoped_lock lck(mtx_);
            std::printf("\033[0;31m[ERROR] - ");
            std::printf(msg.data(), args...);
            std::printf("\n");
        }
        template <typename... Args>
        static void fatal(std::string_view msg, Args... args)
        {
            std::scoped_lock lck(mtx_);
            std::printf("\033[0;35m[FATAL] - ");
            std::printf(msg.data(), args...);
            std::printf("\n");
        }

    private:
        static std::mutex mtx_;
    };
}

#endif