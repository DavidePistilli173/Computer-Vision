#ifndef LAB5_HPP
#define LAB5_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <cstdio>
#include <mutex>
#include <string_view>
#include <vector>

namespace lab5
{
    /********** CONSTANTS **********/
    constexpr double pi{ 3.14159265358979 };

    /********** ENUMS **********/
    /* Keycodes. */
    enum class Key
    {
        esc = 27
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

    /* Basic wrapper for OpenCV windows. */
    class Window
    {
    public:
        /********** CONSTANTS **********/
        static constexpr size_t max_trckbar_num{ 8 }; // Maximum number of trackbars.

        /********** CONSTRUCTORS **********/
        explicit Window(std::string_view name);
        Window(const Window&) = delete;
        Window(Window&& win) = delete;
        ~Window();

        /********** OPERATORS **********/
        Window& operator=(const Window&) = delete;
        Window& operator=(Window&& win) = delete;

        /********** METHODS **********/
        bool addTrackBar(std::string_view name, int maxVal); // Add a trackbar with starting position 0.
        bool addTrackBar(std::string_view name, int startVal, int maxVal); // Add a trackbar.
        /* Fetch trackbar values if they changed since the last call. */
        [[nodiscard]] std::vector<int> fetchTrckVals(); // Return the current trackbar values.
        [[nodiscard]] bool modified() const; // Check whether the trackbar values changed.
        void showImg(const cv::Mat& img) const; // Show an image on the window.

    private:
        /********** METHODS **********/
        static void trckCallbck_(int val, void* ptr); // Trackbar callback.

        /********** VARIABLES **********/
        cv::String name_; // Name of the window.
        std::vector<int> trckBarVals_; // Values of the trackbars.
        bool trckModified_{ true }; // True if trackbar values changed since the last fetch.
    };
}

#endif