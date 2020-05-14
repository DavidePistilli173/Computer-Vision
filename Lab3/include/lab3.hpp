#ifndef LAB3_HPP
#define LAB3_HPP

#include <opencv2/core.hpp>

#include <cstdio>
#include <mutex>
#include <string_view>

namespace lab3
{
    /********** ENUMS **********/
    /* Keycodes. */
    enum class Key
    {
        esc = 27
    };
    /* RGB colour components. */
    enum class RGB
    {
        b,
        g,
        r,
        tot
    };
    /* HSV colour components. */
    enum class HSV
    {
        h,
        s,
        v,
        tot
    };

    /********** CLASSES **********/
    /* Basic console logging functions. */
    class Log
    {
    public:
        template <typename... Args>
        static void info(std::string_view msg, Args... args)
        {
            std::scoped_lock<std::mutex> lck(mtx_);
            std::printf("[INFO] - ");
            std::printf(msg.data(), args...);
            std::printf("\n");
        }
        template <typename... Args>
        static void warn(std::string_view msg, Args... args)
        {
            std::scoped_lock<std::mutex> lck(mtx_);
            std::printf("[WARN] - ");
            std::printf(msg.data(), args...);
            std::printf("\n");
        }
        template <typename... Args>
        static void error(std::string_view msg, Args... args)
        {
            std::scoped_lock<std::mutex> lck(mtx_);
            std::printf("[ERROR] - ");
            std::printf(msg.data(), args...);
            std::printf("\n");
        }
        template <typename... Args>
        static void fatal(std::string_view msg, Args... args)
        {
            std::scoped_lock<std::mutex> lck(mtx_);
            std::printf("[FATAL] - ");
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
        /********** CONSTRUCTORS **********/
        explicit Window(std::string_view name);
        Window(const Window&) = delete;
        Window(Window&& win) = delete;
        ~Window();

        /********** OPERATORS **********/
        Window& operator=(const Window&) = delete;
        Window& operator=(Window&& win) = delete;

        /********** METHODS **********/
        void showImg(const cv::Mat& img) const;

    private:
        cv::String name_; // Name of the window.
    };
}

#endif