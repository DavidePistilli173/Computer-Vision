#include "lab3.hpp"

#include <opencv2/highgui.hpp>

namespace lab3
{
    std::mutex Log::mtx_;

    Window::Window(std::string_view name) :
        name_{ name.data() }
    {
        cv::namedWindow(name.data(), cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
    }

    Window::~Window()
    {
        cv::destroyWindow(name_);
    }

    void Window::showImg(const cv::Mat& img) const
    {
        cv::imshow(name_, img);
    }
}