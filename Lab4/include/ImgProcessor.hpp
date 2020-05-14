#ifndef IMGPROCESSOR_HPP
#define IMGPROCESSOR_HPP

#include "lab4.hpp"

#include <opencv2/core.hpp>

#include <string_view>

namespace lab4
{
    /* Generic image processor with a window. */
    class ImgProcessor
    {
    public:
        ImgProcessor(const cv::Mat& img, std::string_view winName, int resultType);

        [[nodiscard]] const cv::Mat& getResultRef() const; // Return the result image.
        void show() const; // Show the result on the window.

    protected:
        const cv::Mat& srcImg_;
        cv::Mat resultImg_;
        lab4::Window win_;
    };
}

#endif