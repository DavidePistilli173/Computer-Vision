#include "ImgProcessor.hpp"

using namespace lab4;

ImgProcessor::ImgProcessor(const cv::Mat& img, std::string_view winName, int resultType) :
    srcImg_{ img },
    resultImg_{ cv::Mat::zeros(img.size(), resultType) },
    win_{ winName }
{}

const cv::Mat& ImgProcessor::getResultRef() const
{
    return resultImg_;
}

void ImgProcessor::show() const
{
    win_.showImg(resultImg_);
}
