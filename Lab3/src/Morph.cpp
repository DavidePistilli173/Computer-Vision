#include "Morph.hpp"

#include <opencv2/imgproc.hpp>

lab3::Morph::Morph(const cv::Mat& img) :
    img_{ img.clone() },
    result_{ img.clone() }
{
}

void lab3::Morph::execute()
{
    cv::Mat structElem{ cv::getStructuringElement(cv::MORPH_RECT, cv::Size{def_k_size, def_k_size}) };
    cv::morphologyEx(img_, result_, cv::MORPH_DILATE, structElem);
    cv::morphologyEx(result_, result_, cv::MORPH_CLOSE, structElem);
}

cv::Mat lab3::Morph::getResult() const
{
    return result_.clone();
}
