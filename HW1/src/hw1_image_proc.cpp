#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <thread>

#include "hw1_image_proc.hpp"

using namespace hw1;

HW1::HW1(const cv::Mat& input) :
    srcImg_{ input.clone() },
    rgbImg_{ input.clone() },
    hsvImg_{ input.clone() }
{
}

void HW1::run()
{
    /* Create the windows that the program will use. */
    cv::namedWindow(INPUT_WIN_NAME.data());
    cv::namedWindow(RGB_WIN_NAME.data());
    cv::namedWindow(HSV_WIN_NAME.data());
    /* Set the mouse callback for the main window. */
    cv::setMouseCallback(INPUT_WIN_NAME.data(), handleMouseEvents_, this);

    bool done = false;
    while (!done)
    {
        /* Render the images until ESC is pressed. */
        cv::imshow(HSV_WIN_NAME.data(), hsvImg_);
        /* If the data is ready, render it. */
        std::unique_lock<std::mutex> lock(imgProcMutex_);
        cv::imshow(RGB_WIN_NAME.data(), rgbImg_);
        cv::imshow(INPUT_WIN_NAME.data(), srcImg_);
        lock.unlock();
        
        if (cv::waitKey(RENDER_DELAY) == static_cast<int>(hw1::Key::ESC)) done = true;
    }
}

bool HW1::isInRGBRange_(const hw1::vec3uc_t& colour, const std::vector<hw1::Range<uchar>>& range)
{
    if (/* Check the blue component. */
        colour[static_cast<int>(RGB::B)] >= range[static_cast<int>(RGB::B)].min &&
        colour[static_cast<int>(RGB::B)] <= range[static_cast<int>(RGB::B)].max &&
        /* Check the green component. */
        colour[static_cast<int>(RGB::G)] >= range[static_cast<int>(RGB::G)].min &&
        colour[static_cast<int>(RGB::G)] <= range[static_cast<int>(RGB::G)].max &&
        /* Check the red component. */
        colour[static_cast<int>(RGB::R)] >= range[static_cast<int>(RGB::R)].min &&
        colour[static_cast<int>(RGB::R)] <= range[static_cast<int>(RGB::R)].max) return true;
    return false;
}

bool HW1::isInHRange_(const hw1::vec3uc_t& colour, const hw1::Range<uchar> range)
{
    if (/* Check the hue component. */
        colour[static_cast<int>(HSV::H)] >= range.min &&
        colour[static_cast<int>(HSV::H)] <= range.max) return true;
    return false;
}

void HW1::handleMouseEvents_(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        HW1* this_ = reinterpret_cast<HW1*>(userdata);
        std::scoped_lock<std::mutex> lock{ this_->imgProcMutex_ };
        std::thread rgb{ [this_, x, y]() { this_->processRGB_({ x, y }); } };
        std::thread hsv{ [this_, x, y]() { this_->processHSV_({ x, y }); } };

        rgb.join();
        hsv.join();
    }
}

void HW1::processRGB_(cv::Point2i target)
{
    rgbImg_ = srcImg_.clone();

    int startY{ target.y + TARGET_NEIGHBORHOOD.min };
    int endY{ target.y + TARGET_NEIGHBORHOOD.max };
    int startX{ target.x + TARGET_NEIGHBORHOOD.min };
    int endX{ target.x + TARGET_NEIGHBORHOOD.max };
    if (startY < 0 || endY >= rgbImg_.rows || startX < 0 || endX > rgbImg_.cols) return;

    /* Compute the average RGB colour around the target. */
    vec3ui_t avgRGB{0,0,0};
    for (int i = startY; i <= endY; ++i)
    {
        for (int j = startX; j <= endX; ++j)
        {
            avgRGB[static_cast<int>(RGB::B)] += rgbImg_.at<vec3uc_t>(i, j)[static_cast<int>(RGB::B)];
            avgRGB[static_cast<int>(RGB::G)] += rgbImg_.at<vec3uc_t>(i, j)[static_cast<int>(RGB::G)];
            avgRGB[static_cast<int>(RGB::R)] += rgbImg_.at<vec3uc_t>(i, j)[static_cast<int>(RGB::R)];
        }
    }
    avgRGB /= PIXEL_NUM;

    /* Compute the absolute difference between the average and the threshold values. */
    const std::vector<uint> deltas
    {
        static_cast<unsigned int>(COLOUR_THRESHOLD * avgRGB[static_cast<int>(RGB::B)]),
        static_cast<unsigned int>(COLOUR_THRESHOLD * avgRGB[static_cast<int>(RGB::G)]),
        static_cast<unsigned int>(COLOUR_THRESHOLD * avgRGB[static_cast<int>(RGB::R)])
    };
    /* Compute the effective relevant value ranges. */
    const std::vector<Range<uchar>> thresholds
    {
        /* Blue range. */
        Range<uchar>
        { 
            static_cast<uchar>(avgRGB[static_cast<int>(RGB::B)] - deltas[static_cast<int>(RGB::B)]),
            static_cast<uchar>(std::min(avgRGB[static_cast<int>(RGB::B)] + deltas[static_cast<int>(RGB::B)], 255u))
        },
        /* Green range. */
        Range<uchar>
        { 
            static_cast<uchar>(avgRGB[static_cast<int>(RGB::G)] - deltas[static_cast<int>(RGB::G)]),
            static_cast<uchar>(std::min(avgRGB[static_cast<int>(RGB::G)] + deltas[static_cast<int>(RGB::G)], 255u))
        },
        /* Red range. */
        Range<uchar>
        { 
            static_cast<uchar>(avgRGB[static_cast<int>(RGB::R)] - deltas[static_cast<int>(RGB::R)]),
            static_cast<uchar>(std::min(avgRGB[static_cast<int>(RGB::R)] + deltas[static_cast<int>(RGB::R)], 255u))
        }
    };

    /* Change all pixels in rgbImg_ that are within the threshold. */
    for (int i = 0; i < rgbImg_.rows; ++i)
    {
        for (int j = 0; j < rgbImg_.cols; ++j)
        {
            if (isInRGBRange_(rgbImg_.at<vec3uc_t>(i, j), thresholds))
            {
                rgbImg_.at<vec3uc_t>(i, j) =
                {
                    NEW_RGB_VAL[static_cast<int>(RGB::B)],
                    NEW_RGB_VAL[static_cast<int>(RGB::G)],
                    NEW_RGB_VAL[static_cast<int>(RGB::R)]
                };
            }
        }
    }
}

void HW1::processHSV_(cv::Point2i target)
{
    int startY{ target.y + TARGET_NEIGHBORHOOD.min };
    int endY{ target.y + TARGET_NEIGHBORHOOD.max };
    int startX{ target.x + TARGET_NEIGHBORHOOD.min };
    int endX{ target.x + TARGET_NEIGHBORHOOD.max };
    if (startY < 0 || endY >= hsvImg_.rows || startX < 0 || endX > rgbImg_.cols) return;

    /* Convert the input to HSV. */
    cv::cvtColor(srcImg_, hsvImg_, cv::COLOR_BGR2HSV);

    /* Compute the average Hue around the target. */
    int avgH{ 0 };
    for (int i = startY; i <= endY; ++i)
    {
        for (int j = startX; j <= endX; ++j)
        {
            avgH += hsvImg_.at<vec3uc_t>(i, j)[static_cast<int>(HSV::H)];
        }
    }
    avgH /= PIXEL_NUM;

    /* Compute the absolute difference between the average and the threshold value. */
    uint delta{ static_cast<unsigned int>(COLOUR_THRESHOLD * avgH) };
    /* Compute the effective relevant hue ranges. */
    Range<uchar> threshold
    { 
        static_cast<uchar>(avgH - delta),
        static_cast<uchar>(std::min(avgH + delta, 255u))
    };

    /* Change all pixels in rgbImg_ that are within the threshold. */
    for (int i = 0; i < hsvImg_.rows; ++i)
    {
        for (int j = 0; j < hsvImg_.cols; ++j)
        {
            if (isInHRange_(hsvImg_.at<vec3uc_t>(i, j), threshold))
            {
                hsvImg_.at<vec3uc_t>(i, j)[static_cast<int>(HSV::H)] = 
                    std::max(hsvImg_.at<vec3uc_t>(i, j)[static_cast<int>(HSV::H)] + HUE_OFFSET, 255);
            }
        }
    }

    /* Convert the processed image back to BGR. */
    cv::cvtColor(hsvImg_, hsvImg_, cv::COLOR_HSV2BGR);
}