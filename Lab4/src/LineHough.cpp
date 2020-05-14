#include "LineHough.hpp"

#include <opencv2/imgproc.hpp>

using namespace lab4;

LineHough::LineHough(const cv::Mat& img, std::string_view winName) :
    ImgProcessor(img, winName, CV_8UC3)
{
    bool success{ true };
    /* Add trackbars for parameter tuning. */
    for (int i = 0; i < static_cast<int>(Param::tot); ++i)
    {
        success = success && win_.addTrackBar(param_names[i], param_start_vals[i], param_max[i]);
    }

    if (!success) Log::error("Failed to create enough trackbars for all parameters.");
}

bool LineHough::update(bool force)
{
    /* If the update is not forced and if no trackbars were changed don't perform any computation. */
    if (!force && !win_.modified()) return false;

    fetchNormalisedParams_(); // Get current parameter values. 
    preprocessInput_(); // Performing some pre-processing on the input.

    Log::info("Applying HoughLines...");
    cv::HoughLines(processedInputImg_, lines_, dist_res_, ang_res_, th_);
    Log::info("SUCCESS");
    /* The output image will be RGB. */
    cv::cvtColor(processedInputImg_, resultImg_, cv::COLOR_GRAY2BGR); // Reset the result image.
    Log::info("Drawing lines...");
    drawLines_(); // Draw detected lines.
    Log::info("SUCCESS");
    win_.showImg(resultImg_);

    return true;
}

void LineHough::drawLines_()
{        
    for (const auto& line : lines_)
    {
        float rho = line[static_cast<int>(LineParam::rho)];
        float theta = line[static_cast<int>(LineParam::theta)];
        double a = std::cos(theta);
        double b = std::sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;
        cv::Point pt1{ cvRound(x0 + 2 * resultImg_.cols * (-b)), cvRound(y0 + 2 * resultImg_.cols * (a)) };
        cv::Point pt2{ cvRound(x0 - 2 * resultImg_.cols * (-b)), cvRound(y0 - 2 * resultImg_.cols * (a)) };
        cv::line(resultImg_, pt1, pt2, cv::Scalar(0, 0, line_intensity), line_thickness, cv::LINE_AA);
    }
}

void LineHough::fetchNormalisedParams_()
{
    Log::info("Normalising LineHough parameters.");
    if (std::vector<int> params = win_.fetchTrckVals(); params.size() >= static_cast<int>(Param::tot))
    {
        /* Normalise the distance resolution. */
        dist_res_ = scale(
            dist_res_limits.first,
            dist_res_limits.second,
            scaling_coeff(params[static_cast<int>(Param::dist_res)], param_max[static_cast<int>(Param::dist_res)])
        );
        Log::info("Distance resolution = %f.", dist_res_);

        /* Normalise the angle resolution. */
        ang_res_ = scale(
            ang_res_limits.first,
            ang_res_limits.second,
            scaling_coeff(params[static_cast<int>(Param::ang_res)], param_max[static_cast<int>(Param::ang_res)])
        );
        Log::info("Angle resolution = %f.", ang_res_);

        /* Normalise the threshold. */
        th_ = scale(
            th_limits.first,
            th_limits.second,
            scaling_coeff(params[static_cast<int>(Param::th)], param_max[static_cast<int>(Param::th)])
        );
        Log::info("Threshold = %d.", th_);
    }
    else
    {
        Log::error("Not enough parameters available.");
    }
}

std::optional<std::pair<cv::Vec2f, cv::Vec2f>> LineHough::getRelevantLines() const
{
    if (lines_.size() <= 1) return std::optional<std::pair<cv::Vec2f, cv::Vec2f>>();
    return std::optional(std::pair{ lines_[0], lines_[1] });
}

void LineHough::preprocessInput_()
{
    cv::Mat structElem{ cv::getStructuringElement(cv::MORPH_RECT, cv::Size{dilate_width, dilate_height}) };
    cv::morphologyEx(srcImg_, processedInputImg_, cv::MORPH_DILATE, structElem);
}