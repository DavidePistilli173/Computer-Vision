#include "CircleHough.hpp"

#include <opencv2/imgproc.hpp>

using namespace lab4;

CircleHough::CircleHough(const cv::Mat& img, std::string_view winName) :
    ImgProcessor(img, winName, CV_8UC3)
{
    bool success{ true };

    /* Add trackbars for parameters of type double. */
    for (int i = 0; i < static_cast<int>(DoubleParam::tot); ++i)
    {
        success = 
            success && win_.addTrackBar(double_param_names[i], double_param_start_vals[i], double_param_max[i]);
    }
        
    /* Add trackbars for parameters of type int. */
    for (int i = 0; i < static_cast<int>(IntParam::tot); ++i)
    {
        success = success && win_.addTrackBar(int_param_names[i], int_param_start_vals[i], int_param_max[i]);
    }

    if (!success) Log::error("Failed to create enough trackbars for all parameters.");
}

std::optional<cv::Vec3f> CircleHough::getRelevantCircle()
{
    if (circles_.empty()) return std::optional<cv::Vec3f>();
    return std::optional(circles_[0]);
}

bool CircleHough::update(bool force)
{
    /* If the update is not forced and if no trackbars were changed don't perform any computation. */
    if (!force && !win_.modified()) return false;

    /* Convert the input image to greyscale. */
    cv::cvtColor(srcImg_, greyScaleImg_, cv::COLOR_BGR2GRAY);
    fetchNormalisedParams_(); // Get the current parameter values.
    circles_.clear(); // Clear the result from previous computations.

    Log::info("Computing HoughCircles...");
    cv::HoughCircles(
        greyScaleImg_,
        circles_,
        cv::HOUGH_GRADIENT,
        double_params_[static_cast<int>(DoubleParam::res)],
        double_params_[static_cast<int>(DoubleParam::dist)],
        double_params_[static_cast<int>(DoubleParam::th1)],
        double_params_[static_cast<int>(DoubleParam::th2)],
        int_params_[static_cast<int>(IntParam::min_r)],
        int_params_[static_cast<int>(IntParam::max_r)]
    );
    Log::info("SUCCESS");
    resultImg_ = srcImg_.clone();
    Log::info("Drawing circles...");
    drawCircles_(); // Draw detected circles.
    Log::info("SUCCESS");
    win_.showImg(resultImg_);
        
    return true;
}

void CircleHough::drawCircles_()
{
    for (const auto& circle : circles_)
    {
        cv::circle(
            resultImg_, 
            cv::Point
            { 
                cvRound(circle[static_cast<int>(CircleParam::x)]), 
                cvRound(circle[static_cast<int>(CircleParam::y)]) 
            }, 
            cvRound(circle[static_cast<int>(CircleParam::r)]),
            cv::Scalar{ 0, circle_intensity, 0 }, 
            circle_thickness,
            cv::LINE_AA
        );
    }
}

void CircleHough::fetchNormalisedParams_()
{
    Log::info("Normalising CirleHough parameters.");
    if (
        std::vector<int> params = win_.fetchTrckVals();
        params.size() >= static_cast<int>(DoubleParam::tot) + static_cast<int>(IntParam::tot)
    )
    {
        /* Normalise parameters of type double. */
        for (int i = 0; i < double_param_num; ++i)
        {
            double_params_[i] = scale(
                double_param_limits[i].first,
                double_param_limits[i].second,
                scaling_coeff(params[i], double_param_max[i])
            );
            Log::info("Parameter %d = %f.", i, double_params_[i]);
        }

        /* Normalise parameters of type int. */
        for (int i = 0; i < int_param_num; ++i)
        {
            int_params_[i] = scale(
                int_param_limits[i].first,
                int_param_limits[i].second,
                scaling_coeff(params[i + double_param_num], int_param_max[i])
            );
            Log::info("Parameter %d = %d.", i + double_param_num, int_params_[i]);
        }
    }
    else
    {
        Log::error("Not enough parameters available.");
    }
}