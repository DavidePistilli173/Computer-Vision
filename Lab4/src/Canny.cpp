#include "Canny.hpp"

#include <opencv2/imgproc.hpp>

using namespace lab4;

Canny::Canny(const cv::Mat& img, std::string_view winName) :
    ImgProcessor(img, winName, CV_8UC1)
{
    /* Add trackbars for parameter tuning. */
    bool success{ win_.addTrackBar(param_names[static_cast<int>(Param::th1)], param_start_vals.first, param_max) };
    success = 
        success && win_.addTrackBar(param_names[static_cast<int>(Param::th2)], param_start_vals.second, param_max);
    if (!success) Log::error("Failed to create enough trackbars for all parameters.");
}

bool Canny::update()
{
    /* If no trackbars were changed, don't perform any computation. */
    if (!win_.modified()) return false;

    fetchNormalisedParams_(); // Get the updated parameters.
    Log::info("Applying Canny edge detector...");
    cv::Canny(srcImg_, resultImg_, th1_, th2_);
    Log::info("SUCCESS");
    win_.showImg(resultImg_); // Display the result.
    return true;
}

void Canny::fetchNormalisedParams_()
{
    Log::info("Normalising Canny parameters.");
    if (std::vector<int> params = win_.fetchTrckVals(); params.size() >= static_cast<int>(Param::tot))
    {

        th1_ = scale(
            param_limits.first,
            param_limits.second,
            scaling_coeff(params[static_cast<int>(Param::th1)], param_max)
        );
        Log::info("Threshold 1 = %d", th1_);

        th2_ = scale(
            param_limits.first,
            param_limits.second,
            scaling_coeff(params[static_cast<int>(Param::th2)], param_max)
        );
        Log::info("Threshold 2 = %d", th2_);
    }
    else
    {
        Log::error("Not enough parameters available.");
    }
}