#include "TreeDetector.hpp"

#include <opencv2/imgproc.hpp>

using namespace prj;

const cv::Size TreeDetector::analysis_res{ 500, 500 };

cv::Mat TreeDetector::detect(const cv::Mat& input)
{
   Log::info("Resizing image.");
   // Used to convert tree coordinates back to the input's coordinate frame.
   std::pair scalingFactor{
      static_cast<float>(input.cols) / analysis_res.width,
      static_cast<float>(input.rows) / analysis_res.height
   };
   resizedInput_ = input;
   resizedInput_.resize(analysis_res);

   Log::info("Preprocessing.");
   std::array<param, static_cast<int>(PParam::tot)> pParams{
      13,
      250.0,
      100.0,
      90.0,
      120.0
   };
   if (!preProcess_(pParams))
   {
      Log::error("Preprocessing failed.");
      return cv::Mat{};
   }

   Log::info("Analysing data.");
   std::array<param, static_cast<int>(AParam::tot)> aParams;
   if (!analyse_(aParams))
   {
      Log::error("Failed to analyse data.");
      return cv::Mat{};
   }

   Log::info("Drawing result.");
   if (!drawResult_())
   {
      Log::error("Failed to draw the final result.");
      return cv::Mat{};
   }

   return result_.image();
}

bool TreeDetector::analyse_(std::array<param, static_cast<int>(AParam::tot)>& params)
{
   return true;
}

bool TreeDetector::drawResult_()
{
   return true;
}

bool TreeDetector::preProcess_(std::array<param, static_cast<int>(PParam::tot)>& params)
{
   Image filteredImg{ resizedInput_ };
   if constexpr (debug) filteredImg.display();

   filteredImg.bilateralFilter(
      std::get<int>(params[static_cast<int>(PParam::bi_size)]),
      std::get<double>(params[static_cast<int>(PParam::bi_colour_s)]),
      std::get<double>(params[static_cast<int>(PParam::bi_space_s)]));

   if constexpr (debug) filteredImg.display();

   filteredImg.canny(
      std::get<double>(params[static_cast<int>(PParam::canny_th1)]),
      std::get<double>(params[static_cast<int>(PParam::canny_th2)]));

   if constexpr (debug) filteredImg.display();

   return true;
}
