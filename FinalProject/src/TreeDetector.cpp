#include "TreeDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <thread>

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
      cv::Size{ 9, 9 },
      3.5,
      29,
      100.0,
      100.0,
      0.076F,
      30.0,
      60.0,
      0.45
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

   filteredImg.gaussianFilter(
      std::get<cv::Size>(params[static_cast<int>(PParam::gauss_size)]),
      std::get<double>(params[static_cast<int>(PParam::gauss_sig)]));

   filteredImg.bilateralFilter(
      std::get<int>(params[static_cast<int>(PParam::bil_size)]),
      std::get<double>(params[static_cast<int>(PParam::bil_col_sig)]),
      std::get<double>(params[static_cast<int>(PParam::bil_space_sig)]));

   if constexpr (debug) filteredImg.display();

   Image equalisedFilteredImg{ filteredImg };

   filteredImg.segment(
      std::get<double>(params[static_cast<int>(PParam::canny_th1)]),
      std::get<double>(params[static_cast<int>(PParam::canny_th2)]),
      std::get<double>(params[static_cast<int>(PParam::dist_th)]));

   equalisedFilteredImg.equaliseHistogram();
   equalisedFilteredImg.segment(
      std::get<double>(params[static_cast<int>(PParam::canny_th1)]),
      std::get<double>(params[static_cast<int>(PParam::canny_th2)]),
      std::get<double>(params[static_cast<int>(PParam::dist_th)]));

   std::thread t1{ [filteredImg]() { filteredImg.display("Not equalised", true); } };
   std::thread t2{ [equalisedFilteredImg]() { equalisedFilteredImg.display("Equalised", true); } };
   resizedInput_.display(std::string_view{ "Input" });
   t1.join();
   t2.join();

   return true;
}
