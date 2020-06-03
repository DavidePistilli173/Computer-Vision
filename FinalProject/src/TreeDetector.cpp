#include "TreeDetector.hpp"

#include "utility.hpp"

using namespace prj;

cv::Mat TreeDetector::detect(cv::Mat input) {
   Log::info("Preprocessing.");
   if (!preProcess_()) {
      Log::error("Preprocessing failed.");
      return cv::Mat{};
   }

   Log::info("Analysing data.");
   if (!analyse_()) {
      Log::error("Failed to analyse data.");
      return cv::Mat{};
   }

   Log::info("Drawing result.");
   if (!drawResult_()) {
      Log::error("Failed to draw the final result.");
      return cv::Mat{};
   }

   return result_;
}

bool TreeDetector::analyse_() {
   return true;
}

bool TreeDetector::drawResult_() {
   return true;
}

bool TreeDetector::preProcess_() {
   return true;
}
