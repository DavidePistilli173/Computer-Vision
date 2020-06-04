#ifndef TREEDETECTOR_HPP
#define TREEDETECTOR_HPP

#include "utility.hpp"

#include <array>
#include <opencv2/core.hpp>
#include <variant>

namespace prj
{
   // Detect and locate trees in an image.
   class TreeDetector
   {
   public:
      /********** ENUMS **********/
      // Preprocessing parameters.
      enum class PParam
      {
         bi_size,     // Size of the bilateral filter.
         bi_colour_s, // Colour sigma for the bilateral filter.
         bi_space_s,  // Space sigma for the bilateral filter.
         canny_th1,   // Threshold 1 for Canny.
         canny_th2,   // Threshold 2 for Canny.
         tot          // Total number of parameters.
      };
      // Analysis parameters.
      enum class AParam
      {
         tot
      };

      /********** CONSTANTS **********/
      static const cv::Size analysis_res; // Resolution used to analyse the image.

      /********** CONSTRUCTOR **********/
      TreeDetector() = default;

      /********** METHODS **********/
      // Detect trees and output the result.
      cv::Mat detect(const cv::Mat& input);

   private:
      /********** METHODS **********/
      // Analyse the preprocessed data to find trees.
      bool analyse_(std::array<param, static_cast<int>(AParam::tot)>& params);
      // Draw the final result.
      bool drawResult_();
      // Preprocess the input image.
      bool preProcess_(std::array<param, static_cast<int>(PParam::tot)>& params);

      /********** VARIABLES **********/
      Image resizedInput_; // Resized input image.
      Image result_;       // Final result.
   };
} // namespace prj

#endif