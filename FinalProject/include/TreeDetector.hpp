#ifndef TREEDETECTOR_HPP
#define TREEDETECTOR_HPP

#include <opencv2/core.hpp>

namespace prj {
   // Detect and locate trees in an image.
   class TreeDetector {
   public:
      // Detect trees and output the result.
      cv::Mat detect(cv::Mat input);

   private:
      /********** METHODS **********/
      // Analyse the preprocessed data to find trees.
      bool analyse_();
      // Draw the final result.
      bool drawResult_();
      // Preprocess the input image.
      bool preProcess_();

      /********** VARIABLES **********/
      cv::Mat result_;
   };
} // namespace prj

#endif