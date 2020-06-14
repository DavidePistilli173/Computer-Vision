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
         gauss_size,    // Size of the Gaussian filter.
         gauss_sig,     // Sigma of the Gaussian filter.
         bil_size,      // Size of the bilateral filter.
         bil_col_sig,   // Colour sigma of the bilateral filter.
         bil_space_sig, // Space sigma of the bilateral filter.
         canny_th1,     // Threshold 1 for Canny.
         canny_th2,     // Threshold 2 for Canny.
         dist_th,       // Threshold for the distance transform result.
         tot            // Total number of parameters.
      };
      // Analysis parameters.
      enum class AParam
      {
         tot
      };

      /********** CONSTANTS **********/
      static constexpr double score_th{ 0.5 };   // Score threshold for histogram comparison.
      static const cv::Size   analysis_res;      // Resolution used to analyse the image.
      static const cv::Scalar tree_colour;       // Colour of the tree box.
      static constexpr int    pyr_children{ 4 }; // Number of children per cell.
      static constexpr int    pyr_depth{ 4 };    // Depth of the analysis grid.

      /********** CONSTRUCTOR **********/
      TreeDetector() = default;
      explicit TreeDetector(std::string_view bowFile);

      /********** METHODS **********/
      // Detect trees and output the result.
      cv::Mat detect(const cv::Mat& input);

   private:
      /********** METHODS **********/
      // Add all candidate trees in the subtree that starts from node.
      void addCandidateTrees_(ImagePyramid<pyr_children, pyr_depth>::Cell* node);
      // Analyse the preprocessed data to find trees.
      bool analyse_(std::array<param, static_cast<int>(AParam::tot)>& params);
      // Compute the score of an image wrt a reference histogram.
      double computeScore(
         cv::BOWImgDescriptorExtractor& extractor,
         cv::Ptr<cv::xfeatures2d::SIFT> sift,
         const cv::Mat&                 img,
         const cv::Mat&                 referenceHist);
      // Draw the final result.
      bool drawResult_();
      // Preprocess the input image.
      bool preProcess_(std::array<param, static_cast<int>(PParam::tot)>& params);

      /********** VARIABLES **********/
      cv::Mat                               treeVocabulary_;                   // BOW vocabulary for trees.
      cv::Mat                               nonTreeVocabulary_;                // BOW vocabulary for non-trees.
      cv::Mat                               avgTreeHist_;                      // Average tree word histogram.
      cv::Mat                               avgNonTreeHist_;                   // Average non-tree word histogram.
      Image                                 resizedInput_;                     // Resized input image.
      std::vector<Image>                    processedImgs_;                    // Images after pre-processing.
      Image                                 result_;                           // Final result.
      std::vector<Rect<int>>                preliminaryTrees_;                 // All regions containing trees.
      std::vector<Rect<int>>                trees_;                            // Final trees in the image.
      std::pair<float, float>               scale_{ 1.0F, 1.0F };              // Scaling factor of the image.
      ImagePyramid<pyr_children, pyr_depth> pyramid_{ img_width, img_height }; // Analysis grid.
   };
} // namespace prj

#endif