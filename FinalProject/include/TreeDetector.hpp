#ifndef TREEDETECTOR_HPP
#define TREEDETECTOR_HPP

#include "Image.hpp"
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
      /********** CONSTANTS **********/
      static constexpr double score_th{ 0.5 }; // Score threshold for histogram comparison.
      static const cv::Size   analysis_res;    // Resolution used to analyse the image.
      static const cv::Scalar tree_colour;     // Colour of the tree box.
      // Growth parameters.
      static constexpr int cell_w{ 200 };
      static constexpr int cell_h{ 200 };
      // Score thresholds for confirmed trees.
      static constexpr double base_threshold{ 0.025 };
      static constexpr double growth_th{ 0.85 };
      static constexpr float  overlap_th{ 0.3 };

      /********** CONSTRUCTOR **********/
      TreeDetector() = default;
      explicit TreeDetector(std::string_view bowFile);

      /********** METHODS **********/
      // Detect trees and output the result.
      cv::Mat detect(const cv::Mat& input);

   private:
      /********** METHODS **********/
      // Analyse the preprocessed data to find trees.
      bool analyse_();
      // Compute the score of an image from the class distances.
      double computeScore_(double treeDist, double nonTreeDist);
      // Draw the final result.
      bool drawResult_();
      // Extend a cell in a given direction and compute its score.
      double extendCell_(
         Rect<int>&      rect,
         Rect<int>::Side side,
         int             amount,
         int             limit);
      // Combine trees that overlap.
      int fuseTrees_(int ref);
      // Grow each candidate tree while it improves its score.
      void growCandidates_();
      // Preprocess the input image.
      bool preProcess_();

      /********** VARIABLES **********/
      BOWExtractor            treeExtractor_;       // Histogram extractor for trees.
      BOWExtractor            nonTreeExtractor_;    // Histogram extractor for non-trees.
      Image                   resizedInput_;        // Resized input image.
      Image                   result_;              // Final result.
      std::vector<Rect<int>>  segments_;            // Segmentation result.
      std::vector<Tree>       preliminaryTrees_;    // All regions containing trees.
      std::vector<Rect<int>>  trees_;               // Final trees in the image.
      std::pair<float, float> scale_{ 1.0F, 1.0F }; // Scaling factor of the image.
   };
} // namespace prj

#endif