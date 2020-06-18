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
      // Score thresholds for confirmed trees.
      static constexpr double base_threshold{ 0.04 };
      static constexpr double child_th_coeff{ 0.04 };
      static constexpr double growth_th{ 0.9 };
      static constexpr float  overlap_th{ 0.3 };

      /********** TYPE ALIASES **********/
      using Cell = ImagePyramid<pyr_children, pyr_depth>::Cell;

      /********** CONSTRUCTOR **********/
      TreeDetector() = default;
      explicit TreeDetector(std::string_view bowFile);

      /********** METHODS **********/
      // Detect trees and output the result.
      cv::Mat detect(const cv::Mat& input);

   private:
      /********** METHODS **********/
      // Add all candidate trees in the subtree that starts from node.
      void addCandidateTrees_(Cell* node);
      // Analyse the preprocessed data to find trees.
      bool analyse_(std::array<param, static_cast<int>(AParam::tot)>& params);
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
      // Grow each candidate tree while it improves its score.
      void growCandidates_();
      // Preprocess the input image.
      bool preProcess_(std::array<param, static_cast<int>(PParam::tot)>& params);

      /********** VARIABLES **********/
      BOWExtractor                          treeExtractor_;                    // Histogram extractor for trees.
      BOWExtractor                          nonTreeExtractor_;                 // Histogram extractor for non-trees.
      Image                                 resizedInput_;                     // Resized input image.
      std::vector<Image>                    processedImgs_;                    // Images after pre-processing.
      Image                                 result_;                           // Final result.
      std::vector<Tree>                     preliminaryTrees_;                 // All regions containing trees.
      std::vector<Rect<int>>                trees_;                            // Final trees in the image.
      std::pair<float, float>               scale_{ 1.0F, 1.0F };              // Scaling factor of the image.
      ImagePyramid<pyr_children, pyr_depth> pyramid_{ img_width, img_height }; // Analysis grid.
   };
} // namespace prj

#endif