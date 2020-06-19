#ifndef PRJ_UTILITY_HPP
#define PRJ_UTILITY_HPP

#include "Rect.hpp"

#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

namespace prj
{
   /********** CONSTANTS **********/
   // Tag for tree words.
   constexpr std::string_view xml_tree_voc{ "TreeVocabulary" };
   // Tag for non-tree words.
   constexpr std::string_view xml_nontree_voc{ "NonTreeVocabulary" };
   // Tag for tree histogram.
   constexpr std::string_view xml_tree_hist{ "Tree" };
   // Tag for non-tree histogram.
   constexpr std::string_view xml_nontree_hist{ "NonTree" };
   // Maximum of features for each tree.
   constexpr int max_features{ 2048 };
   constexpr int min_features{ 256 };
   // Number of words in a vocabulary.
   constexpr int num_words{ 128 };
   // Image dimensions.
   constexpr int img_width{ 2048 };
   constexpr int img_height{ 2048 };

#ifdef PRJ_DEBUG
   constexpr bool debug{ true };
#else
   constexpr bool debug{ false };
#endif

#ifdef PRJ_DEBUG_MSGS
   constexpr bool debug_msgs{ true };
#else
   constexpr bool debug_msgs{ false };
#endif

   /********** ENUMS **********/
   // Keycodes.
   enum class Key
   {
      esc = 27
   };
   // HSV colour channels
   enum class HSV
   {
      h, // Hue
      s, // Saturation
      v  // Value
   };

   /********** STRUCTS **********/
   struct Tree
   {
      constexpr Tree() = default;
      constexpr Tree(const Rect<int>& newRect, double newScore) :
         rect{ newRect },
         score{ newScore } {}
      Rect<int> rect{ 0, 0, 0, 0 };
      double    score{ 0.0 };
   };

   /********** FUNCTIONS **********/
   // Get an image portion described by a scaled rectangle.
   cv::Mat getTree(
      const cv::Mat&          mat,
      Rect<int>               tree,
      std::pair<float, float> scale = std::pair{ 1.0F, 1.0F });

   // Constexpr power.
   template<typename T>
   constexpr T pow(T base, T exp)
   {
      static_assert(std::is_integral<T>::value);
      if (exp == 0) return 1;
      if (exp == 1) return base;
      return base * pow(base, exp - 1);
   }

   template<typename T>
   constexpr T sqrt_helper(T x, T lo, T hi)
   {
      if (lo == hi)
         return lo;

      const T mid = (lo + hi + 1) / 2;

      if (x / mid < mid)
         return sqrt_helper<T>(x, lo, mid - 1);
      else
         return sqrt_helper(x, mid, hi);
   }

   template<typename T>
   constexpr T ct_sqrt(T x)
   {
      return sqrt_helper<T>(x, 0, x / 2 + 1);
   }

   /********** CLASSES **********/
   class BOWExtractor
   {
   public:
      BOWExtractor() = default;

      // Compute the distance between img and the reference histogram.
      double computeDist(const cv::Mat& img);
      // Initialise the BOW extractor.
      bool initExtractor(cv::Ptr<cv::xfeatures2d::SIFT> sift, cv::Ptr<cv::BFMatcher> matcher);
      // Set the data to work on.
      bool setData(const cv::Mat& hist, const cv::Mat& vocab);

   private:
      cv::Mat                        refHist_; // Reference histogram.
      cv::Mat                        vocab_;   // Vocabulary,
      cv::Ptr<cv::xfeatures2d::SIFT> sift_;    // Sift.
      // BOW extractor.
      std::unique_ptr<cv::BOWImgDescriptorExtractor> extractor_;
   };

   // Quad-tree used to segment an image.
   template<int Children, int Depth>
   class ImagePyramid
   {
   public:
      // Single grid cell.
      struct Cell
      {
         // Cell detection status.
         enum class Status
         {
            tree,     // The cell has a tree.
            non_tree, // The cell does not have a tree.
            discard   // This cell and all its parents will be discarded.
         };

         Rect<int> rect;                       // Position and size of the cell.
         double    score{ -1.0 };              // If it is positive there is a tree in the cell.
         Status    status{ Status::non_tree }; // Current staus of the cell.

         // Children of the cell.
         std::array<std::array<std::unique_ptr<Cell>, ct_sqrt(Children)>, ct_sqrt(Children)> children{ nullptr };
      };

      /********** CONSTANTS **********/
      static constexpr int side_elems{ ct_sqrt(Children) };

      /********** CONSTRUCTOR **********/
      ImagePyramid(int imgWidth, int imgHeight)
      {
         static_assert(Depth > 0);

         root_.rect = Rect<int>(0, 0, imgWidth - 1, imgHeight - 1);
         buildTree_(&root_, 1);
      }

      /********** METHODS **********/
      int minCellHeight() const
      {
         return minCellHeight_;
      }

      int minCellWidth() const
      {
         return minCellWidth_;
      }

      template<typename Callable>
      void visit(Callable func)
      {
         visitTree_(func, &root_, 1);
      }

      Cell* root()
      {
         return &root_;
      }

   private:
      /********** METHODS **********/
      void buildTree_(Cell* root, int lvl)
      {
         if (lvl == Depth)
         {
            minCellWidth_ = root->rect.w;
            minCellHeight_ = root->rect.h;
            return;
         }

         int cellWidth{ root->rect.w / side_elems };
         int cellHeight{ root->rect.h / side_elems };
         for (int row = 0; row < side_elems; ++row)
         {
            for (int col = 0; col < side_elems; ++col)
            {
               root->children[row][col] = std::make_unique<Cell>();
               auto rect = Rect<int>(
                  root->rect.x + (col * root->rect.w) / side_elems,
                  root->rect.y + (row * root->rect.h) / side_elems,
                  cellWidth,
                  cellHeight);

               int maxX{ rect.x + rect.w };
               int maxY{ rect.y + rect.h };
               int maxRootX{ root->rect.x + root->rect.w };
               int maxRootY{ root->rect.y + root->rect.h };
               if (maxX >= maxRootX) rect.w -= maxRootX - maxX + 1;
               if (maxY >= maxRootY) rect.h -= maxRootY - maxY + 1;
               root->children[row][col]->rect = rect;

               buildTree_(root->children[row][col].get(), lvl + 1);
            }
         }
      }

      template<typename Callable>
      void visitTree_(Callable func, Cell* node, int lvl)
      {
         if (node->children[0][0] != nullptr)
         {
            for (auto& row : node->children)
            {
               for (auto& child : row)
               {
                  visitTree_(func, child.get(), lvl + 1);
               }
            }
         }
         std::invoke(func, node);
      }

      /********** VARIABLES **********/
      // Root of the tree.
      Cell root_;
      // Minimum size of one cell.
      int minCellWidth_{ 0 };
      int minCellHeight_{ 0 };
   };
} // namespace prj

#endif