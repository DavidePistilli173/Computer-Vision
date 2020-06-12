#include "TreeDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <thread>

using namespace prj;

const cv::Size   TreeDetector::analysis_res{ img_width, img_height };
const cv::Scalar TreeDetector::tree_colour{ 0, 0, 255 };

TreeDetector::TreeDetector(std::string_view bowFile)
{
   cv::FileStorage input{ bowFile.data(), cv::FileStorage::READ };
   if (!input.isOpened())
   {
      Log::error("Failed to open data file %s.", bowFile.data());
      throw std::exception();
   }

   input[xml_words.data()] >> bow_;
   input[xml_tree_hist.data()] >> avgTreeHist_;
   input[xml_nontree_hist.data()] >> avgNonTreeHist_;
}

cv::Mat TreeDetector::detect(const cv::Mat& input)
{
   result_ = input;

   Log::info("Resizing image.");
   // Used to convert tree coordinates back to the input's coordinate frame.
   scale_ = {
      static_cast<float>(input.cols) / analysis_res.width,
      static_cast<float>(input.rows) / analysis_res.height
   };
   resizedInput_ = input;
   resizedInput_.resize(analysis_res);

   Log::info("Preprocessing.");
   std::array<param, static_cast<int>(PParam::tot)> pParams{
      cv::Size{ 9, 9 },
      3.5,
      9,
      150.0,
      50.0,
      30.0,
      60.0,
      0.75
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
   using Cell = decltype(pyramid_)::Cell;

   auto sift = cv::xfeatures2d::SIFT::create(max_features);
   auto matcher = cv::BFMatcher::create(cv::NORM_L2);

   cv::BOWImgDescriptorExtractor bowExtractor{ sift, matcher };
   bowExtractor.setVocabulary(bow_);

   // Scan all cells for trees.
   auto preliminaryAnalysis = [this, &sift, &matcher, &bowExtractor](Cell* cell) {
      cv::Mat treeImg{ getTree(resizedInput_.image(), cell->rect, std::pair{ 1.0F, 1.0F }) };

      std::vector<cv::KeyPoint>     keypoints;
      cv::Mat                       descriptor;
      std::vector<std::vector<int>> currentHistogram;

      sift->detect(treeImg, keypoints);

      if (keypoints.size() > min_features)
      {
         bowExtractor.compute(treeImg, keypoints, descriptor, &currentHistogram);

         // Compute the BOW normalised histogram.
         cv::Mat hist{ cv::Mat::zeros(num_words, 1, CV_32F) };
         for (int i = 0; i < num_words; ++i)
         {
            hist.at<float>(i) += currentHistogram[i].size();
         }
         cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

         // Check whether the histogram represents a tree or not.
         double treeDistance{ cv::compareHist(avgTreeHist_, hist, cv::HISTCMP_BHATTACHARYYA) };
         double nonTreeDistance{ cv::compareHist(avgNonTreeHist_, hist, cv::HISTCMP_BHATTACHARYYA) };
         if (treeDistance < nonTreeDistance)
         {
            Log::info_d(
               "Tree detected: (%d, %d, %d, %d) with score %f.",
               cell->rect.x,
               cell->rect.y,
               cell->rect.w,
               cell->rect.h,
               treeDistance);
            cell->tree = true;
         }
      }
   };
   pyramid_.visit(preliminaryAnalysis);

   // Label the highest trees in the pyramid.
   auto confirm = [](Cell* cell) {
      // If the cell is a leaf.
      if (cell->children[0][0] == nullptr)
      {
         cell->confirmed = cell->tree;
         return;
      }

      if (cell->tree)
      {
         int confirmed{ 0 };
         for (const auto& row : cell->children)
         {
            for (const auto& child : row)
            {
               if (child->confirmed) ++confirmed;
            }
         }

         if (confirmed >= pyr_children - 1) cell->confirmed = true;
      }
   };
   pyramid_.visit(confirm);

   // Get the final candidate trees.
   addCandidateTrees_(pyramid_.root());

   trees_ = preliminaryTrees_;

   return true;
}

bool TreeDetector::drawResult_()
{
   int i{ 1 };
   for (const auto& tree : trees_)
   {
      cv::Point pt1{ cvRound(tree.x * scale_.first), cvRound(tree.y * scale_.second) };
      cv::Point pt2{
         cvRound((tree.x + tree.w) * scale_.first),
         cvRound((tree.y + tree.h) * scale_.second)
      };
      result_.draw(
         Image::Shape::rect,
         { pt1, pt2 },
         tree_colour);
      result_.drawText(std::to_string(i), pt1 + (pt2 - pt1) / 2, tree_colour);
      ++i;
   }
   return true;
}

bool TreeDetector::preProcess_(std::array<param, static_cast<int>(PParam::tot)>& params)
{
   //resizedInput_.equaliseHistogram();
   return true;
}

void TreeDetector::addCandidateTrees_(ImagePyramid<pyr_children, pyr_depth>::Cell* node)
{
   if (node == nullptr) return;

   if (node->confirmed)
   {
      preliminaryTrees_.emplace_back(node->rect);
      return;
   }

   if (node->tree) preliminaryTrees_.emplace_back(node->rect);
   for (const auto& row : node->children)
   {
      for (const auto& child : row)
      {
         addCandidateTrees_(child.get());
      }
   }
}
