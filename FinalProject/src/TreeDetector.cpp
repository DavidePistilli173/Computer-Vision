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

   input[xml_tree_voc.data()] >> treeVocabulary_;
   if (treeVocabulary_.empty())
   {
      Log::error("Failed to load tree vocabulary.");
      throw std::exception();
   }

   input[xml_nontree_voc.data()] >> nonTreeVocabulary_;
   if (nonTreeVocabulary_.empty())
   {
      Log::error("Failed to load non-tree vocabulary.");
      throw std::exception();
   }

   input[xml_tree_hist.data()] >> avgTreeHist_;
   if (avgTreeHist_.empty())
   {
      Log::error("Failed to load average tree histogram.");
      throw std::exception();
   }

   input[xml_nontree_hist.data()] >> avgNonTreeHist_;
   if (avgNonTreeHist_.empty())
   {
      Log::error("Failed to load average non-tree histogram.");
      throw std::exception();
   }
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

   cv::BOWImgDescriptorExtractor treeExtractor{ sift, matcher };
   treeExtractor.setVocabulary(treeVocabulary_);
   cv::BOWImgDescriptorExtractor nonTreeExtractor{ sift, matcher };
   nonTreeExtractor.setVocabulary(nonTreeVocabulary_);

   // Scan all cells for trees.
   auto preliminaryAnalysis = [this, &sift, &treeExtractor, &nonTreeExtractor](Cell* cell) {
      cv::Mat treeImg{ getTree(resizedInput_.image(), cell->rect, std::pair{ 1.0F, 1.0F }) };

      double treeDist{ computeScore(treeExtractor, sift, treeImg, avgTreeHist_) };
      double nonTreeDist{ computeScore(nonTreeExtractor, sift, treeImg, avgNonTreeHist_) };

      // Check whether the histogram represents a tree or not.
      if (treeDist < 0)
         cell->score = treeDist;
      else
         cell->score = nonTreeDist - treeDist;
      if (cell->score > 0)
      {
         Log::info_d(
            "Tree detected: (%d, %d, %d, %d) with score %f.",
            cell->rect.x,
            cell->rect.y,
            cell->rect.w,
            cell->rect.h,
            cell->score);
      }
   };
   pyramid_.visit(preliminaryAnalysis);

   // Label the highest trees in the pyramid.
   auto confirm = [](Cell* cell) {
      // If the cell is a leaf.
      if (cell->children[0][0] == nullptr)
      {
         Log::info_d("Leaf with score %f.", cell->score);
         if (cell->score > base_threshold)
         {
            Log::warn_d(
               "Tree confirmed.\n  rect: (%d, %d, %d, %d).",
               cell->rect.x,
               cell->rect.y,
               cell->rect.w,
               cell->rect.h);
            cell->status = Cell::Status::tree;
         }
         else
            cell->status = Cell::Status::non_tree;
         return;
      }

      Log::info_d("Intermediate with score %f.", cell->score);
      int    confirmedChildren{ 0 };
      double avgChildrenScore{ 0.0 };
      for (const auto& row : cell->children)
      {
         for (const auto& child : row)
         {
            if (child->status != Cell::Status::non_tree)
               ++confirmedChildren;
            avgChildrenScore += child->score;
         }
      }
      avgChildrenScore /= pyr_children;

      if (confirmedChildren > 0)
         cell->status = Cell::Status::discard;
      else if (cell->score > 0)
      {
         if (
            cell->score > base_threshold ||
            (cell->score >= avgChildrenScore && avgChildrenScore > child_th_coeff * cell->score))
         {
            cell->status = Cell::Status::tree;
            Log::warn_d(
               "Tree confirmed.  rect: (%d, %d, %d, %d)\n  confirmedChildren = %d, score = %f, avgChildrenScore = %f.",
               cell->rect.x,
               cell->rect.y,
               cell->rect.w,
               cell->rect.h,
               confirmedChildren,
               cell->score,
               avgChildrenScore);
         }
         else
            cell->status = Cell::Status::non_tree;
      }
      else
      {
         cell->status = Cell::Status::non_tree;
      }
   };
   pyramid_.visit(confirm);

   // Get the final candidate trees.
   addCandidateTrees_(pyramid_.root());

   trees_ = preliminaryTrees_;

   return true;
}

double prj::TreeDetector::computeScore(
   cv::BOWImgDescriptorExtractor& extractor,
   cv::Ptr<cv::xfeatures2d::SIFT> sift,
   const cv::Mat&                 img,
   const cv::Mat&                 referenceHist)
{
   std::vector<cv::KeyPoint>     keypoints;
   cv::Mat                       descriptor;
   std::vector<std::vector<int>> currentHistogram;

   sift->detect(img, keypoints);

   if (keypoints.size() > min_features)
   {
      extractor.compute(img, keypoints, descriptor, &currentHistogram);

      // Compute the BOW normalised histogram.
      cv::Mat hist{ cv::Mat::zeros(num_words, 1, CV_32F) };
      for (int i = 0; i < num_words; ++i)
      {
         hist.at<float>(i) += currentHistogram[i].size();
      }
      cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

      return cv::compareHist(referenceHist, hist, cv::HISTCMP_BHATTACHARYYA);
   }

   return -1.0;
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
   using Cell = ImagePyramid<pyr_children, pyr_depth>::Cell;

   if (node == nullptr) return;

   if (node->status == Cell::Status::tree)
   {
      preliminaryTrees_.emplace_back(node->rect);
      return;
   }

   for (const auto& row : node->children)
   {
      for (const auto& child : row)
      {
         addCandidateTrees_(child.get());
      }
   }
}
