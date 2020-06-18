#include "TreeDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <thread>

using namespace prj;

const cv::Size   TreeDetector::analysis_res{ img_width, img_height };
const cv::Scalar TreeDetector::tree_colour{ 0, 0, 255 };

TreeDetector::TreeDetector(std::string_view bowFile)
{
   // Open the BOW data file.
   cv::FileStorage input{ bowFile.data(), cv::FileStorage::READ };
   if (!input.isOpened())
   {
      Log::error("Failed to open data file %s.", bowFile.data());
      throw std::exception();
   }

   // Initialise SIFT and BFMatcher.
   auto sift = cv::xfeatures2d::SIFT::create(max_features);
   if (sift == nullptr)
   {
      Log::error("Failed to initialise SIFT.");
      throw std::exception();
   }
   auto matcher = cv::BFMatcher::create(cv::NORM_L2);
   if (matcher == nullptr)
   {
      Log::error("Failed to initialise BFMatcher.");
      throw std::exception();
   }

   cv::Mat vocab;
   cv::Mat hist;
   // Load tree data.
   input[xml_tree_voc.data()] >> vocab;
   if (vocab.empty())
   {
      Log::error("Failed to load tree vocabulary.");
      throw std::exception();
   }
   input[xml_tree_hist.data()] >> hist;
   if (hist.empty())
   {
      Log::error("Failed to load average tree histogram.");
      throw std::exception();
   }

   // Initialise the tree extractor.
   if (!treeExtractor_.initExtractor(sift, matcher))
   {
      Log::error("Failed to initialise the tree extractor.");
      throw std::exception();
   }
   if (!treeExtractor_.setData(hist, vocab))
   {
      Log::error("Failed to set the tree extractor data.");
      throw std::exception();
   }

   // Load non-tree data.
   input[xml_nontree_voc.data()] >> vocab;
   if (vocab.empty())
   {
      Log::error("Failed to load non-tree vocabulary.");
      throw std::exception();
   }
   input[xml_nontree_hist.data()] >> hist;
   if (hist.empty())
   {
      Log::error("Failed to load average non-tree histogram.");
      throw std::exception();
   }

   // Initialise the non-tree extractor.
   if (!nonTreeExtractor_.initExtractor(sift, matcher))
   {
      Log::error("Failed to initialise the non-tree extractor.");
      throw std::exception();
   }
   if (!nonTreeExtractor_.setData(hist, vocab))
   {
      Log::error("Failed to set the non-tree extractor data.");
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
   // Scan all cells for trees.
   Log::info("Performing prelimiary analysis.");
   auto preliminaryAnalysis = [this](Cell* cell) {
      cv::Mat treeImg{ getTree(resizedInput_.image(), cell->rect) };

      double treeDist{ treeExtractor_.computeDist(treeImg) };
      double nonTreeDist{ nonTreeExtractor_.computeDist(treeImg) };

      cell->score = computeScore_(treeDist, nonTreeDist);

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
   Log::info("Confirming candidate trees.");
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

   // Compute the final trees.
   Log::info("Computing final trees.");
   // Sort the preliminary trees by increasing score.
   std::sort(
      preliminaryTrees_.begin(),
      preliminaryTrees_.end(),
      [](const auto& left, const auto& right) {
         return left.score > right.score;
      });
   growCandidates_();

   // Set the final trees.
   for (const auto& tree : preliminaryTrees_)
   {
      trees_.emplace_back(tree.rect);
   }

   return true;
}

double prj::TreeDetector::computeScore_(double treeDist, double nonTreeDist)
{
   if (treeDist < 0)
      return treeDist;
   else
      return nonTreeDist - treeDist;
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

double prj::TreeDetector::extendCell_(
   Rect<int>&      rect,
   Rect<int>::Side side,
   int             amount,
   int             limit)
{
   rect.extend(side, amount, limit);

   cv::Mat img{ getTree(resizedInput_.image(), rect) };
   double  treeDist{ treeExtractor_.computeDist(img) };
   double  nonTreeDist{ nonTreeExtractor_.computeDist(img) };
   return computeScore_(treeDist, nonTreeDist);
}

void TreeDetector::growCandidates_()
{
   using Side = Rect<int>::Side;

   auto sift = cv::xfeatures2d::SIFT::create(max_features);

   int cellW{ pyramid_.minCellWidth() };
   int cellH{ pyramid_.minCellHeight() };

   int rightLimit{ resizedInput_.image().cols - 1 };
   int bottomLimit{ resizedInput_.image().rows - 1 };

   int i{ 0 };
   while (i < preliminaryTrees_.size())
   {
      bool   done{ false };
      double scoreTh{ preliminaryTrees_[i].score * growth_th };

      while (!done)
      {
         std::array newTrees{
            Tree{ preliminaryTrees_[i].rect, 0.0 },
            Tree{ preliminaryTrees_[i].rect, 0.0 },
            Tree{ preliminaryTrees_[i].rect, 0.0 },
            Tree{ preliminaryTrees_[i].rect, 0.0 }
         };

         std::array extensions{
            preliminaryTrees_[i].rect.getExtension(Side::right, cellW, rightLimit),
            preliminaryTrees_[i].rect.getExtension(Side::top, cellH, 0),
            preliminaryTrees_[i].rect.getExtension(Side::left, cellH, 0),
            preliminaryTrees_[i].rect.getExtension(Side::bottom, cellW, bottomLimit)
         };

         auto hasFeatures = [this, &sift](const Rect<int>& rect) {
            cv::Mat                   img{ getTree(resizedInput_.image(), rect) };
            std::vector<cv::KeyPoint> keypoints;
            sift->detect(img, keypoints);
            return keypoints.size() >= min_features;
         };

         if (hasFeatures(extensions[static_cast<int>(Side::right)]))
         {
            newTrees[static_cast<int>(Side::right)].score =
               extendCell_(newTrees[static_cast<int>(Side::right)].rect, Side::right, cellW, rightLimit);
         }
         else
            newTrees[static_cast<int>(Side::right)].score = -1.0;

         if (hasFeatures(extensions[static_cast<int>(Side::top)]))
         {
            newTrees[static_cast<int>(Side::top)].score =
               extendCell_(newTrees[static_cast<int>(Side::top)].rect, Side::top, cellH, 0);
         }
         else
            newTrees[static_cast<int>(Side::top)].score = -1.0;

         if (hasFeatures(extensions[static_cast<int>(Side::left)]))
         {
            newTrees[static_cast<int>(Side::left)].score =
               extendCell_(newTrees[static_cast<int>(Side::left)].rect, Side::left, cellW, 0);
         }
         else
            newTrees[static_cast<int>(Side::left)].score = -1.0;

         if (hasFeatures(extensions[static_cast<int>(Side::bottom)]))
         {
            newTrees[static_cast<int>(Side::bottom)].score =
               extendCell_(newTrees[static_cast<int>(Side::bottom)].rect, Side::bottom, cellH, bottomLimit);
         }
         else
            newTrees[static_cast<int>(Side::bottom)].score = -1.0;

         auto maxScore = std::max_element(
            newTrees.begin(),
            newTrees.end(),
            [](const auto& left, const auto& right) { return left.score < right.score; });

         if (maxScore->rect != preliminaryTrees_[i].rect && maxScore->score > scoreTh)
         {
            preliminaryTrees_[i] = { maxScore->rect, maxScore->score };
         }
         else
            done = true;
      }

      // Remove regions that have been enveloped.
      int j{ 0 };
      while (j < preliminaryTrees_.size())
      {
         if (i != j)
         {
            if (preliminaryTrees_[i].rect.contains(preliminaryTrees_[j].rect))
            {
               auto it = preliminaryTrees_.begin() + j;
               if (j < i) --i;
               preliminaryTrees_.erase(it);
            }
            else
            {
               Rect<int> overlapResult{
                  preliminaryTrees_[i].rect.overlaps(preliminaryTrees_[j].rect, overlap_th)
               };

               if (overlapResult.w != 0 && overlapResult.h != 0)
               {
                  preliminaryTrees_[i].rect = overlapResult;
                  auto it = preliminaryTrees_.begin() + j;
                  if (j < i) --i;
                  preliminaryTrees_.erase(it);
                  j = 0;
               }
               else
                  ++j;
            }
         }
         else
            ++j;
      }

      ++i;
   }
}

bool TreeDetector::preProcess_(std::array<param, static_cast<int>(PParam::tot)>& params)
{
   return true;
}

void TreeDetector::addCandidateTrees_(ImagePyramid<pyr_children, pyr_depth>::Cell* node)
{
   if (node == nullptr) return;

   if (node->status == Cell::Status::tree)
   {
      preliminaryTrees_.emplace_back(Tree(node->rect, node->score));
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
