#include "TreeDetector.hpp"

#include "Log.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <thread>

using namespace prj;

const cv::Size             TreeDetector::analysis_res{ img_width, img_height };
const cv::Scalar           TreeDetector::tree_colour{ 0, 0, 255 };
const GaussianFilterParams TreeDetector::gauss_filt_params{
   cv::Size{ 9, 9 }, 3.5
};

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
   if (sift.empty() || matcher.empty()) Log::error("ABOFSABGFOASBGOASB");
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
   if (!preProcess_())
   {
      Log::error("Preprocessing failed.");
      return cv::Mat{};
   }

   Log::info("Analysing data.");
   if (!analyse_())
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

bool TreeDetector::analyse_()
{
   Log::info("Choosing most relevant segments.");
   for (const auto& segment : segments_)
   {
      cv::Mat img{ getTree(resizedInput_.image(), segment) };
      double  treeDist{ treeExtractor_.computeDist(img) };
      double  nonTreeDist{ nonTreeExtractor_.computeDist(img) };
      if (treeDist >= 0 || nonTreeDist >= 0)
      {
         double score{ computeScore_(treeDist, nonTreeDist) };
         if (score > base_threshold)
         {
            preliminaryTrees_.emplace_back(segment, score);
         }
      }
   }

   Log::info("Combining overlapping candidates.");
   // Sort the preliminary trees by increasing score.
   std::sort(
      preliminaryTrees_.begin(),
      preliminaryTrees_.end(),
      [](const auto& left, const auto& right) {
         return left.score > right.score;
      });
   // Combine overlapping candidates.
   int i{ 0 };
   while (i < preliminaryTrees_.size())
   {
      i = fuseTrees_(i, true);
      ++i;
   }

   if constexpr (debug)
   {
      for (const auto& candidate : preliminaryTrees_)
      {
         trees_.emplace_back(candidate.rect);
      }
      drawResult_();
      result_.display();
      trees_.clear();
   }

   // Compute the final trees.
   Log::info("Computing final trees.");
   growCandidates_();

   // Set the final trees.
   for (const auto& tree : preliminaryTrees_)
   {
      trees_.emplace_back(tree.rect);
   }

   //trees_ = segments_;

   return true;
}

double TreeDetector::computeScore_(double treeDist, double nonTreeDist)
{
   if (treeDist < 0) return treeDist;

   return nonTreeDist - treeDist;
}

bool TreeDetector::drawResult_()
{
   int i{ 1 };
   for (const auto& tree : trees_)
   {
      cv::Point pt1{ cvRound(tree.x * scale_.first), cvRound(tree.y * scale_.second) };
      cv::Point pt2{
         cvRound(static_cast<float>(tree.x + tree.w) * scale_.first),
         cvRound(static_cast<float>(tree.y + tree.h) * scale_.second)
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

int TreeDetector::fuseTrees_(int ref, bool removeRef)
{
   int j{ 0 };
   while (j < preliminaryTrees_.size())
   {
      if (ref != j)
      {
         if (preliminaryTrees_[ref].rect.contains(preliminaryTrees_[j].rect))
         {
            if (removeRef && preliminaryTrees_[j].score > growth_th * preliminaryTrees_[ref].score)
            {
               auto it = preliminaryTrees_.begin() + ref;
               preliminaryTrees_.erase(it);
               return ref;
            }

            auto it = preliminaryTrees_.begin() + j;
            if (j < ref) --ref;
            preliminaryTrees_.erase(it);
         }
         else
         {
            Rect<int> overlapResult{
               preliminaryTrees_[ref].rect.overlaps(preliminaryTrees_[j].rect, overlap_th)
            };

            if (overlapResult.w != 0 && overlapResult.h != 0)
            {
               preliminaryTrees_[ref].rect = overlapResult;
               auto it = preliminaryTrees_.begin() + j;
               if (j < ref) --ref;
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
   return ref;
}

void TreeDetector::growCandidates_()
{
   using Side = Rect<int>::Side;

   auto sift = cv::xfeatures2d::SIFT::create(max_features);

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
            preliminaryTrees_[i].rect.getExtension(Side::right, cell_w, rightLimit),
            preliminaryTrees_[i].rect.getExtension(Side::top, cell_h, 0),
            preliminaryTrees_[i].rect.getExtension(Side::left, cell_h, 0),
            preliminaryTrees_[i].rect.getExtension(Side::bottom, cell_w, bottomLimit)
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
               extendCell_(newTrees[static_cast<int>(Side::right)].rect, Side::right, cell_w, rightLimit);
         }
         else
            newTrees[static_cast<int>(Side::right)].score = -1.0;

         if (hasFeatures(extensions[static_cast<int>(Side::top)]))
         {
            newTrees[static_cast<int>(Side::top)].score =
               extendCell_(newTrees[static_cast<int>(Side::top)].rect, Side::top, cell_h, 0);
         }
         else
            newTrees[static_cast<int>(Side::top)].score = -1.0;

         if (hasFeatures(extensions[static_cast<int>(Side::left)]))
         {
            newTrees[static_cast<int>(Side::left)].score =
               extendCell_(newTrees[static_cast<int>(Side::left)].rect, Side::left, cell_w, 0);
         }
         else
            newTrees[static_cast<int>(Side::left)].score = -1.0;

         if (hasFeatures(extensions[static_cast<int>(Side::bottom)]))
         {
            newTrees[static_cast<int>(Side::bottom)].score =
               extendCell_(newTrees[static_cast<int>(Side::bottom)].rect, Side::bottom, cell_h, bottomLimit);
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
      i = fuseTrees_(i);

      ++i;
   }
}

bool TreeDetector::preProcess_()
{
   Log::info("Filtering image.");
   Image processedImage{ resizedInput_ };
   processedImage.bilateralFilter(bi_filt_params);
   processedImage.gaussianFilter(gauss_filt_params);
   Log::info("Segmenting image.");
   processedImage.segment(seg_params);

   if constexpr (debug) processedImage.display(Image::RegionType::label);

   Log::info("Retrieving segments.");
   segments_ = processedImage.getRegions(Image::RegionType::label);

   return true;
}
