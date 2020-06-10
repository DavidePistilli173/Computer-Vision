#include "TreeDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <thread>

using namespace prj;

const cv::Size   TreeDetector::analysis_res{ 1000, 1000 };
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
   input[xml_hist.data()] >> avgHist_;
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
      29,
      100.0,
      100.0,
      0.076F,
      30.0,
      60.0,
      0.45
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
   // Extract regions from the preprocessed images.
   Log::info("Extracting relevant image regions.");
   std::vector<std::vector<Rect<int>>> regions;
   for (const auto& img : processedImgs_)
   {
      regions.emplace_back(img.getRegions());
   }

   auto sift = cv::xfeatures2d::SIFT::create(num_features);
   auto matcher = cv::BFMatcher::create(cv::NORM_L2);

   cv::BOWImgDescriptorExtractor bowExtractor{ sift, matcher };
   bowExtractor.setVocabulary(bow_);

   Log::info("Checking histograms.");
   for (const auto& img : regions)
   {
      for (const auto& region : img)
      {
         cv::Mat treeImg{ getTree(resizedInput_.image(), region, std::pair{ 1.0F, 1.0F }) };

         std::vector<cv::KeyPoint>     keypoints;
         cv::Mat                       descriptor;
         std::vector<std::vector<int>> currentHistogram;

         sift->detect(treeImg, keypoints);

         if (keypoints.size() > 0)
         {
            bowExtractor.compute(treeImg, keypoints, descriptor, &currentHistogram);

            cv::Mat hist{ cv::Mat::zeros(num_words, 1, CV_32F) };
            for (int i = 0; i < num_words; ++i)
            {
               hist.at<float>(i) += currentHistogram[i].size();
            }
            cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

            double distance{ cv::compareHist(avgHist_, hist, cv::HISTCMP_BHATTACHARYYA) };
            Log::info_d("Score %f.", distance);
            if (distance < score_th)
            {
               Log::info_d(
                  "Tree detected: (%d, %d, %d, %d) with score %f.",
                  region.x,
                  region.y,
                  region.w,
                  region.h,
                  distance);
               trees_.emplace_back(region);
            }
         }
      }
   }

   return true;
}

bool TreeDetector::drawResult_()
{
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
   }
   return true;
}

bool TreeDetector::preProcess_(std::array<param, static_cast<int>(PParam::tot)>& params)
{
   Image filteredImg{ resizedInput_ };
   if constexpr (debug) filteredImg.display();

   filteredImg.gaussianFilter(
      std::get<cv::Size>(params[static_cast<int>(PParam::gauss_size)]),
      std::get<double>(params[static_cast<int>(PParam::gauss_sig)]));

   filteredImg.bilateralFilter(
      std::get<int>(params[static_cast<int>(PParam::bil_size)]),
      std::get<double>(params[static_cast<int>(PParam::bil_col_sig)]),
      std::get<double>(params[static_cast<int>(PParam::bil_space_sig)]));

   if constexpr (debug) filteredImg.display();

   Image equalisedFilteredImg{ filteredImg };

   filteredImg.segment(
      std::get<double>(params[static_cast<int>(PParam::canny_th1)]),
      std::get<double>(params[static_cast<int>(PParam::canny_th2)]),
      std::get<double>(params[static_cast<int>(PParam::dist_th)]));

   equalisedFilteredImg.equaliseHistogram();
   equalisedFilteredImg.segment(
      std::get<double>(params[static_cast<int>(PParam::canny_th1)]),
      std::get<double>(params[static_cast<int>(PParam::canny_th2)]),
      std::get<double>(params[static_cast<int>(PParam::dist_th)]));

   if constexpr (debug) resizedInput_.display();
   if constexpr (debug) filteredImg.display(true);
   if constexpr (debug) equalisedFilteredImg.display(true);

   processedImgs_.reserve(2);
   processedImgs_.emplace_back(filteredImg);
   processedImgs_.emplace_back(equalisedFilteredImg);

   return true;
}
