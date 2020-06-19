#include "utility.hpp"

#include <cstdint>
#include <limits>
#include <opencv2/imgproc.hpp>

using namespace prj;

double BOWExtractor::computeDist(const cv::Mat& img)
{
   if (extractor_ == nullptr || refHist_.empty()) return -1.0;

   std::vector<cv::KeyPoint>     keypoints;
   cv::Mat                       descriptor;
   std::vector<std::vector<int>> currentHistogram;

   sift_->detect(img, keypoints);

   if (keypoints.size() > min_features)
   {
      extractor_->compute(img, keypoints, descriptor, &currentHistogram);

      // Compute the BOW normalised histogram.
      cv::Mat hist{ cv::Mat::zeros(num_words, 1, CV_32F) };
      for (int i = 0; i < num_words; ++i)
      {
         hist.at<float>(i) += currentHistogram[i].size();
      }
      cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

      return cv::compareHist(refHist_, hist, cv::HISTCMP_BHATTACHARYYA);
   }

   return -1.0;
}

bool BOWExtractor::initExtractor(cv::Ptr<cv::xfeatures2d::SIFT> sift, cv::Ptr<cv::BFMatcher> matcher)
{
   if (sift == nullptr || matcher == nullptr) return false;
   sift_ = sift;
   extractor_ = std::make_unique<cv::BOWImgDescriptorExtractor>(
      cv::BOWImgDescriptorExtractor{ sift_, matcher });
   if (!vocab_.empty()) extractor_->setVocabulary(vocab_);
}

bool BOWExtractor::setData(const cv::Mat& hist, const cv::Mat& vocab)
{
   if (hist.empty() || vocab.empty()) return false;
   refHist_ = hist.clone();
   vocab_ = vocab.clone();
   if (extractor_ != nullptr) extractor_->setVocabulary(vocab_);
}
