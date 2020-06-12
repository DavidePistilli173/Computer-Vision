#include "TreeDetectorTrainer.hpp"

#include <algorithm>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <thread>

using namespace prj;

const cv::Size         TreeDetectorTrainer::img_size{ img_width, img_height };
const cv::TermCriteria TreeDetectorTrainer::cluster_criteria{
   cv::TermCriteria::Type::COUNT,
   TreeDetectorTrainer::num_cluster_iter,
   0.0
};

bool TreeDetectorTrainer::train(std::string_view cfgFile, std::string_view imgFolder)
{
   Log::info("Scanning the dataset configuration file.");
   if (!parse_(cfgFile))
   {
      Log::error("Parsing failed.");
      return false;
   }

   Log::info("Computing features.");
   if (!compute_(imgFolder))
   {
      Log::error("Failed to compute tree features.");
      return false;
   }

   Log::info("Clustering features.");
   buildVocabulary_();

   Log::info("Computing average tree histogram.");
   buildHistogram_(imgFolder);

   std::string outputFile{ imgFolder };
   outputFile += output_file;
   Log::info("Saving results to %s.", outputFile.c_str());
   if (!save_(outputFile))
   {
      Log::error("Failed to save results.");
      return false;
   }

   return true;
}

void TreeDetectorTrainer::buildHistogram_(std::string_view folder)
{
   avgTreeHist_.first = cv::Mat::zeros(num_words, 1, CV_32F);
   avgNonTreeHist_.first = cv::Mat::zeros(num_words, 1, CV_32F);

   std::atomic<size_t> treeCount{ 0 };    // Total amount of trees processed.
   std::atomic<size_t> nonTreeCount{ 0 }; // Total amount of non-tree cells processed.
   index_ = 0;                            // Reset the training data index.

   unsigned int num_threads{ std::thread::hardware_concurrency() };
   num_threads = std::max(1U, num_threads);
   Log::info_d("Running on %d threads.", num_threads);

   std::vector<std::thread> threads;
   threads.reserve(num_threads);
   for (int i = 0; i < num_threads; ++i)
   {
      threads.emplace_back(
         std::thread([this, folder, &treeCount, &nonTreeCount]() {
            this->histogramWorker_(folder, treeCount, nonTreeCount);
         }));
   }
   for (auto& thread : threads)
   {
      thread.join();
   }

   avgTreeHist_.first /= treeCount;
   cv::normalize(avgTreeHist_.first, avgTreeHist_.first, 1.0, 0.0, cv::NORM_L1);
   avgNonTreeHist_.first /= nonTreeCount;
   cv::normalize(avgNonTreeHist_.first, avgNonTreeHist_.first, 1.0, 0.0, cv::NORM_L1);
}

void TreeDetectorTrainer::buildVocabulary_()
{
   cv::BOWKMeansTrainer trainer{ num_words, cluster_criteria };

   // Add all descriptors to the trainer.
   for (const auto& image : trainingData_)
   {
      for (const auto& featureSet : image.features)
      {
         if (!featureSet.empty()) trainer.add(featureSet);
      }
   }

   // Compute the clusters.
   clusters_ = trainer.cluster();
}

bool TreeDetectorTrainer::compute_(std::string_view folder)
{
   std::atomic<size_t> treeCount{ 0 };    // Total amount of trees processed.
   std::atomic<size_t> nonTreeCount{ 0 }; // Total amount of non-tree cells processed.
   index_ = 0;                            // Reset the training data index.

   unsigned int num_threads{ std::thread::hardware_concurrency() };
   num_threads = std::max(1U, num_threads);
   Log::info_d("Running on %d threads.", num_threads);

   std::vector<std::thread> threads;
   threads.reserve(num_threads);
   for (int i = 0; i < num_threads; ++i)
   {
      threads.emplace_back(
         std::thread([this, folder, &treeCount, &nonTreeCount]() {
            this->extractFeatures_(folder, treeCount, nonTreeCount);
         }));
   }
   for (auto& thread : threads)
   {
      thread.join();
   }

   if (treeCount == 0)
   {
      Log::error("No tree features computed.");
      return false;
   }
   if (nonTreeCount == 0)
   {
      Log::error("No non-tree features computed.");
      return false;
   }

   Log::info("Computed features on %d trees.", treeCount.load());
   Log::info("Computed features on %d non-tree cells.", nonTreeCount.load());
   return true;
}

void TreeDetectorTrainer::extractFeatures_(
   std::string_view     folder,
   std::atomic<size_t>& treeCount,
   std::atomic<size_t>& nonTreeCount)
{
   auto sift = cv::xfeatures2d::SIFT::create(max_features);
   for (size_t i = index_++; i < trainingData_.size(); i = index_++)
   {
      // Open the image if possible, otherwise skip it.
      cv::String path{ folder.data() + trainingData_[i].file };
      Image      img{ cv::imread(path) };
      if (img.empty())
      {
         Log::warn("Failed to open image %s.", path.c_str());
         continue;
      }

      // Image pre-processing.
      //img.equaliseHistogram();
      std::pair scalingFactor{
         static_cast<float>(img.image().cols) / img_size.width,
         static_cast<float>(img.image().rows) / img_size.height
      };
      img.resize(img_size);

      Log::info("Computing features on image %s.", trainingData_[i].file.c_str());
      // If there are trees in the image
      if (!trainingData_[i].trees.empty())
      {
         for (const auto& tree : trainingData_[i].trees)
         {
            cv::Mat treeImg{ getTree(img.image(), tree, scalingFactor) };

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat&                  descriptors = trainingData_[i].features.emplace_back();
            sift->detectAndCompute(treeImg, cv::Mat{}, keypoints, descriptors);
            if (!keypoints.empty()) ++treeCount;
         }
      }
      else
      {
         using Cell = decltype(pyramid_)::Cell;
         auto analyseCell = [this, sift, &img, &nonTreeCount, i](const Cell* node) {
            cv::Mat cellImg{ getTree(img.image(), node->rect, std::pair{ 1.0F, 1.0F }) };

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat&                  descriptors = trainingData_[i].features.emplace_back();
            sift->detectAndCompute(cellImg, cv::Mat{}, keypoints, descriptors);
            if (!keypoints.empty()) ++nonTreeCount;
         };
         pyramid_.visit(analyseCell);
      }
   }
}

void TreeDetectorTrainer::histogramWorker_(
   std::string_view     folder,
   std::atomic<size_t>& treeCount,
   std::atomic<size_t>& nonTreeCount)
{
   auto sift = cv::xfeatures2d::SIFT::create(max_features);
   auto matcher = cv::BFMatcher::create(cv::NORM_L2);

   cv::BOWImgDescriptorExtractor bowExtractor{ sift, matcher };
   bowExtractor.setVocabulary(clusters_);

   for (size_t i = index_++; i < trainingData_.size(); i = index_++)
   {
      Log::info("Computing histogram for image %s.", trainingData_[i].file.c_str());
      cv::String path{ folder.data() + trainingData_[i].file };
      Image      img{ cv::imread(path) };
      if (img.empty())
      {
         Log::error("Failed to open image %s.", trainingData_[i].file.c_str());
         continue;
      }

      // Preprocessing.
      std::pair scalingFactor{
         static_cast<float>(img.image().cols) / img_size.width,
         static_cast<float>(img.image().rows) / img_size.height
      };
      img.resize(img_size);
      //img.equaliseHistogram();

      // If there are trees in the image.
      if (!trainingData_[i].trees.empty())
      {
         for (const auto& tree : trainingData_[i].trees)
         {
            if (updateHistogram_(avgTreeHist_, sift, bowExtractor, img, tree, scalingFactor))
               ++treeCount;
         }
      }
      // If there are not trees in the image.
      else
      {
         using Cell = decltype(pyramid_)::Cell;

         auto computeHist = [this, &img, sift, &bowExtractor, &nonTreeCount](const Cell* node) {
            if (updateHistogram_(avgNonTreeHist_, sift, bowExtractor, img, node->rect, std::pair{ 1.0F, 1.0F }))
               ++nonTreeCount;
         };
         pyramid_.visit(computeHist);
      }
   }
}

bool TreeDetectorTrainer::parse_(std::string_view cfgFile)
{
   std::ifstream dataset{ cfgFile.data() };
   if (!dataset.is_open())
   {
      Log::error("Failed to open configuration file %s.", cfgFile.data());
      return false;
   }

   int i = -1;
   while (dataset.good() && !dataset.eof())
   {
      std::string line;
      std::getline(dataset, line);
      // An image FILE is specified in the current line.
      if (size_t token{ line.find(file_token) }; token != std::string::npos)
      {
         ++i;
         TrainingImage image{};
         image.file = line.substr(token + file_token.size() + 1, line.size());
         Log::info_d("Adding image %s.", image.file.c_str());
         trainingData_.emplace_back(std::move(image));
      }
      // A tree's rectangle is specified in the current line.
      else
      {
         std::array<int, Rect<int>::fields> tree;
         size_t                             separator{ 0 };
         for (int j = 0; j < Rect<int>::fields; ++j)
         {
            size_t newSeparator = line.find(int_separator, separator);
            tree[j] = std::atoi(line.substr(separator, newSeparator - separator).c_str());
            separator = newSeparator + 1;
         }
         Log::info_d("Adding tree: (%d, %d, %d, %d).", tree[0], tree[1], tree[2], tree[3]);
         trainingData_[i].trees.emplace_back(tree);
      }
   }
}

bool TreeDetectorTrainer::save_(std::string_view file)
{
   cv::FileStorage output(file.data(), cv::FileStorage::WRITE);
   if (!output.isOpened())
   {
      Log::error("Failed to open file %s.", file.data());
      return false;
   }
   output << xml_words.data() << clusters_;
   output << xml_tree_hist.data() << avgTreeHist_.first;
   output << xml_nontree_hist.data() << avgNonTreeHist_.first;

   return true;
}

bool prj::TreeDetectorTrainer::updateHistogram_(
   Histogram&                     hist,
   cv::Ptr<cv::xfeatures2d::SIFT> sift,
   cv::BOWImgDescriptorExtractor& bowExtractor,
   const Image&                   img,
   Rect<int>                      rect,
   std::pair<float, float>        scalingFactor)
{
   std::vector<cv::KeyPoint> keypoints;

   cv::Mat mat{ getTree(img.image(), rect, scalingFactor) };
   sift->detect(mat, keypoints);

   if (keypoints.size() >= min_features)
   {
      cv::Mat                       descriptor;
      std::vector<std::vector<int>> currentHistogram;
      bowExtractor.compute(mat, keypoints, descriptor, &currentHistogram);

      std::scoped_lock lck{ hist.second };
      for (int i = 0; i < num_words; ++i)
      {
         hist.first.at<float>(i) += currentHistogram[i].size();
      }

      return true;
   }

   return false;
}
