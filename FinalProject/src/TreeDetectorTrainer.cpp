#include "TreeDetectorTrainer.hpp"

#include <algorithm>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <thread>

using namespace prj;

const cv::Size         TreeDetectorTrainer::img_size{ 1000, 1000 };
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
   auto sift = cv::xfeatures2d::SIFT::create(num_features);
   auto matcher = cv::BFMatcher::create(cv::NORM_L2);

   avgHistogram_ = cv::Mat::zeros(num_words, 1, CV_32F);

   cv::BOWImgDescriptorExtractor bowExtractor{ sift, matcher };
   bowExtractor.setVocabulary(clusters_);

   int count{ 0 };
   for (const auto& image : trainingData_)
   {
      if (!image.trees.empty())
      {
         Log::info("Computing histogram for image %s.", image.file.c_str());
         cv::String path{ folder.data() + image.file };
         cv::Mat    mat{ cv::imread(path) };
         if (mat.empty())
         {
            Log::error("Failed to open image %s.", image.file.c_str());
         }
         else
         {
            std::pair scalingFactor{
               static_cast<float>(mat.cols) / img_size.width,
               static_cast<float>(mat.rows) / img_size.height
            };
            cv::resize(mat, mat, img_size);
            for (const auto& tree : image.trees)
            {
               ++count;
               std::vector<cv::KeyPoint>     keypoints;
               cv::Mat                       descriptor;
               std::vector<std::vector<int>> currentHistogram;

               cv::Mat treeImg{ getTree(mat, tree, scalingFactor) };
               sift->detect(treeImg, keypoints);
               bowExtractor.compute(treeImg, keypoints, descriptor, &currentHistogram);

               for (int i = 0; i < num_words; ++i)
               {
                  avgHistogram_.at<float>(i) += currentHistogram[i].size();
               }
            }
         }
      }
   }
   avgHistogram_ /= count;
   cv::normalize(avgHistogram_, avgHistogram_, 1.0, 0.0, cv::NORM_L1);
}

void TreeDetectorTrainer::buildVocabulary_()
{
   cv::BOWKMeansTrainer trainer{ num_words, cluster_criteria };

   // Add all descriptors to the trainer.
   for (const auto& image : trainingData_)
   {
      for (const auto& trees : image.features)
      {
         trainer.add(trees);
      }
   }

   // Compute the clusters.
   clusters_ = trainer.cluster();
}

bool TreeDetectorTrainer::compute_(std::string_view folder)
{
   std::atomic<size_t> count{ 0 }; // Total amount of trees processed.
   index_ = 0;                     // Reset the training data index.

   unsigned int num_threads{ std::thread::hardware_concurrency() };
   num_threads = std::max(1U, num_threads);
   Log::info_d("Running on %d threads.", num_threads);

   std::vector<std::thread> threads;
   threads.reserve(num_threads);
   for (int i = 0; i < num_threads; ++i)
   {
      threads.emplace_back(
         std::thread([this, folder, &count]() { this->extractFeatures_(folder, count); }));
   }
   for (auto& thread : threads)
   {
      thread.join();
   }

   if (count == 0)
   {
      Log::error("No features computed.");
      return false;
   }

   Log::info("Computed features on %d trees.", count.load());
   return true;
}

void TreeDetectorTrainer::extractFeatures_(std::string_view folder, std::atomic<size_t>& count)
{
   auto sift = cv::xfeatures2d::SIFT::create(num_features);
   for (size_t i = index_++; i < trainingData_.size(); i = index_++)
   {
      if (!trainingData_[i].trees.empty())
      {
         cv::String path{ folder.data() + trainingData_[i].file };
         cv::Mat    mat{ cv::imread(path) };
         if (mat.empty())
         {
            Log::warn("Failed to open image %s.", path.c_str());
         }
         else
         {
            Log::info("Computing features on image %s.", trainingData_[i].file.c_str());
            std::pair scalingFactor{
               static_cast<float>(mat.cols) / img_size.width,
               static_cast<float>(mat.rows) / img_size.height
            };
            cv::resize(mat, mat, img_size);
            for (const auto& tree : trainingData_[i].trees)
            {
               ++count;
               cv::Mat treeImg{ getTree(mat, tree, scalingFactor) };

               std::vector<cv::KeyPoint> keypoints;
               cv::Mat&                  descriptors = trainingData_[i].features.emplace_back();
               sift->detectAndCompute(treeImg, cv::Mat{}, keypoints, descriptors);
            }
         }
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
         TreeImage image{};
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
   output << xml_hist.data() << avgHistogram_;

   return true;
}
