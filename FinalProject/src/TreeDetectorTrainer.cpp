#include "TreeDetectorTrainer.hpp"

#include <algorithm>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <thread>

using namespace prj;

const cv::Size         TreeDetectorTrainer::img_size{ 500, 500 };
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

void TreeDetectorTrainer::buildVocabulary_()
{
   cv::BOWKMeansTrainer trainer{ num_features, cluster_criteria };

   // Add all descriptors to the trainer.
   for (const auto& image : trainingData_)
   {
      for (const auto& tree : image.features)
      {
         trainer.add(tree);
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
               cv::Mat treeImg{ mat(
                  cv::Range{
                     cvRound(tree.y / scalingFactor.second),
                     cvRound((tree.y + tree.h) / scalingFactor.second) },
                  cv::Range{
                     cvRound(tree.x / scalingFactor.first),
                     cvRound((tree.x + tree.w) / scalingFactor.first) }) };

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
   std::ofstream output(file.data());
   if (!output.is_open())
   {
      Log::error("Failed to open file %s.", file.data());
      return false;
   }

   output << clusters_;

   return true;
}
