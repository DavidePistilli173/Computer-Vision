#ifndef TREEDETECTORTRAINER_HPP
#define TREEDETECTORTRAINER_HPP

#include "utility.hpp"

#include <atomic>
#include <opencv2/core.hpp>
#include <string_view>
#include <vector>

namespace prj
{
   class TreeDetectorTrainer
   {
   public:
      // Data about an image of a tree.
      struct TreeImage
      {
         cv::String             file;     // Name of the file containing the image.
         std::vector<Rect<int>> trees;    // List of trees in the image.
         std::vector<cv::Mat>   features; // List of features of the image.
      };

      /********** CONSTANTS **********/
      // Token that precedes a file name.
      static constexpr std::string_view file_token{ "FILE" };
      // Separator for rectangle data.
      static constexpr std::string_view int_separator{ "," };
      static constexpr std::string_view output_file{ "trees.dat" };
      // Number of features for each tree.
      static constexpr int num_features{ 1000 };
      // Size required to normalise all images.
      static const cv::Size img_size;
      // Terminating criteria for the kmeans clustering.
      static const cv::TermCriteria cluster_criteria;
      // Number of clustering iterations.
      static constexpr int num_cluster_iter{ 200 };

      /********** CONSTRUCTOR **********/
      TreeDetectorTrainer() = default;

      /********** METHODS **********/
      // Train the tree detector.
      bool train(std::string_view cfgFile, std::string_view imgFolder);

   private:
      /********** METHODS **********/
      // Build the bag of words vocabulary.
      void buildVocabulary_();
      // Compute features for all trees in the dataset.
      bool compute_(std::string_view folder);
      // Extract tree features from the training data.
      void extractFeatures_(std::string_view folder, std::atomic<size_t>& count);
      // Parse the configuration file.
      bool parse_(std::string_view cfgFile);
      // Save the current training output.
      bool save_(std::string_view file);

      /********** VARIABLES **********/
      std::vector<TreeImage> trainingData_; // Training input and partial output.
      std::atomic<size_t>    index_{ 0 };   // Thread-safe index.
      cv::Mat                clusters_;     // Feature clusters.
   };
} // namespace prj

#endif