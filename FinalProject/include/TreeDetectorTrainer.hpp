#ifndef TREEDETECTORTRAINER_HPP
#define TREEDETECTORTRAINER_HPP

#include "Image.hpp"
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
      using Histogram = std::pair<cv::Mat, std::mutex>;

      // Data about an image of a tree.
      struct TrainingImage
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
      // Name of the output file.
      static constexpr std::string_view output_file{ "trees.xml" };
      // Size required to normalise all images.
      static const cv::Size img_size;
      // Minimum number of features for training.
      static constexpr int min_train_features{ 128 };
      // Image pyramid constants for non-tree analysis.
      static constexpr int pyr_depth{ 2 };
      static constexpr int pyr_children{ 4 };
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
      // Build the average word histogram of a tree.
      void buildHistogram_(std::string_view folder);
      // Build the bag of words vocabulary.
      void buildVocabulary_();
      // Compute features for all trees in the dataset.
      bool compute_(std::string_view folder);
      // Extract tree features from the training data.
      void extractFeatures_(
         std::string_view     folder,
         std::atomic<size_t>& treeCount,
         std::atomic<size_t>& nonTreeCount);
      // Compute the BOW histogram.
      void histogramWorker_(
         std::string_view     folder,
         std::atomic<size_t>& treeCount,
         std::atomic<size_t>& nonTreeCount);
      // Parse the configuration file.
      bool parse_(std::string_view cfgFile);
      // Save the current training output.
      bool save_(std::string_view file);
      // Update one of the average histograms.
      static bool updateHistogram_(
         Histogram&                     hist,
         cv::Ptr<cv::xfeatures2d::SIFT> sift,
         cv::BOWImgDescriptorExtractor& bowExtractor,
         const Image&                   img,
         Rect<int>                      rect,
         std::pair<float, float>        scalingFactor);

      /********** VARIABLES **********/
      std::vector<TrainingImage> trainingData_;      // Training input and partial output.
      std::atomic<size_t>        index_{ 0 };        // Thread-safe index.
      cv::Mat                    treeVocabulary_;    // Clusters of tree features.
      cv::Mat                    nonTreeVocabulary_; // Clusters of non-tree features.
      Histogram                  avgTreeHist_;       // Average tree word histogram.
      Histogram                  avgNonTreeHist_;    // Average non-tree word histogram.
      // Areas to analyse in images that don't contain trees.
      ImagePyramid<pyr_children, pyr_depth> pyramid_{ img_width, img_height };
   };
} // namespace prj

#endif