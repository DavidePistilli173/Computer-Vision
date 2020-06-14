#include "TreeDetector.hpp"
#include "TreeDetectorTrainer.hpp"
#include "utility.hpp"

#include <opencv2/imgproc.hpp>

// Command line arguments.
enum class Arg
{
   mode = 1, // Execution mode: train / detect.
   file,     // Dataset text file / Training result.
   img,      // Dataset folder / Input image.
   tot       // Total number of arguments.
};

constexpr std::string_view train_mode{ "train" };   // Training mode keyword.
constexpr std::string_view detect_mode{ "detect" }; // Detection mode keyword.
constexpr std::string_view output_win{ "Trees" };   // Output window name.

using prj::Log;

int main(int argc, char* argv[])
{
   // Input check.
   if (argc < static_cast<int>(Arg::tot))
   {
      Log::fatal("Wrong number of input arguments.");
      return 1;
   }

   // Training mode.
   if (argv[static_cast<int>(Arg::mode)] == train_mode)
   {
      prj::TreeDetectorTrainer trainer;
      Log::info("Training started.");
      if (!trainer.train(argv[static_cast<int>(Arg::file)], argv[static_cast<int>(Arg::img)]))
      {
         Log::fatal("Failed to train the detector.");
         Log::fatal("Configuration file: %s.", argv[static_cast<int>(Arg::file)]);
         Log::fatal("Dataset folder: %s.", argv[static_cast<int>(Arg::img)]);
         return 1;
      }
   }
   // Detection mode.
   else if (argv[static_cast<int>(Arg::mode)] == detect_mode)
   {
      // Read the input image.
      Log::info("Loading input image.");
      cv::Mat img{ cv::imread(argv[static_cast<int>(Arg::img)]) };
      if (img.empty())
      {
         Log::fatal("Failed to open image %s.", argv[static_cast<int>(Arg::img)]);
         return 1;
      }

      // Initialise the tree detector.
      Log::info("Initialising the tree detector.");

      prj::TreeDetector td;
      try
      {
         td = prj::TreeDetector(argv[static_cast<int>(Arg::file)]);
      }
      catch (std::exception)
      {
         Log::fatal("Failed to initialise tree detector.");
         return 1;
      }

      // Run the tree detector.
      Log::info("Detecting trees.");
      cv::Mat result{ td.detect(img) };
      if (result.empty())
      {
         Log::fatal("Failed to detect trees.");
         return 1;
      }
      Log::info("Detection complete.");

      prj::Window output{ output_win };
      output.showImg(result);
      cv::waitKey(0);
   }

   return 0;
}