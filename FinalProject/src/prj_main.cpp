#include "TreeDetector.hpp"
#include "utility.hpp"

#include <opencv2/imgproc.hpp>

// Command line arguments.
enum class Arg
{
   img = 1, // Input image.
   tot      // Total number of arguments.
};

using prj::Log;

int main(int argc, char* argv[])
{
   // Input check.
   if (argc < static_cast<int>(Arg::tot))
   {
      Log::fatal("Wrong number of input arguments.");
      return 1;
   }

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

   // Run the tree detector.
   Log::info("Detecting trees.");
   cv::Mat result{ td.detect(img) };
   if (result.empty())
   {
      Log::fatal("Failed to detect trees.");
      return 1;
   }
   Log::info("Detection complete.");

   return 0;
}