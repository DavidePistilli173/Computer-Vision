#include "lab4.hpp"
#include "StreetAnaliser.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

/* List of command line arguments. */
enum class Argument
{
    input = 1,
    tot
};

using lab4::Log;

int main(int argc, char* argv[])
{
    /* Argument check. */
    if (argc < static_cast<int>(Argument::tot))
    {
        Log::fatal("Specify an image to process.");
        return -1;
    }

    /* Load the image. */
    Log::info("Loading image %s.", argv[static_cast<int>(Argument::input)]);
    cv::Mat img{ cv::imread(argv[static_cast<int>(Argument::input)]) };
    if (img.empty())
    {
        Log::fatal("FAILED");
        return -1;
    }
    Log::info("SUCCESS");

    Log::info("Initialising street analyser.");
    lab4::StreetAnaliser strtAnlzr{ img };
    Log::info("SUCCESS");

    Log::info("Running street analyser.");
    strtAnlzr.run();
    Log::info("Analysis complete.");

    return 0;
}