#include "lab5.hpp"
#include "PanoramicImage.hpp"

/* List of command line arguments. */
enum class Argument
{
    folder = 1,
    fov,
    ratio,
    tot
};

constexpr std::string_view win_name{ "Panorama" }; // Name of the output window.

using lab5::Log;

int main(int argc, char* argv[])
{
    /* Input argument check. */
    if (argc < static_cast<int>(Argument::tot))
    {
        Log::fatal("Required parameters: <folder name> <camera fov> <ratio>.");
        return -1;
    }

    lab5::PanoramicImage panImg;
    
    Log::info("Loading input images.");
    if (!panImg.loadImages(argv[static_cast<int>(Argument::folder)]))
    {
        Log::fatal("Loading failed.");
    }
    Log::info("Loading complete.");

    Log::info("Performing cylindrical projection.");
    panImg.projectImages(static_cast<int>(std::atoi(argv[static_cast<int>(Argument::fov)])/2.F));
    Log::info("Projection complete.");

    Log::info("Extracting ORB features.");
    panImg.extractORB();
    Log::info("Feature extraction complete.");

    Log::info("Matching keypoint descriptors.");
    if (!panImg.computeMatches(std::atof(argv[static_cast<int>(Argument::ratio)])))
    {
        Log::fatal("Failed to compute matches.");
        return -1;
    }
    Log::info("Matching complete.");

    Log::info("Computing final panoramic image.");
    lab5::Window win(win_name);
    win.showImg(panImg.computePanorama());
    cv::waitKey(0);

    return 0;
}