#include "lab5.hpp"
#include "PanoramicImage.hpp"

#include <thread>

/* List of command line arguments. */
enum class Argument
{
    folder = 1,
    fov,
    ratio,
    tot
};

/* Names of the output windows. */
constexpr std::string_view orb_win_name{ "ORB panorama" };
constexpr std::string_view sift_win_name{ "SIFT panorama" };

using lab5::Log;

/* Computation parameters. */
struct Params
{
    lab5::PanoramicImage::Mode mode;
    std::string_view folder;
    int fov;
    double ratio;
};

/* Compute a panoramic image. */
void compute(cv::Mat& result, Params params);

int main(int argc, char* argv[])
{
    /* Input argument check. */
    if (argc < static_cast<int>(Argument::tot))
    {
        Log::fatal("Required parameters: <folder name> <camera fov> <ratio>.");
        return -1;
    }

    /* Extract command line parameters. */
    std::string_view folder{ argv[static_cast<int>(Argument::folder)] };
    int fov{ static_cast<int>(std::atoi(argv[static_cast<int>(Argument::fov)]) / 2.F) };
    double ratio{ std::atof(argv[static_cast<int>(Argument::ratio)]) };

    /* Final images. */
    cv::Mat orbImg;
    cv::Mat siftImg;

    /* Start the computation. */
    std::thread orbTh{
        &compute,
        std::ref(orbImg),
        Params{ lab5::PanoramicImage::Mode::orb, folder, fov, ratio }
    };
    std::thread siftTh{
        &compute,
        std::ref(siftImg),
        Params{ lab5::PanoramicImage::Mode::sift, folder, fov, ratio }
    };

    /* Join threads. */
    orbTh.join();
    siftTh.join();

    /* Result check. */
    if (orbImg.empty())
    {
        Log::fatal("Failed to compute the panoramic image with ORB.");
        return -1;
    }
    if (siftImg.empty())
    {
        Log::fatal("Failed to compute the panoramic image with SIFT.");
        return -1;
    }

    /* Show ORB panorama. */
    lab5::Window orbWin{ orb_win_name };
    orbWin.showImg(orbImg);

    /* Show SIFT panorama. */
    lab5::Window siftWin{ sift_win_name };
    siftWin.showImg(siftImg);

    cv::waitKey(0);

    return 0;
}

void compute(cv::Mat& result, Params params)
{
    lab5::PanoramicImage panImg;

    Log::info("Loading input images.");
    if (!panImg.loadImages(params.folder))
    {
        Log::error("Loading failed.");
        return;
    }
    Log::info("Loading complete.");

    Log::info("Performing cylindrical projection.");
    if (!panImg.projectImages(params.fov))
    {
        Log::error("Projection failed.");
        return;
    }
    Log::info("Projection complete.");

    switch (params.mode)
    {
    case lab5::PanoramicImage::Mode::orb:
        Log::info("Extracting ORB features.");
        if (!panImg.extractORB())
        {
            Log::error("Failed to extract features.");
            return;
        }
        break;
    case lab5::PanoramicImage::Mode::sift:
        Log::info("Extracting SIFT features.");
        if (!panImg.extractSIFT())
        {
            Log::error("Failed to extract features.");
            return;
        }
        break;
    }
    Log::info("Feature extraction complete.");

    Log::info("Matching keypoint descriptors.");
    if (!panImg.computeMatches(params.ratio))
    {
        Log::error("Failed to compute matches.");
        return;
    }
    Log::info("Matching complete.");

    Log::info("Computing final panoramic image.");
    result = panImg.computePanorama();
}
