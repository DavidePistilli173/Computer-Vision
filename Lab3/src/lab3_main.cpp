#include "FilterComparator.hpp"
#include "HistEqualiser.hpp"
#include "lab3.hpp"
#include "Morph.hpp"

#include <opencv2/imgcodecs.hpp>

#include <string_view>

/* List of command line arguments. */
enum class Argument
{
    img = 1,
    tot
};

constexpr std::string_view morph_win_name{ "Morphological operator" };

using lab3::Log;

/* Analyse the histogram of an image and equalise it. */
cv::Mat equalise(const cv::Mat& srcImg);
/* Compare different types of filters on the same image. */
void filterComparison(const cv::Mat& img);

int main(int argc, char* argv[])
{
    /* Check input arguments. */
    if (argc < static_cast<int>(Argument::tot))
    {
        Log::fatal("Specify an image to process.");
        return -1;
    }

    Log::info("Loading input image...");
    cv::Mat srcImg{ cv::imread(argv[static_cast<int>(Argument::img)]) };
    if (srcImg.empty())
    {
        Log::fatal("Failed to read image %s.", argv[static_cast<int>(Argument::img)]);
        return -1;
    }
    Log::info("Input image loaded.");

    Log::info("Beginning equalisation...");
    cv::Mat equalisedImg{ equalise(srcImg) };
    Log::info("Equalisation complete.");

    Log::info("Beginning filter comparison.");
    filterComparison(equalisedImg);
    Log::info("Filter comparison complete.");

    Log::info("Starting to apply morphological operators.");
    lab3::Morph morphOp{ equalisedImg };
    morphOp.execute();
    lab3::Window morphWin{ morph_win_name };
    morphWin.showImg(morphOp.getResult());
    cv::waitKey(0);
    Log::info("Program complete.");

    return 0;
}

cv::Mat equalise(const cv::Mat& srcImg)
{
    Log::info("Computing histograms...");
    lab3::HistEqualiser histEq{ srcImg };
    Log::info("Histograms computed.");
    Log::info("Displaying results.");
    histEq.show();

    Log::info("Equalising image (RGB)...");
    histEq.equalise();
    Log::info("Image equalised.");
    Log::info("Displaying results.");
    histEq.show();

    Log::info("Equalising image (HSV)...");
    for (int i = 0; i < static_cast<int>(lab3::HSV::tot); ++i)
    {
        histEq.setMode(lab3::HistEqualiser::Mode::hsv);
        Log::info("Equalising on channel %d...", i);
        histEq.equalise(static_cast<lab3::HSV>(i));
        Log::info("Image equalised.");
        histEq.setMode(lab3::HistEqualiser::Mode::rgb);
        Log::info("Displaying results.");
        histEq.show();
        Log::info("Resetting equaliser.");
        histEq.reset();
    }

    Log::info("Equalising on V channel.");
    histEq.setMode(lab3::HistEqualiser::Mode::hsv);
    histEq.equalise(lab3::HSV::v);
    histEq.setMode(lab3::HistEqualiser::Mode::rgb);
    return histEq.getResult();
}

void filterComparison(const cv::Mat& img)
{
    Log::info("Starting filter comparator.");
    lab3::FilterComparator filtComp(img);
    Log::info("Running filter comparator.");
    filtComp.run();
}
