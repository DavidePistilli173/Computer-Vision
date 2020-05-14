#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <sstream>
#include <string_view>

#include "CameraCalibrator.hpp"

/* List of command line arguments. */
enum class Argument
{
    input_folder = 1,
    test_img,
    tot
};

constexpr std::string_view file_pattern{ "*.png" }; // Pattern for file retrieval.
constexpr std::string_view win_original{ "Test image" }; // Output window.

constexpr lab2::Size cell_size{ 0.11F, 0.11F }; // Size of each pattern cell.
constexpr lab2::Size pattern_size{ 6, 5 }; // Dimensions of the pattern grid.

using lab2::Log;

/* Load the calibration images and a test image. */
bool loadImages(
    const std::vector<cv::String>& calibFiles, std::vector<cv::Mat>& calibImgs,
    const cv::String& testFile, cv::Mat& testImg
);

int main(int argc, char* argv[])
{
    /* Check the number of arguments. */
    if (argc < static_cast<int>(Argument::tot))
    {
        Log::fatal("Required parameters: <folder with calibration images> <test image>");
        return -1;
    }

    /* Load the list of calibration images. */
    std::vector<cv::String> calibrationImageFiles;
    cv::utils::fs::glob(
        argv[static_cast<int>(Argument::input_folder)],
        file_pattern.data(), 
        calibrationImageFiles
    );

    /* Load the actual images. */
    std::vector<cv::Mat> calibrationImages;
    cv::Mat testImg;
    if (!loadImages(calibrationImageFiles, calibrationImages, argv[static_cast<int>(Argument::test_img)], testImg))
    {
        Log::fatal("Failed to load input images.");
        return -1;
    }

    /* Create the camera calibrator. */
    Log::info("Initialising camera calibrator...");
    lab2::CameraCalibrator calibrator;
    if (!calibrator.setCalibImgs(&calibrationImages, cell_size, pattern_size))
    {
        Log::fatal("FAIL");
        return -1;
    }
    Log::info("SUCCESS");

    Log::info("Starting calibration.");
    calibrator.calibrate();
    Log::info("Calibration complete.");
    calibrator.printResults();

    /* Compute the errors and vest/worst images. */
    lab2::CalibrationStats results{ calibrator.meanProjErr() };
    Log::info("Mean reprojection error: %f", results.meanRE);
    Log::info("Best image: \"%s\"; error: %f", calibrationImageFiles[results.minRE.indx].c_str(), results.minRE.err);
    Log::info("Worst image: \"%s\"; error: %f", calibrationImageFiles[results.maxRE.indx].c_str(), results.maxRE.err);

    Log::info("Undistorting test image...");
    cv::Mat undistortedTestImg{ testImg.clone() };
    calibrator.undistort(undistortedTestImg);
    Log::info("SUCCESS");
    
    Log::info("Preparing output...");
    cv::resize(testImg, testImg, testImg.size() / 3);
    cv::resize(undistortedTestImg, undistortedTestImg, undistortedTestImg.size() / 3);
    cv::hconcat(testImg, undistortedTestImg, testImg);
    Log::info("SUCCESS");

    Log::info("Displaying risults.");
    cv::namedWindow(win_original.data());
    cv::imshow(win_original.data(), testImg);
    cv::waitKey(0);
}

bool loadImages(
    const std::vector<cv::String>& calibFiles, std::vector<cv::Mat>& calibImgs,
    const cv::String& testFile, cv::Mat& testImg
    )
{
    Log::info("Loading calibration images.");
    for (const auto& fileName : calibFiles)
    {
        /* If an image was not loaded correctly, exit. */
        Log::info("Loading image \"%s\".", fileName.c_str());
        if (auto lastImg = calibImgs.emplace_back(cv::imread(fileName, cv::IMREAD_GRAYSCALE));
            lastImg.empty())
        {
            Log::error("Failed to load image \"%s\".", fileName);
            return false;
        }
    }
    Log::info("Loaded calibration images.");

    /* Load the test image. */
    testImg = cv::imread(testFile);
    if (testImg.empty())
    {
        Log::error("Failed to load test image \"%s\".", testFile);
        return false;
    }
    Log::info("Loaded test image.");

    return true;
}

