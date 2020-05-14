#ifndef HW1_IMAGE_PROC
#define HW1_IMAGE_PROC

#include <mutex>
#include <opencv2/core.hpp>
#include <string_view>

#include "../include/hw1.hpp"

class HW1
{
public:
    HW1(const cv::Mat& input);

    /********** CONSTANTS **********/
    /* Rendering delay in ms. */
    static constexpr int RENDER_DELAY = 20;

    /* Name of the window that will hold the image processed through HSV. */
    static constexpr std::string_view HSV_WIN_NAME{ "HSV image" };
    /* Name of thw /* Name of the window that will hold the input image. */
    static constexpr std::string_view INPUT_WIN_NAME{ "Input image" };
    /* Name of the window that will hold the image processed through RGB. */
    static constexpr std::string_view RGB_WIN_NAME{ "RGB image" };

    /* Threshold for including different colours in the transformation. */
    static constexpr float COLOUR_THRESHOLD{ 0.295f };
    /* Destination value for the involved pixels. */
    static constexpr uchar NEW_RGB_VAL[]{ 201, 37, 92 };
    /* Hue offset to use during HSV processing. */
    static constexpr uchar HUE_OFFSET{ 30 };

    /* Pixel range around the target pixel that will contribute to the average colour. */
    static constexpr hw1::Range<int> TARGET_NEIGHBORHOOD{ -4, 4 };
    /* Number of pixels considered when computing the average colour. */
    static constexpr int PIXEL_NUM = hw1::pow((TARGET_NEIGHBORHOOD.max - TARGET_NEIGHBORHOOD.min + 1), 2);

    /********** METHODS **********/
    /* Window and input handler. */
    void run();

private:
    /********** METHODS **********/
    /* Check whether a given colour is within a specified range. */
    bool isInRGBRange_(const hw1::vec3uc_t& colour, const std::vector<hw1::Range<uchar>>& range); // RGB check.
    bool isInHRange_(const hw1::vec3uc_t& colour, const hw1::Range<uchar> range); // Hue check.
    /* Mouse event callback. */
    static void handleMouseEvents_(int event, int x, int y, int flags, void* userdata);
    void processRGB_(cv::Point2i target); // Process the image through RGB.
    void processHSV_(cv::Point2i target); // Process the image through HSV.

    /********** VARIABLES **********/
    cv::Mat srcImg_; // Input image.
    cv::Mat rgbImg_; // Image processed through RGB.
    cv::Mat hsvImg_; // Image processed through HSV.
    std::mutex imgProcMutex_; // Mutex used to guard changes to rgbImg and hsvImg.
};

#endif