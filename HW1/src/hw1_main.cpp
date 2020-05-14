#include <iostream>
#include <opencv2/highgui.hpp>
#include <string_view>

#include "hw1_image_proc.hpp"

/* Ordered list of input arguments for the program. */
enum class Argument
{
    INPUT_IMAGE = 1,
    TOT
};

int main(int argc, char* argv[])
{
    /* Check the number of arguments. */
    if (argc < static_cast<int>(Argument::TOT))
    {
        std::cout << "Insert at least " << static_cast<int>(Argument::TOT) - 1 << " arguments.\n";
        return -1;
    }

    /* Load the input image. */
    cv::Mat srcImg{ cv::imread(argv[static_cast<int>(Argument::INPUT_IMAGE)]) };
    if (srcImg.empty())
    {
        std::cout << "Failed to load image \"" << argv[static_cast<int>(Argument::INPUT_IMAGE)] << "\".\n";
        return -1;
    }

    /* Process the image. */
    HW1 imageProcessor(srcImg);
    imageProcessor.run();

    return 0;
}