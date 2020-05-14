#include "lab2.hpp"

#include <sstream>

using namespace lab2;

std::mutex Log::mtx_;

void CalibrationData::reset()
{
    camera = cv::Mat::eye(3, 3, CV_64F);
    distParams.clear();
    rotVecs.clear();
    tVecs.clear();
}