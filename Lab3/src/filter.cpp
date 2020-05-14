#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Filter.h"

// constructor
Filter::Filter(cv::Mat input_img, int size) 
{
	input_image = input_img;
	if (size % 2 == 0)
		size++;
	filter_size = size;
}

// for base class do nothing (in derived classes it performs the corresponding filter)
void Filter::doFilter() 
{
	// it just returns a copy of the input image
	result_image = input_image.clone();
}

// get output of the filter
cv::Mat Filter::getResult() 
{
	return result_image;
}

//set window size (it needs to be odd)
void Filter::setSize(int size) 
{
	if (size % 2 == 0)
		size++;
	filter_size = size;
}

//get window size 
int Filter::getSize() 
{
	return filter_size;
}

// Write your code to implement the Gaussian, median and bilateral filters
GaussianFilter::GaussianFilter(const cv::Mat& input_img, int filter_size) :
	Filter{ input_img, filter_size }
{}

void GaussianFilter::doFilter()
{
	if (sigma_ == 0) Filter::doFilter();
	else cv::GaussianBlur(input_image, result_image, cv::Size{ filter_size, filter_size }, sigma_);
}

void GaussianFilter::setParam(double sigma)
{
	if (sigma >= 0) sigma_ = sigma;
}

MedianFilter::MedianFilter(const cv::Mat& input_img, int filter_size) :
	Filter{ input_img, filter_size }
{}

void MedianFilter::doFilter()
{
	cv::medianBlur(input_image, result_image, filter_size);
}

BilateralFilter::BilateralFilter(const cv::Mat& input_img, int filter_size) :
	Filter{ input_img, filter_size }
{}

void BilateralFilter::doFilter()
{
	if (sigmaRange_ == 0 && sigmaSpace_ == 0) Filter::doFilter();
	else cv::bilateralFilter(input_image, result_image, filter_size, sigmaRange_, sigmaSpace_);
}

void BilateralFilter::setParams(double sRange, double sSpace)
{
	if (sRange >= 0) sigmaRange_ = sRange;
	if (sSpace >= 0) sigmaSpace_ = sSpace;
}
