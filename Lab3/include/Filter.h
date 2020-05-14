#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Generic class implementing a filter with the input and output image data and the parameters
class Filter
{
// Methods
public:
	// constructor 
	// input_img: image to be filtered
	// filter_size : size of the kernel/window of the filter
	Filter(cv::Mat input_img, int filter_size);

	// perform filtering (in base class do nothing, to be reimplemented in the derived filters)
	void doFilter();

	// get the output of the filter
	cv::Mat getResult();

	//set the window size (square window of dimensions size x size)
	void setSize(int size);
	
	//get the Window Size
	int getSize();

// Data
protected:
	// input image
	cv::Mat input_image;
	// output image (filter result)
	cv::Mat result_image;
	// window size
	int filter_size;
};

// Gaussian Filter
class GaussianFilter : public Filter  
{
public:
	GaussianFilter(const cv::Mat& input_img, int filter_size);

	void doFilter();
	void setParam(double sigma); // Set the sigma value for the filter.

private:
	double sigma_{ 0.0 }; // Sigma value for the filter.
};

class MedianFilter : public Filter 
{
public:
	MedianFilter(const cv::Mat& input_img, int filter_size);

	void doFilter();
};

class BilateralFilter : public Filter 
{
public:
	BilateralFilter(const cv::Mat& input_img, int filter_size);

	void doFilter();
	void setParams(double sRange, double sSpace); // Set both sigma values for the filter.

private:
	double sigmaRange_{ 0.0 }; // Colour sigma value.
	double sigmaSpace_{ 0.0 }; // Spatial sigma value.
};