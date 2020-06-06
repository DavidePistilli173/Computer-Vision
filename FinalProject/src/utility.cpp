#include "utility.hpp"

#include <cstdint>
#include <limits>
#include <opencv2/imgproc.hpp>

using namespace prj;

std::mutex Log::mtx_;

Window::Window(std::string_view name) :
   name_{ name.data() }
{
   cv::namedWindow(name.data(), cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
   trckBarVals_.reserve(max_trckbar_num); // Reserve space for the maximum number of trackbars.
}

Window::~Window()
{
   cv::destroyWindow(name_);
}

bool Window::addTrackBar(std::string_view name, int maxVal)
{
   return addTrackBar(name, 0, maxVal);
}

bool Window::addTrackBar(std::string_view name, int startVal, int maxVal)
{
   if (trckBarVals_.size() == max_trckbar_num)
   {
      Log::warn("Maximum number of trackbars reached.");
      return false;
   }
   if (startVal > maxVal)
   {
      Log::warn("Initial trackbar value too high. Setting it to 0.");
      startVal = 0;
   }

   int* valPtr{ &trckBarVals_.emplace_back(startVal) };
   cv::createTrackbar(name.data(), name_, valPtr, maxVal, trckCallbck_, this);

   return true;
}

std::vector<int> Window::fetchTrckVals()
{
   /* Return the current values and reset the modification flag. */
   trckModified_ = false;
   return trckBarVals_;
}

bool Window::modified() const
{
   return trckModified_;
}

void Window::showImg(const cv::Mat& img) const
{
   cv::imshow(name_, img);
}

void Window::trckCallbck_(int val, void* ptr)
{
   Window* winPtr{ reinterpret_cast<Window*>(ptr) };
   winPtr->trckModified_ = true; // Set the modification flag.
}

Image::Image(const cv::Mat& mat) :
   mat_{ mat.clone() },
   labels_{ cv::Mat::zeros(cv::Size{ mat.cols, mat.rows }, CV_32SC1) }
{
   // Set the appropriate colour space.
   // Colour images are considered bgr.
   if (mat_.channels() == 1)
      colSpace_ = ColourSpace::grey;
   else
      colSpace_ = ColourSpace::bgr;
}

Image::Image(cv::Mat&& mat) :
   mat_{ mat },
   labels_{ cv::Mat::zeros(cv::Size{ mat.cols, mat.rows }, CV_32SC1) }
{
   // Set the appropriate colour space.
   // Colour images are considered bgr.
   if (mat_.channels() == 1)
      colSpace_ = ColourSpace::grey;
   else
      colSpace_ = ColourSpace::bgr;
}

Image::Image(const Image& img) :
   mat_{ img.mat_.clone() },
   labels_{ img.labels_.clone() },
   colSpace_{ img.colSpace_ } {}

Image& Image::operator=(const Image& img)
{
   if (&img == this) return *this;

   mat_ = img.mat_.clone();
   labels_ = img.labels_.clone();
   colSpace_ = img.colSpace_;
   return *this;
}

Image& Image::operator=(const cv::Mat& mat)
{
   mat_ = mat.clone();
   labels_ = cv::Mat::zeros(cv::Size{ mat.cols, mat.rows }, CV_32SC1);

   // Set the appropriate colour space.
   // Colour images are considered bgr.
   if (mat_.channels() == 1)
      colSpace_ = ColourSpace::grey;
   else
      colSpace_ = ColourSpace::bgr;

   return *this;
}

void Image::bilateralFilter(int size, double colour_sig, double space_sig)
{
   cv::Mat result;
   cv::bilateralFilter(
      mat_,
      result,
      size,
      colour_sig,
      space_sig);
   mat_ = result;
}

void prj::Image::canny(double th1, double th2)
{
   cv::Mat result;
   cv::Canny(mat_, result, th1, th2);
   mat_ = result;
}

void Image::connectedComponents()
{
   if (mat_.depth() == CV_8U)
      cv::connectedComponents(mat_, labels_, 8, CV_32S);
   else
   {
      cv::Mat mat8U;
      mat_.convertTo(mat8U, CV_8U);
      cv::connectedComponents(mat8U, labels_, 8, CV_32S);
   }
}

float Image::contrast()
{
   ColourSpace colourSpace{ colSpace_ };
   setColourSpace(ColourSpace::hsv);

   std::vector<cv::Mat> channels;
   cv::split(mat_, channels);

   cv::Mat kernel{ cv::Mat::ones(cv::Size{ 9, 9 }, CV_8U) };
   Image   min{ channels[static_cast<int>(HSV::v)] };
   Image   max{ channels[static_cast<int>(HSV::v)] };
   min.erode(kernel);
   max.dilate(kernel);
   min.convert(CV_32F);
   max.convert(CV_32F);

   cv::Mat    contrast{ (max.image() - min.image()) / (max.image() + min.image()) };
   cv::Scalar mean{ cv::mean(contrast) };

   setColourSpace(colourSpace);
   return mean[0];
}

void Image::convert(int type)
{
   cv::Mat result;
   mat_.convertTo(result, type);
   mat_ = result;
}

void Image::dilate(cv::Mat kernel)
{
   cv::dilate(mat_, mat_, kernel);
}

void Image::display(bool useLabels) const
{
   // The thread ID in the name is required to ensure thread-safe windows.
   std::stringstream s;
   s << "Thread " << std::this_thread::get_id();
   display(s.str(), useLabels);
}

void Image::display(std::string_view winName, bool useLabels) const
{
   Window win{ winName };

   if (!useLabels)
      win.showImg(mat_);
   else
   {
      cv::Mat                result = cv::Mat::zeros(cv::Size{ mat_.cols, mat_.rows }, CV_8UC3);
      std::vector<cv::Vec3b> colours;
      for (int i = 0; i < mat_.rows; ++i)
      {
         for (int j = 0; j < mat_.cols; ++j)
         {
            if (labels_.at<int>(i, j) != -1)
            {
               if (size_t prevSize{ colours.size() }; colours.size() <= labels_.at<int>(i, j))
               {
                  colours.resize(labels_.at<int>(i, j) + 1);
                  for (; prevSize < colours.size(); ++prevSize)
                  {
                     colours[prevSize] = {
                        static_cast<unsigned char>(cv::theRNG().uniform(0, std::numeric_limits<unsigned char>::max())),
                        static_cast<unsigned char>(cv::theRNG().uniform(0, std::numeric_limits<unsigned char>::max())),
                        static_cast<unsigned char>(cv::theRNG().uniform(0, std::numeric_limits<unsigned char>::max()))
                     };
                  }
               }
               result.at<cv::Vec3b>(i, j) = colours[labels_.at<int>(i, j)];
            }
         }
      }
      win.showImg(result);
   }

   cv::waitKey(0);
}

void Image::distanceTransform()
{
   cv::distanceTransform(mat_, mat_, cv::DIST_L2, cv::DIST_MASK_5);
}

void Image::erode(cv::Mat kernel)
{
   cv::erode(mat_, mat_, kernel);
}

void Image::equaliseHistogram()
{
   switch (colSpace_)
   {
   case ColourSpace::grey:
      cv::equalizeHist(mat_, mat_);
      break;
   case ColourSpace::bgr:
      setColourSpace(ColourSpace::hsv);
      equaliseHSV_();
      setColourSpace(ColourSpace::bgr);
      break;
   case ColourSpace::hsv:
      equaliseHSV_();
      break;
   }
}

void Image::gaussianFilter(cv::Size size, double sigma)
{
   cv::Mat result;
   cv::GaussianBlur(
      mat_,
      result,
      size,
      sigma);
   mat_ = result;
}

Image::ColourSpace Image::getColourSpace() const
{
   return colSpace_;
}

const cv::Mat& Image::image() const
{
   return mat_;
}

const cv::Mat& Image::labels() const
{
   return labels_;
}

void Image::log()
{
   mat_ += 1;
   cv::log(mat_, mat_);
}

cv::Scalar prj::Image::mean() const
{
   return cv::mean(mat_);
}

void Image::negative()
{
   switch (mat_.depth())
   {
   case CV_8U:
      mat_ = std::numeric_limits<std::uint8_t>::max() - mat_;
      break;
   case CV_8S:
      mat_ = std::numeric_limits<std::int8_t>::max() - mat_;
      break;
   case CV_16U:
      mat_ = std::numeric_limits<std::uint16_t>::max() - mat_;
      break;
   case CV_16S:
      mat_ = std::numeric_limits<std::int16_t>::max() - mat_;
      break;
   case CV_32S:
      mat_ = std::numeric_limits<std::int32_t>::max() - mat_;
      break;
   case CV_32F:
      mat_ = std::numeric_limits<float>::max() - mat_;
      break;
   case CV_64F:
      mat_ = std::numeric_limits<double>::max() - mat_;
      break;
   }
}

void Image::normalise(double lowerLimit, double upperLimit, int normType)
{
   cv::normalize(mat_, mat_, lowerLimit, upperLimit, normType);
}

void Image::resize(const cv::Size& newSize)
{
   cv::resize(mat_, mat_, newSize);
}

void Image::segment(double cannyTh1, double cannyTh2, double distTh)
{
   Image edgeMap{ mat_ };
   edgeMap.canny(cannyTh1, cannyTh2);
   if constexpr (debug) edgeMap.display();
   edgeMap.negative();
   edgeMap.distanceTransform();
   edgeMap.log();
   edgeMap.normalise(0.0, 1.0, cv::NORM_MINMAX);
   if constexpr (debug) edgeMap.display();
   edgeMap.threshold(distTh, 1.0, cv::THRESH_BINARY);
   if constexpr (debug) edgeMap.display();
   edgeMap.connectedComponents();

   labels_ = edgeMap.labels();
   setColourSpace(ColourSpace::bgr);
   cv::watershed(mat_, labels_);
}

void Image::setColourSpace(ColourSpace newColSpace)
{
   if (newColSpace == colSpace_) return;

   switch (colSpace_)
   {
   case ColourSpace::grey:
      if (newColSpace == ColourSpace::bgr)
      {
         cv::cvtColor(mat_, mat_, cv::COLOR_GRAY2BGR);
      }
      else if (newColSpace == ColourSpace::hsv)
      {
         cv::cvtColor(mat_, mat_, cv::COLOR_GRAY2BGR);
         cv::cvtColor(mat_, mat_, cv::COLOR_BGR2HSV);
      }
      break;
   case ColourSpace::bgr:
      if (newColSpace == ColourSpace::grey)
      {
         cv::cvtColor(mat_, mat_, cv::COLOR_BGR2GRAY);
      }
      else if (newColSpace == ColourSpace::hsv)
      {
         cv::cvtColor(mat_, mat_, cv::COLOR_BGR2HSV);
      }
      break;
   case ColourSpace::hsv:
      if (newColSpace == ColourSpace::grey)
      {
         cv::cvtColor(mat_, mat_, cv::COLOR_HSV2BGR);
         cv::cvtColor(mat_, mat_, cv::COLOR_BGR2GRAY);
      }
      else if (newColSpace == ColourSpace::bgr)
      {
         cv::cvtColor(mat_, mat_, cv::COLOR_HSV2BGR);
      }
      break;
   }

   colSpace_ = newColSpace;
}

void Image::threshold(double th, double maxVal, int type)
{
   cv::threshold(mat_, mat_, th, maxVal, type);
}

void Image::equaliseHSV_()
{
   std::vector<cv::Mat> channels;
   cv::split(mat_, channels);

   cv::equalizeHist(channels[static_cast<int>(HSV::v)], channels[static_cast<int>(HSV::v)]);

   cv::merge(channels, mat_);
}
