#include "utility.hpp"

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
   mat_{ mat.clone() }
{
   if (mat_.channels() == 1)
      colSpace_ = ColourSpace::grey;
   else
      colSpace_ = ColourSpace::bgr;
}

Image::Image(const Image& img) :
   mat_{ img.mat_.clone() },
   colSpace_{ img.colSpace_ } {}

Image& Image::operator=(const Image& img)
{
   if (&img == this) return *this;

   mat_ = img.mat_.clone();
   colSpace_ = img.colSpace_;
   return *this;
}

Image& Image::operator=(const cv::Mat& mat)
{
   mat_ = mat.clone();

   if (mat_.channels() == 1)
      colSpace_ = ColourSpace::grey;
   else
      colSpace_ = ColourSpace::bgr;

   return *this;
}

void Image::display() const
{
   Window win{ "Test" };
   win.showImg(mat_);
   cv::waitKey(0);
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

void Image::filter(Filter filter, const std::vector<param>& params)
{
   cv::Mat result;
   switch (filter)
   {
   case Filter::bilateral:
      cv::bilateralFilter(
         mat_,
         result,
         std::get<int>(params[static_cast<int>(BilateralParam::size)]),
         std::get<double>(params[static_cast<int>(BilateralParam::colour_sig)]),
         std::get<double>(params[static_cast<int>(BilateralParam::space_sig)]));
      break;
   case Filter::gaussian:
      cv::GaussianBlur(
         mat_,
         result,
         std::get<cv::Size>(params[static_cast<int>(GaussianParam::size)]),
         std::get<double>(params[static_cast<int>(GaussianParam::sig)]));
      break;
   }
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

void prj::Image::resize(const cv::Size& newSize)
{
   cv::resize(mat_, mat_, newSize);
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

void Image::equaliseHSV_()
{
   std::vector<cv::Mat> channels;
   cv::split(mat_, channels);

   cv::equalizeHist(channels[static_cast<int>(HSV::v)], channels[static_cast<int>(HSV::v)]);

   cv::merge(channels, mat_);
}
