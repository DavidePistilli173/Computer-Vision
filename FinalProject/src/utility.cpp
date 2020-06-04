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

void prj::displayImage(const cv::Mat& img)
{
   Window win{ "Test" };
   win.showImg(img);
   cv::waitKey(0);
}

bool prj::equaliseImg(cv::Mat& img)
{
   std::vector<cv::Mat> channels;
   cv::split(img, channels);

   cv::equalizeHist(channels[static_cast<int>(HSV::v)], channels[static_cast<int>(HSV::v)]);

   cv::merge(channels, img);
   return true;
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

Image::ColourSpace Image::getColourSpace() const
{
   return colSpace_;
}

cv::Mat& Image::getImage()
{
   return mat_;
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
