#include "Window.hpp"

#include "Log.hpp"

#include <opencv2/highgui.hpp>

using namespace prj;

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