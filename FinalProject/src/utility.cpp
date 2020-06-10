#include "utility.hpp"

#include <cstdint>
#include <limits>
#include <opencv2/imgproc.hpp>

using namespace prj;

const cv::Scalar Image::default_colour{ 0, 0, 255 };

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
   blobs_{ img.blobs_ },
   colSpace_{ img.colSpace_ } {}

Image& Image::operator=(const Image& img)
{
   if (&img == this) return *this;

   mat_ = img.mat_.clone();
   labels_ = img.labels_.clone();
   blobs_ = img.blobs_;
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

void Image::blobDetection(const cv::SimpleBlobDetector::Params& params)
{
   auto detector = cv::SimpleBlobDetector::create(params);
   detector->detect(mat_, blobs_);
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

void Image::draw(Shape shape, const std::vector<cv::Point>& pts, cv::Scalar colour)
{
   int thickness{ static_cast<int>(std::min(mat_.rows, mat_.cols) * thickness_coeff) };
   switch (shape)
   {
   case Shape::rect:
      cv::rectangle(mat_, pts[0], pts[1], colour, thickness);
      break;
   }
}

void Image::dilate(cv::Mat kernel)
{
   cv::dilate(mat_, mat_, kernel);
}

void Image::display(RegionType type) const
{
   // The thread ID in the name is required to ensure thread-safe windows.
   std::stringstream s;
   s << "Thread " << std::this_thread::get_id();
   display(s.str(), type);
}

void Image::display(std::string_view winName, RegionType type) const
{
   Window win{ winName };

   switch (type)
   {
   case RegionType::none:
      win.showImg(mat_);
      break;
   case RegionType::label:
      win.showImg(drawLabels_());
      break;
   case RegionType::blob:
      win.showImg(drawBlobs_());
      break;
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

std::vector<Rect<int>> Image::getRegions(RegionType type) const
{
   switch (type)
   {
   case RegionType::none:
      return std::vector<Rect<int>>();
      break;
   case RegionType::label:
      return computeLabelRegions_();
      break;
   case RegionType::blob:
      return computeBlobRegions_();
      break;
   }
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

std::vector<Rect<int>> prj::Image::computeBlobRegions_() const
{
   std::vector<Rect<int>> result;
   result.reserve(blobs_.size());

   for (const auto& blob : blobs_)
   {
      float     radius{ blob.size / 2.0F };
      Rect<int> rect;
      rect.x = cvRound(blob.pt.x - radius);
      rect.y = cvRound(blob.pt.y - radius);
      rect.w = cvRound(rect.x + blob.size);
      rect.h = cvRound(rect.y + blob.size);
      result.emplace_back(rect);
   }

   return result;
}

std::vector<Rect<int>> prj::Image::computeLabelRegions_() const
{
   std::vector<Rect<int>> result;
   std::vector<int>       ids;

   // Scan the entire label image.
   for (int y = 0; y < labels_.rows; ++y)
   {
      for (int x = 0; x < labels_.cols; ++x)
      {
         if (int currentLabel{ labels_.at<int>(y, x) }; currentLabel != -1)
         {
            auto currentId = std::find(ids.begin(), ids.end(), currentLabel);
            // If the current label is new, add it.
            if (currentId == ids.end())
            {
               ids.emplace_back(currentLabel);
               result.emplace_back(x, y, 1, 1);
            }
            // Otherwise update the rectangle.
            else
            {
               long long index{ std::distance(ids.begin(), currentId) };
               if (!result[index].isInside(x, y))
               {
                  // Update x and w.
                  if (x < result[index].x)
                  {
                     result[index].w += (result[index].x - x);
                     result[index].x = x;
                  }
                  else if (int maxX{ result[index].x + result[index].w }; x > maxX)
                  {
                     result[index].w += (x - maxX);
                  }

                  // Update y and h.
                  if (y < result[index].y)
                  {
                     result[index].h += (result[index].y - y);
                     result[index].y = y;
                  }
                  else if (int maxY{ result[index].y + result[index].h }; y > maxY)
                  {
                     result[index].h += (y - maxY);
                  }
               }
            }
         }
      }
   }

   return result;
}

cv::Mat Image::drawBlobs_() const
{
   cv::Mat result{ mat_.clone() };

   cv::drawKeypoints(mat_, blobs_, result, default_colour, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

   return result;
}

cv::Mat Image::drawLabels_() const
{
   cv::Mat                result = cv::Mat::zeros(cv::Size{ mat_.cols, mat_.rows }, CV_8UC3);
   std::vector<cv::Vec3b> colours;
   for (int i = 0; i < mat_.rows; ++i)
   {
      for (int j = 0; j < mat_.cols; ++j)
      {
         if (int currentLabel{ labels_.at<int>(i, j) }; currentLabel != -1)
         {
            if (size_t prevSize{ colours.size() }; colours.size() <= currentLabel)
            {
               colours.resize(currentLabel + 1);
               for (; prevSize < colours.size(); ++prevSize)
               {
                  colours[prevSize] = {
                     static_cast<unsigned char>(cv::theRNG().uniform(0, std::numeric_limits<unsigned char>::max())),
                     static_cast<unsigned char>(cv::theRNG().uniform(0, std::numeric_limits<unsigned char>::max())),
                     static_cast<unsigned char>(cv::theRNG().uniform(0, std::numeric_limits<unsigned char>::max()))
                  };
               }
            }
            result.at<cv::Vec3b>(i, j) = colours[currentLabel];
         }
      }
   }
   return result;
}

void Image::equaliseHSV_()
{
   std::vector<cv::Mat> channels;
   cv::split(mat_, channels);

   cv::equalizeHist(channels[static_cast<int>(HSV::v)], channels[static_cast<int>(HSV::v)]);

   cv::merge(channels, mat_);
}

cv::Mat prj::getTree(const cv::Mat& mat, Rect<int> tree, std::pair<float, float> scale)
{
   return mat(
      cv::Range{
         cvRound(tree.y / scale.second),
         cvRound((tree.y + tree.h) / scale.second) },
      cv::Range{
         cvRound(tree.x / scale.first),
         cvRound((tree.x + tree.w) / scale.first) });
}
