#include "Image.hpp"

#include "Window.hpp"

#include <opencv2/imgproc.hpp>

using namespace prj;

const cv::Scalar Image::default_colour{ 0, 0, 255 };
const cv::Size   Image::contrast_kernel_size{ 9, 9 };

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

void Image::bilateralFilter(const BilateralFilterParams& params)
{
   cv::Mat result;
   cv::bilateralFilter(
      mat_,
      result,
      params.size,
      params.colSig,
      params.spaceSig);
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
      cv::connectedComponents(mat_, labels_, neighbours, CV_32S);
   else
   {
      cv::Mat mat8U;
      mat_.convertTo(mat8U, CV_8U);
      cv::connectedComponents(mat8U, labels_, neighbours, CV_32S);
   }
}

float Image::contrast()
{
   ColourSpace colourSpace{ colSpace_ };
   setColourSpace(ColourSpace::hsv);

   std::vector<cv::Mat> channels;
   cv::split(mat_, channels);

   cv::Mat kernel{ cv::Mat::ones(contrast_kernel_size, CV_8U) };
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

void Image::draw(Shape shape, const std::vector<cv::Point>& pts, const cv::Scalar& colour)
{
   int thickness{ static_cast<int>(std::min(mat_.rows, mat_.cols) * thickness_coeff) };
   switch (shape)
   {
   case Shape::rect:
      cv::rectangle(mat_, pts[0], pts[1], colour, thickness);
      break;
   }
}

void Image::drawText(std::string_view text, const cv::Point& pt, const cv::Scalar& colour)
{
   double size{ mat_.rows * thickness_coeff };
   int    thickness{ static_cast<int>(std::min(mat_.rows, mat_.cols) * thickness_coeff) };
   cv::putText(mat_, text.data(), pt, cv::FONT_HERSHEY_PLAIN, size, colour, thickness);
}

void Image::dilate(const cv::Mat& kernel)
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

bool prj::Image::empty() const
{
   return mat_.empty();
}

void Image::erode(const cv::Mat& kernel)
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

void Image::gaussianFilter(const GaussianFilterParams& params)
{
   cv::Mat result;
   cv::GaussianBlur(
      mat_,
      result,
      params.size,
      params.sigma);
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
   case RegionType::label:
      return computeLabelRegions_();
      break;
   case RegionType::blob:
      return computeBlobRegions_();
      break;
   default:
      return std::vector<Rect<int>>();
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

void Image::segment(const SegmentationParams& params)
{
   Image edgeMap{ mat_ };
   edgeMap.canny(params.cannyTh1, params.cannyTh2);
   if constexpr (debug) edgeMap.display();
   edgeMap.negative();
   edgeMap.distanceTransform();
   edgeMap.log();
   edgeMap.normalise(0.0, 1.0, cv::NORM_MINMAX);
   if constexpr (debug) edgeMap.display();
   edgeMap.threshold(params.distTh, 1.0, cv::THRESH_BINARY);
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

   std::vector<int> ids;
   // Cumulative region parameters used to compute average centre and size.
   std::vector<std::pair<cv::Point2l, int>> cumulatives;

   // Compute the region centres.
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
               cumulatives.emplace_back(cv::Point2l{ x, y }, 1);
            }
            // Otherwise update the centre.
            else
            {
               long long index{ std::distance(ids.begin(), currentId) };
               cumulatives[index].first += cv::Point2l{ x, y };
               ++cumulatives[index].second;
            }
         }
      }
   }

   // Compute the average centres.
   result.reserve(cumulatives.size());
   for (auto& centre : cumulatives)
   {
      centre.first /= centre.second;
      result.emplace_back(centre.first.x, centre.first.y, 1, 1);
      centre.first = cv::Point2l{ 0, 0 };
   }

   // Compute region sizes.
   for (int y = 0; y < labels_.rows; ++y)
   {
      for (int x = 0; x < labels_.cols; ++x)
      {
         if (int currentLabel{ labels_.at<int>(y, x) }; currentLabel != -1)
         {
            // Update the current region's size.
            auto      currentId = std::find(ids.begin(), ids.end(), currentLabel);
            long long index{ std::distance(ids.begin(), currentId) };
            cumulatives[index].first += cv::Point2l{
               std::abs(x - result[index].x),
               std::abs(y - result[index].y)
            };
         }
      }
   }

   // Compute the average sizes.
   for (int i = 0; i < cumulatives.size(); ++i)
   {
      result[i].w = 3 * 2 * cumulatives[i].first.x / cumulatives[i].second;
      result[i].h = 3 * 2 * cumulatives[i].first.y / cumulatives[i].second;
      result[i].x -= result[i].w / 2;
      result[i].y -= result[i].h / 2;

      if (result[i].x < 0) result[i].x = 0;
      if (result[i].y < 0) result[i].y = 0;
      int xDiff{ result[i].x + result[i].w - mat_.cols + 1 };
      int yDiff{ result[i].y + result[i].h - mat_.rows + 1 };
      if (xDiff > 0) result[i].w -= xDiff;
      if (yDiff > 0) result[i].h -= yDiff;
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
               colours.resize(currentLabel + 1LL);
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