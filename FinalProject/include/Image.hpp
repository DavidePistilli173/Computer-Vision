#ifndef IMAGE_HPP
#define IMAGE_HPP

#include "Rect.hpp"
#include "utility.hpp"

#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

namespace prj
{
   class Image
   {
   public:
      /********** ENUMS **********/
      // Supported colour spaces of the image.
      enum class ColourSpace
      {
         grey, // Greyscale.
         bgr,
         hsv
      };
      // Drawable shapes.
      enum class Shape
      {
         rect
      };
      // Region types.
      enum class RegionType
      {
         none,
         blob,
         label
      };

      /********** CONSTANTS **********/
      // Thickness of a line wrt the smallest image dimension.
      static constexpr float thickness_coeff{ 0.003F };
      // Default drawing colour.
      static const cv::Scalar default_colour;

      /********** CONSTRUCTORS **********/
      Image() = default;
      explicit Image(const cv::Mat& mat);
      explicit Image(cv::Mat&& mat);
      Image(const Image& img);
      Image(Image&&) = default;
      ~Image() = default;

      /********** OPERATORS **********/
      Image& operator=(const Image& img);
      Image& operator=(Image&&) = default;
      Image& operator=(const cv::Mat& mat);

      /********** METHODS **********/
      // Apply a bilateral filter to the image.
      void bilateralFilter(int size, double colour_sig, double space_sig);
      // Perform blob detection on the image.
      void blobDetection(const cv::SimpleBlobDetector::Params& params);
      // Apply the Canny edge detector.
      void canny(double th1, double th2);
      // Compute the connected components of the image.
      void connectedComponents();
      // Compute the average local constrast of the image.
      [[nodiscard]] float contrast();
      // Change the data type used for each channel.
      void convert(int type);
      // Draw a shape onto the image.
      void draw(Shape shape, const std::vector<cv::Point>& pts, cv::Scalar colour);
      // Draw text on the image.
      void drawText(std::string_view text, cv::Point pt, cv::Scalar colour);
      // Dilate the image.
      void dilate(cv::Mat kernel);
      // Display the image.
      void display(RegionType type = RegionType::none) const;
      // Display the image in a window with custom name.
      void display(std::string_view winName, RegionType type = RegionType::none) const;
      // Compute the distance transform.
      void distanceTransform();
      // Return true if there is no image stored.
      [[nodiscard]] bool empty() const;
      // Erode the image.
      void erode(cv::Mat kernel);
      // Equalise the histogram of the image.
      void equaliseHistogram();
      // Apply a Gaussian filter to the image.
      void gaussianFilter(cv::Size size, double sigma);
      // Get the current colour space of the image.
      [[nodiscard]] ColourSpace getColourSpace() const;
      // Get a rectangle for each segment of the image.
      [[nodiscard]] std::vector<Rect<int>> getRegions(RegionType type = RegionType::label) const;
      // Get the stored image.
      [[nodiscard]] const cv::Mat& image() const;
      // Get the image labels.
      [[nodiscard]] const cv::Mat& labels() const;
      // Apply the log transform to the image.
      void log();
      // Compute the mean value for each channel of the image.
      cv::Scalar mean() const;
      // Compute the negative image.
      void negative();
      // Normalise the image values.
      void normalise(double lowerLimit, double upperLimit, int normType);
      // Resize the image.
      void resize(const cv::Size& newSize);
      // Segment the image.
      void segment(double cannyTh1, double cannyTh2, double distTh);
      // Set the colour space of the image.
      void setColourSpace(ColourSpace newColSpace);
      // Apply a threshold the image values.
      void threshold(double th, double maxVal, int type);

   private:
      /********** METHODS **********/
      [[nodiscard]] std::vector<Rect<int>> computeBlobRegions_() const;
      [[nodiscard]] std::vector<Rect<int>> computeLabelRegions_() const;
      [[nodiscard]] cv::Mat                drawBlobs_() const;
      [[nodiscard]] cv::Mat                drawLabels_() const;
      void                                 equaliseHSV_();

      /********** VARIABLES **********/
      cv::Mat                   mat_;                           // Image data.
      cv::Mat                   labels_;                        // Labels for image pixels.
      std::vector<cv::KeyPoint> blobs_;                         // List of blobs in the image.
      ColourSpace               colSpace_{ ColourSpace::grey }; // Current colour space of the image.
   };
} // namespace prj

#endif