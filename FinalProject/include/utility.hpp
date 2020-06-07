#ifndef LAB5_HPP
#define LAB5_HPP

#include <cstdio>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

namespace prj
{
   /********** TYPEDEFS **********/
   using param = std::variant<int, float, double, cv::Size>; // Generic parameter of different types.

   /********** CONSTANTS **********/
   constexpr double pi{ 3.14159265358979 };

#ifdef PRJ_DEBUG
   constexpr bool debug{ true };
#else
   constexpr bool debug{ false };
#endif

#ifdef PRJ_DEBUG_MSGS
   constexpr bool debug_msgs{ true };
#else
   constexpr bool debug_msgs{ false };
#endif

   /********** ENUMS **********/
   // Keycodes.
   enum class Key
   {
      esc = 27
   };
   // HSV colour channels
   enum class HSV
   {
      h, // Hue
      s, // Saturation
      v  // Value
   };

   /********** STRUCTS **********/
   // Rectangle.
   template<typename T>
   struct Rect
   {
      static constexpr int fields{ 4 }; // Number of fields.
      constexpr Rect() = default;
      constexpr Rect(const T px, const T py, const T pw, const T ph) :
         x{ px }, y{ py }, w{ pw }, h{ ph } {}
      constexpr Rect(std::array<T, fields> data) :
         x{ data[0] }, y{ data[1] }, w{ data[2] }, h{ data[3] } {}

      // Position of the top-left vertex.
      T x{};
      T y{};
      // Size of the rectangle.
      T w{};
      T h{};
   };

   /********** CLASSES **********/
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
      // Apply the Canny edge detector.
      void canny(double th1, double th2);
      // Compute the connected components of the image.
      void connectedComponents();
      // Compute the average local constrast of the image.
      [[nodiscard]] float contrast();
      // Change the data type used for each channel.
      void convert(int type);
      // Dilate the image.
      void dilate(cv::Mat kernel);
      // Display the image.
      void display(bool useLabels = false) const;
      // Display the image in a window with custom name.
      void display(std::string_view winName, bool useLabels = false) const;
      // Compute the distance transform.
      void distanceTransform();
      // Erode the image.
      void erode(cv::Mat kernel);
      // Equalise the histogram of the image.
      void equaliseHistogram();
      // Apply a Gaussian filter to the image.
      void gaussianFilter(cv::Size size, double sigma);
      // Get the current colour space of the image.
      [[nodiscard]] ColourSpace getColourSpace() const;
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
      void equaliseHSV_();

      /********** VARIABLES **********/
      cv::Mat     mat_;                           // Image data.
      cv::Mat     labels_;                        // Labels for image pixels.
      ColourSpace colSpace_{ ColourSpace::grey }; // Current colour space of the image.
   };

   // Basic console logging functions.
   class Log
   {
   public:
      /********** CONSTANTS **********/
      static constexpr const char* asterisks =
         "********************************************************************************";
      static constexpr const char* spaces =
         "                                                                                ";
      static constexpr const char* hyphens =
         "--------------------------------------------------------------------------------";

      /********** METHODS **********/
      // Standard versions, always active.
      template<typename... Args>
      static void info(std::string_view msg, Args... args)
      {
         std::scoped_lock  lck(mtx_);
         std::stringstream s;
         s << std::this_thread::get_id();
         std::printf("\033[1;37m[INFO - T%s] - ", s.str().c_str());
         std::printf(msg.data(), args...);
         std::printf("\n\033[0m");
      }
      template<typename... Args>
      static void warn(std::string_view msg, Args... args)
      {
         std::scoped_lock  lck(mtx_);
         std::stringstream s;
         s << std::this_thread::get_id();
         std::printf("\033[0;33m[WARN - T%s] - ", s.str().c_str());
         std::printf(msg.data(), args...);
         std::printf("\n\033[0m");
      }
      template<typename... Args>
      static void error(std::string_view msg, Args... args)
      {
         std::scoped_lock  lck(mtx_);
         std::stringstream s;
         s << std::this_thread::get_id();
         std::printf("\033[0;31m[ERROR - T%s] - ", s.str().c_str());
         std::printf(msg.data(), args...);
         std::printf("\n\033[0m");
      }
      template<typename... Args>
      static void fatal(std::string_view msg, Args... args)
      {
         std::scoped_lock  lck(mtx_);
         std::stringstream s;
         s << std::this_thread::get_id();
         std::printf("\033[0;35m[FATAL - T%s] - ", s.str().c_str());
         std::printf(msg.data(), args...);
         std::printf("\n\033[0m");
      }

      // Debug versions, only active if debug_msgs is true.
      template<typename... Args>
      static void info_d(std::string_view msg, Args... args)
      {
         if constexpr (debug_msgs)
         {
            std::scoped_lock  lck(mtx_);
            std::stringstream s;
            s << std::this_thread::get_id();
            std::printf("\033[1;37m[DINFO - T%s] - ", s.str().c_str());
            std::printf(msg.data(), args...);
            std::printf("\n\033[0m");
         }
      }
      template<typename... Args>
      static void warn_d(std::string_view msg, Args... args)
      {
         if constexpr (debug_msgs)
         {
            std::scoped_lock  lck(mtx_);
            std::stringstream s;
            s << std::this_thread::get_id();
            std::printf("\033[0;33m[DWARN - T%s] - ", s.str().c_str());
            std::printf(msg.data(), args...);
            std::printf("\n\033[0m");
         }
      }
      template<typename... Args>
      static void error_d(std::string_view msg, Args... args)
      {
         if constexpr (debug_msgs)
         {
            std::scoped_lock  lck(mtx_);
            std::stringstream s;
            s << std::this_thread::get_id();
            std::printf("\033[0;31m[DERROR - T%s] - ", s.str().c_str());
            std::printf(msg.data(), args...);
            std::printf("\n\033[0m");
         }
      }
      template<typename... Args>
      static void fatal_d(std::string_view msg, Args... args)
      {
         if constexpr (debug_msgs)
         {
            std::scoped_lock  lck(mtx_);
            std::stringstream s;
            s << std::this_thread::get_id();
            std::printf("\033[0;35m[DFATAL - T%s] - ", s.str().c_str());
            std::printf(msg.data(), args...);
            std::printf("\n\033[0m");
         }
      }

   private:
      static std::mutex mtx_;
   };

   // Basic wrapper for OpenCV windows.
   class Window
   {
   public:
      /********** CONSTANTS **********/
      static constexpr size_t max_trckbar_num{ 8 }; // Maximum number of trackbars.

      /********** CONSTRUCTORS **********/
      explicit Window(std::string_view name);
      Window(const Window&) = delete;
      Window(Window&& win) = delete;
      ~Window();

      /********** OPERATORS **********/
      Window& operator=(const Window&) = delete;
      Window& operator=(Window&& win) = delete;

      /********** METHODS **********/
      // Add a trackbar with starting position 0.
      bool addTrackBar(std::string_view name, int maxVal);
      // Add a trackbar.
      bool addTrackBar(std::string_view name, int startVal, int maxVal);
      // Return the current trackbar values.
      [[nodiscard]] std::vector<int> fetchTrckVals();
      // Check whether the trackbar values changed.
      [[nodiscard]] bool modified() const;
      // Show an image on the window.
      void showImg(const cv::Mat& img) const;

   private:
      /********** METHODS **********/
      static void trckCallbck_(int val, void* ptr); // Trackbar callback.

      /********** VARIABLES **********/
      cv::String       name_;                 // Name of the window.
      std::vector<int> trckBarVals_;          // Values of the trackbars.
      bool             trckModified_{ true }; // True if trackbar values changed since the last fetch.
   };
} // namespace prj

#endif