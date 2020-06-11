#ifndef PRJ_UTILITY_HPP
#define PRJ_UTILITY_HPP

#include <cstdio>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string_view>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

namespace prj
{
   /********** TYPEDEFS **********/
   using param = std::variant<int, float, double, cv::Size>; // Generic parameter of different types.

   /********** CONSTANTS **********/
   // Tag for tree words.
   constexpr std::string_view xml_words{ "Words" };
   // Tag for tree histogram.
   constexpr std::string_view xml_hist{ "histogram" };
   // Number of features for each tree.
   constexpr int num_features{ 1000 };
   // Number of words in a vocabulary.
   constexpr int num_words{ 128 };
   // Image dimensions.
   constexpr int img_width{ 1000 };
   constexpr int img_height{ 1000 };

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
      /********** CONSTRUCTORS **********/
      static constexpr int fields{ 4 }; // Number of fields.
      constexpr Rect() = default;
      constexpr Rect(const T px, const T py, const T pw, const T ph) :
         x{ px }, y{ py }, w{ pw }, h{ ph } {}
      constexpr explicit Rect(std::array<T, fields> data) :
         x{ data[0] }, y{ data[1] }, w{ data[2] }, h{ data[3] } {}

      /********** METHODS **********/
      // Check whether a point is inside the rectangle or not.
      [[nodiscard]] constexpr bool isInside(T ptX, T ptY) const
      {
         return (ptX >= x && ptX <= x + w && ptY >= y && ptY <= y + h);
      }

      // Position of the top-left vertex.
      T x{};
      T y{};
      // Size of the rectangle.
      T w{};
      T h{};
   };

   /********** FUNCTIONS **********/
   // Get an image portion described by a scaled rectangle.
   cv::Mat getTree(const cv::Mat& mat, Rect<int> tree, std::pair<float, float> scale);

   // Constexpr power.
   template<typename T>
   constexpr T pow(T base, T exp)
   {
      static_assert(std::is_integral<T>::value);
      if (exp == 0) return 1;
      if (exp == 1) return base;
      return base * pow(base, exp - 1);
   }

   template<typename T>
   constexpr T sqrt_helper(T x, T lo, T hi)
   {
      if (lo == hi)
         return lo;

      const T mid = (lo + hi + 1) / 2;

      if (x / mid < mid)
         return sqrt_helper<T>(x, lo, mid - 1);
      else
         return sqrt_helper(x, mid, hi);
   }

   template<typename T>
   constexpr T ct_sqrt(T x)
   {
      return sqrt_helper<T>(x, 0, x / 2 + 1);
   }

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
      static constexpr float thickness_coeff{ 0.0025F };
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
      // Dilate the image.
      void dilate(cv::Mat kernel);
      // Display the image.
      void display(RegionType type = RegionType::none) const;
      // Display the image in a window with custom name.
      void display(std::string_view winName, RegionType type = RegionType::none) const;
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

   // Quad-tree used to segment an image.
   template<int Children, int Depth>
   class ImagePyramid
   {
   public:
      // Single grid cell.
      struct Cell
      {
         Rect<int> rect;               // Position and size of the cell.
         bool      tree{ false };      // True if the BOW analysis detects a tree.
         bool      confirmed{ false }; // True if a further analysis confirms a tree.

         // Children of the cell.
         std::array<std::array<std::unique_ptr<Cell>, ct_sqrt(Children)>, ct_sqrt(Children)> children{ nullptr };
      };

      /********** CONSTANTS **********/
      static constexpr int side_elems{ ct_sqrt(Children) };

      /********** CONSTRUCTOR **********/
      ImagePyramid(int imgWidth, int imgHeight)
      {
         static_assert(Depth > 0);

         root_.rect = Rect<int>(0, 0, imgWidth - 1, imgHeight - 1);
         buildTree_(&root_, 1);
      }

      /********** METHODS **********/
      template<typename Callable>
      void visit(Callable func)
      {
         visitTree_(func, &root_, 1);
      }

      Cell* root()
      {
         return &root_;
      }

   private:
      /********** METHODS **********/
      void buildTree_(Cell* root, int lvl)
      {
         if (lvl == Depth) return;

         int cellWidth{ root->rect.w / side_elems };
         int cellHeight{ root->rect.h / side_elems };
         for (int row = 0; row < side_elems; ++row)
         {
            for (int col = 0; col < side_elems; ++col)
            {
               root->children[row][col] = std::make_unique<Cell>();
               auto rect = Rect<int>(
                  (col * root->rect.w) / side_elems,
                  (row * root->rect.h) / side_elems,
                  cellWidth,
                  cellHeight);

               int maxX{ rect.x + rect.w };
               int maxY{ rect.y + rect.h };
               if (maxX >= root->rect.w) rect.w -= root->rect.w - maxX + 1;
               if (maxY >= root->rect.h) rect.h -= root->rect.h - maxY + 1;
               root->children[row][col]->rect = rect;

               buildTree_(root->children[row][col].get(), lvl + 1);
            }
         }
      }

      template<typename Callable>
      void visitTree_(Callable func, Cell* node, int lvl)
      {
         if (node->children[0][0] != nullptr)
         {
            for (auto& row : node->children)
            {
               for (auto& child : row)
               {
                  visitTree_(func, child.get(), lvl + 1);
               }
            }
         }
         func(node);
      }

      /********** VARIABLES **********/
      // Root of the tree.
      Cell root_;
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