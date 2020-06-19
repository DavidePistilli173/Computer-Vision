#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <opencv2/core.hpp>
#include <string_view>
#include <vector>

namespace prj
{
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