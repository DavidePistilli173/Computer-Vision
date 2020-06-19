#ifndef LOG_HPP
#define LOG_HPP

#include <mutex>
#include <string_view>
#include <thread>

namespace prj
{
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
} // namespace prj

#endif