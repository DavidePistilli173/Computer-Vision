cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17) # Set the C++ standard to use.
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Set the specified C++ standard as mandatory.

project(HW1) # Specify a project name. 

# If on Windows, set the appropriate directory for OpenCV
if (WIN32)
    set (OpenCV_DIR $ENV{OPENCV_DIR}/build)
endif ()

# Set the appropriate compiler flags.
if (MSVC)
    add_compile_options(/W4)
else ()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# Find Pthread if on Unix.
if (UNIX)
    find_package(Threads)
endif (UNIX)
find_package(OpenCV REQUIRED)

include_directories(include 
                    ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} 
               src/hw1_main.cpp 
               src/hw1_image_proc.cpp)

# If on Unix, link to Pthread as well.
if (WIN32)
    target_link_libraries(${PROJECT_NAME} 
                          ${OpenCV_LIBS})
elseif (UNIX)
    target_link_libraries(${PROJECT_NAME} 
                          ${OpenCV_LIBS} 
                          ${CMAKE_THREAD_LIBS_INIT})
endif (UNIX)

# Copy program data to the build directory.
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
