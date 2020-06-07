cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17) # Set the C++ standard to use.
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Set the specified C++ standard as mandatory.

project(Prj) # Specify a project name. 

# If on Windows, set the appropriate directory for OpenCV
if (WIN32)
    set (OpenCV_DIR $ENV{OPENCV_DIR}/build)
endif ()

# Set the appropriate compiler flags.
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DPRJ_DEBUG -DPRJ_DEBUG_MSGS /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DPRJ_DEBUG -DPRJ_DEBUG_MSGS -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif (MSVC)

if (UNIX)
    find_package(Threads)
endif (UNIX)
find_package(OpenCV REQUIRED)

include_directories(
    include 
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    ${PROJECT_NAME} 
    "src/prj_main.cpp"
    "src/utility.cpp"
    "src/TreeDetector.cpp"
    "src/TreeDetectorTrainer.cpp")

# If on Unix, link to Pthread as well.
if (WIN32)
    target_link_libraries(
        ${PROJECT_NAME} 
        ${OpenCV_LIBS}
    )
elseif (UNIX)
    target_link_libraries(
        ${PROJECT_NAME} 
        ${OpenCV_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
    )
endif (WIN32)