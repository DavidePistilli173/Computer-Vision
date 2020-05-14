cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17) # Set the C++ standard to use.
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Set the specified C++ standard as mandatory.

project(Lab4) # Specify a project name. 

# If on Windows, set the appropriate directory for OpenCV
if (WIN32)
    set (OpenCV_DIR $ENV{OPENCV_DIR}/build)
endif ()

# Set the appropriate compiler flags.
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif (MSVC)

find_package(OpenCV REQUIRED)

include_directories(
    include 
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    ${PROJECT_NAME} 
    "src/lab4_main.cpp"
    "src/lab4.cpp"
    "src/StreetAnaliser.cpp"
    "src/Canny.cpp" 
    "src/ImgProcessor.cpp" 
    "src/LineHough.cpp" 
    "src/CircleHough.cpp"
)

target_link_libraries(
    ${PROJECT_NAME} 
    ${OpenCV_LIBS}
)

# Copy program data to the build directory.
if (WIN32 AND MSVC)
    file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}" NATIVE_BIN_DIR)
    file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data" NATIVE_DATA_DIR)
    add_custom_command(
        TARGET "${PROJECT_NAME}" POST_BUILD
        COMMAND xcopy "${NATIVE_DATA_DIR}" "${NATIVE_BIN_DIR}\\data" /I /C /K /E /R /Y
    )
endif (WIN32 AND MSVC)