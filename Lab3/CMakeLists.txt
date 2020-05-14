cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17) # Set the C++ standard to use.
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Set the specified C++ standard as mandatory.

project(Lab3) # Specify a project name. 

# If on Windows, set the appropriate directory for OpenCV
if (WIN32)
    set (OpenCV_DIR $ENV{OPENCV_DIR}/build)
endif ()

# Set the appropriate compiler flags.
if (MSVC)
    add_compile_options(/W4 /permissive-)
else ()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

find_package(OpenCV REQUIRED)

include_directories(
    include 
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    ${PROJECT_NAME} 
    "src/lab3_main.cpp"
    "src/Filter.cpp"
    "src/lab3.cpp"
    
    "src/HistEqualiser.cpp"
    "src/FilterComparator.cpp"
    "src/Morph.cpp"
)

target_link_libraries(
    ${PROJECT_NAME} 
    ${OpenCV_LIBS}
)

# Copy program data to the build directory.
file(COPY "data" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})