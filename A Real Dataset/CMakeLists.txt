cmake_minimum_required(VERSION 3.1)
project(ARealDataset)

set(CMAKE_WIN32_EXECUTABLE OFF)

# Set OpenCV path
set(OpenCV_DIR "C:/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/lib")

# Include directories
include_directories(
    ${OpenCV_DIR}/include
    "C:/Program Files/WindowsApps/PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0/Include"
    "C:/users/lenovo/appdata/local/packages/pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0/localcache/local-packages/python39/site-packages/numpy/_core/include"
    "F:/Codes/Neural Networks From Scratch/Github Repo/cereal-1.3.2/include"
)

link_directories(
    "C:/Program Files/WindowsApps/PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0/libs"
)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Default source file
set(SOURCE_FILE "main.cpp")

# Allow overriding the source file with a command line argument
if(DEFINED SOURCE_FILE_ARG)
    set(SOURCE_FILE ${SOURCE_FILE_ARG})
endif()

# Add executable
add_executable(MyProject ${SOURCE_FILE})

# Find OpenCV
find_package(OpenCV REQUIRED)

# Link OpenCV and Python libraries
target_link_libraries(MyProject PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} Python39)  # Adjust Python version as necessary

# Set additional compiler options
set_target_properties(MyProject PROPERTIES
    COMPILE_FLAGS "-Ofast -march=native -ftree-vectorize -flto -fopenmp -funroll-loops -fomit-frame-pointer"
)
