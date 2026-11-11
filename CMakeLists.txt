cmake_minimum_required(VERSION 3.15)
# Enable C++ and Objective-C++
project(m2_vision_pipeline CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Automatic Reference Counting (ARC) for Apple APIs
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")

# Point CMake to Apple Silicon's Homebrew directory
set(CMAKE_PREFIX_PATH "/opt/homebrew" ${CMAKE_PREFIX_PATH})

# Link OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Link ONNX Runtime
include_directories("/opt/homebrew/include/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS "/opt/homebrew/lib" REQUIRED)

# Bring in llama.cpp with Metal GPU enabled
set(GGML_METAL ON CACHE BOOL "Enable Metal GPU" FORCE)
add_subdirectory(../llama.cpp llama_build)

# Build the executable using the new .mm file
add_executable(vision_app main.mm)

# Link OpenCV, ONNX Runtime, llama.cpp, and macOS Hardware Frameworks
target_link_libraries(vision_app 
    ${OpenCV_LIBS} 
    ${ONNXRUNTIME_LIB} 
    llama
    "-framework Foundation"
    "-framework CoreWLAN"
)
