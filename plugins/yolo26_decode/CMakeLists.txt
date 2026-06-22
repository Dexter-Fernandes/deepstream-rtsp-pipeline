cmake_minimum_required(VERSION 3.18)
project(yolo26_decode LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# TensorRT headers ship in /usr/include/x86_64-linux-gnu in the NGC DS 9.0 container.
# Override with -DTRT_INCLUDE_DIR=<path> if your layout differs.
set(TRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu"
    CACHE PATH "Directory containing NvInfer.h")

include_directories(${TRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})

add_library(yolo26_decode SHARED
    yolo26_decode_plugin.cpp
    yolo26_decode_kernel.cu
)

# SM 75 = GTX 1660 Ti (Turing). Override with -DCMAKE_CUDA_ARCHITECTURES=<arch>.
set_property(TARGET yolo26_decode PROPERTY CUDA_ARCHITECTURES 75)

target_link_libraries(yolo26_decode
    PRIVATE
    nvinfer
    CUDA::cudart
)

install(TARGETS yolo26_decode
    LIBRARY DESTINATION /opt/ds_plugins
)

# CLI helper: builds a TRT decode engine without needing the Python TRT package.
add_executable(build_yolo26_engine build_engine.cpp)
target_link_libraries(build_yolo26_engine PRIVATE nvinfer nvonnxparser CUDA::cudart dl)
install(TARGETS build_yolo26_engine RUNTIME DESTINATION /opt/ds_plugins)
