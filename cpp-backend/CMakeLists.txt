# =============================================================================
# AI Forge Studio - C++ Backend
# CMake Configuration for CUDA + TensorRT + NVML
# =============================================================================
cmake_minimum_required(VERSION 3.24)

# Project name and languages
project(AIForgeBackend LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# CUDA Configuration
# =============================================================================
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures for RTX 5090 (Blackwell = SM 100)
# Also include SM 89 for RTX 40xx compatibility
set(CMAKE_CUDA_ARCHITECTURES "89;100")

message(STATUS "=== AI Forge Backend Build Configuration ===")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# =============================================================================
# TensorRT Configuration (Optional for now)
# =============================================================================
set(TENSORRT_ROOT "C:/Program Files/NVIDIA/TensorRT" CACHE PATH "TensorRT installation path")

if(EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    set(TENSORRT_FOUND TRUE)
    set(TENSORRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
    set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")
    
    # Find TensorRT libraries
    find_library(NVINFER_LIB nvinfer PATHS ${TENSORRT_LIB_DIR})
    find_library(NVONNXPARSER_LIB nvonnxparser PATHS ${TENSORRT_LIB_DIR})
    
    message(STATUS "TensorRT Found: ${TENSORRT_ROOT}")
    message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIRS}")
else()
    set(TENSORRT_FOUND FALSE)
    message(WARNING "TensorRT not found at ${TENSORRT_ROOT}")
endif()

# =============================================================================
# Source Files
# =============================================================================

# Main executable - GPU Info (Step 1)
add_executable(gpu_info
    src/gpu_info.cpp
    src/cuda_helper.cu
)

target_include_directories(gpu_info PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gpu_info PRIVATE
    CUDA::cudart
    CUDA::nvml
)

# Set output directory
set_target_properties(gpu_info PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# =============================================================================
# TensorRT Inference Executable (Will be added later)
# =============================================================================
if(TENSORRT_FOUND)
    add_executable(trt_inference
        src/trt_inference.cpp
        src/cuda_helper.cu
    )
    
    target_include_directories(trt_inference PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIRS}
    )
    
    target_link_libraries(trt_inference PRIVATE
        CUDA::cudart
        CUDA::nvml
        ${NVINFER_LIB}
        ${NVONNXPARSER_LIB}
    )
    
    set_target_properties(trt_inference PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# =============================================================================
# Installation
# =============================================================================
install(TARGETS gpu_info DESTINATION bin)
if(TENSORRT_FOUND)
    install(TARGETS trt_inference DESTINATION bin)
endif()

message(STATUS "==============================================")
