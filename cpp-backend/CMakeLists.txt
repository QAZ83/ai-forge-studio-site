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
# TensorRT 10.14 Configuration
# =============================================================================
set(TENSORRT_ROOT "C:/Users/hamad/ai-forge-studio-site/dist/TensorRT-10.14.1.48" CACHE PATH "TensorRT installation path")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")
set(TENSORRT_BIN_DIR "${TENSORRT_ROOT}/bin")

if(EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
    set(TENSORRT_FOUND TRUE)
    
    # TensorRT 10.x uses versioned library names
    find_library(NVINFER_LIB NAMES nvinfer_10 nvinfer PATHS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH)
    find_library(NVINFER_PLUGIN_LIB NAMES nvinfer_plugin_10 nvinfer_plugin PATHS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH)
    find_library(NVONNXPARSER_LIB NAMES nvonnxparser_10 nvonnxparser PATHS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH)
    
    message(STATUS "=== TensorRT Configuration ===")
    message(STATUS "TensorRT Root: ${TENSORRT_ROOT}")
    message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT Lib: ${TENSORRT_LIB_DIR}")
    message(STATUS "nvinfer: ${NVINFER_LIB}")
    message(STATUS "nvinfer_plugin: ${NVINFER_PLUGIN_LIB}")
    message(STATUS "nvonnxparser: ${NVONNXPARSER_LIB}")
else()
    set(TENSORRT_FOUND FALSE)
    message(WARNING "TensorRT not found at ${TENSORRT_ROOT}")
endif()

# =============================================================================
# Source Files
# =============================================================================

# Main executable - GPU Info (Step 1) - Human readable output
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

# GPU Info JSON - For Electron integration
add_executable(gpu_info_json
    src/gpu_info_json.cpp
)

target_include_directories(gpu_info_json PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gpu_info_json PRIVATE
    CUDA::cudart
    CUDA::nvml
)

# =============================================================================
# TensorRT Inference JSON - For Electron Inference Dashboard
# Supports: CUDA benchmark, Mock mode, and Real TensorRT inference
# =============================================================================
add_executable(trt_inference_json
    src/trt_inference_json.cpp
)

set_source_files_properties(src/trt_inference_json.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(trt_inference_json PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Add TensorRT support if available
if(TENSORRT_FOUND)
    target_compile_definitions(trt_inference_json PRIVATE HAS_TENSORRT=1)
    target_include_directories(trt_inference_json PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_directories(trt_inference_json PRIVATE ${TENSORRT_LIB_DIR})
    target_link_libraries(trt_inference_json PRIVATE
        CUDA::cudart
        CUDA::nvml
        ${NVINFER_LIB}
        ${NVINFER_PLUGIN_LIB}
        ${NVONNXPARSER_LIB}
    )
    message(STATUS "trt_inference_json: TensorRT ENABLED")
else()
    target_link_libraries(trt_inference_json PRIVATE
        CUDA::cudart
        CUDA::nvml
    )
    message(STATUS "trt_inference_json: TensorRT DISABLED (mock/cuda modes only)")
endif()

# Set output directory
set_target_properties(gpu_info gpu_info_json trt_inference_json PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# =============================================================================
# Legacy TensorRT Inference Executable (Full featured)
# =============================================================================
if(TENSORRT_FOUND)
    add_executable(trt_inference
        src/trt_inference.cpp
        src/cuda_helper.cu
    )
    
    target_include_directories(trt_inference PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
    )
    
    target_link_directories(trt_inference PRIVATE ${TENSORRT_LIB_DIR})
    
    target_link_libraries(trt_inference PRIVATE
        CUDA::cudart
        CUDA::nvml
        ${NVINFER_LIB}
        ${NVINFER_PLUGIN_LIB}
        ${NVONNXPARSER_LIB}
    )
    
    set_target_properties(trt_inference PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# =============================================================================
# Installation
# =============================================================================
install(TARGETS gpu_info gpu_info_json trt_inference_json DESTINATION bin)
if(TENSORRT_FOUND)
    install(TARGETS trt_inference DESTINATION bin)
endif()

# =============================================================================
# Runtime Note: Add TensorRT bin to PATH before running
# =============================================================================
if(TENSORRT_FOUND)
    message(STATUS "")
    message(STATUS ">>> IMPORTANT: Add TensorRT DLLs to PATH before running:")
    message(STATUS "    set PATH=${TENSORRT_BIN_DIR};%PATH%")
    message(STATUS "")
endif()

message(STATUS "==============================================")
