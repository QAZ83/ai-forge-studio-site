# ==============================================================================
# AI Forge Studio - Main CMakeLists.txt
# GPU-Accelerated Desktop Application with CUDA, TensorRT, Vulkan, Qt 6
# Author: M.3R3
# ==============================================================================

cmake_minimum_required(VERSION 3.24)
project(AIForgeStudio VERSION 1.0.0 LANGUAGES CXX CUDA)

# ==============================================================================
# Build Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Windows-specific settings
if(WIN32)
    set(CMAKE_WIN32_EXECUTABLE ON)
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-DNOMINMAX)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# ==============================================================================
# TODO: Set your SDK paths here (or use environment variables)
# ==============================================================================
# CUDA Toolkit (usually auto-detected)
# set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")

# TensorRT path
set(TENSORRT_ROOT "C:/TensorRT-10.0" CACHE PATH "Path to TensorRT installation")
# TODO: Update this path to your TensorRT installation

# Vulkan SDK (usually set via VULKAN_SDK environment variable)
# set(VULKAN_SDK "C:/VulkanSDK/1.3.280.0")

# ==============================================================================
# Find Required Packages
# ==============================================================================

# CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")

# Qt 6
find_package(Qt6 REQUIRED COMPONENTS 
    Core 
    Gui 
    Widgets 
    Qml 
    Quick 
    Charts
)
qt_standard_project_setup()
message(STATUS "Qt6 Version: ${Qt6_VERSION}")

# Vulkan
find_package(Vulkan REQUIRED)
message(STATUS "Vulkan Include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Library: ${Vulkan_LIBRARIES}")

# TensorRT
if(EXISTS ${TENSORRT_ROOT})
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT}/include)
    set(TENSORRT_LIBRARY_DIRS ${TENSORRT_ROOT}/lib)
    
    find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_LIBRARY_DIRS})
    find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_LIBRARY_DIRS})
    find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_LIBRARY_DIRS})
    
    if(NVINFER_LIB)
        message(STATUS "TensorRT found: ${TENSORRT_ROOT}")
        set(TENSORRT_FOUND TRUE)
    endif()
else()
    message(WARNING "TensorRT not found at ${TENSORRT_ROOT}. TensorRT features will be disabled.")
    set(TENSORRT_FOUND FALSE)
endif()

# NVML (for GPU monitoring)
find_library(NVML_LIB nvml 
    HINTS 
        ${CUDAToolkit_LIBRARY_DIR}
        "C:/Program Files/NVIDIA Corporation/NVSMI"
        "C:/Windows/System32"
)
if(NVML_LIB)
    message(STATUS "NVML found: ${NVML_LIB}")
else()
    message(WARNING "NVML not found. GPU monitoring will use fallback.")
endif()

# ==============================================================================
# Source Files
# ==============================================================================

# GPU Module
set(GPU_SOURCES
    src/gpu/cuda_inference.cpp
    src/gpu/cuda_inference.h
    src/gpu/cuda_kernels.cu
    src/gpu/tensorrt_engine.cpp
    src/gpu/tensorrt_engine.h
    src/gpu/vulkan_renderer.cpp
    src/gpu/vulkan_renderer.h
)

# Monitor Module
set(MONITOR_SOURCES
    src/monitor/gpu_monitor.cpp
    src/monitor/gpu_monitor.h
)

# Main Application (with embedded UI)
set(APP_SOURCES
    src/main.cpp
    ${GPU_SOURCES}
    ${MONITOR_SOURCES}
)

# ==============================================================================
# Create Executable
# ==============================================================================
qt_add_executable(AIForgeStudio
    ${APP_SOURCES}
)

# ==============================================================================
# Include Directories
# ==============================================================================
target_include_directories(AIForgeStudio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/monitor
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ui
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
)

if(TENSORRT_FOUND)
    target_include_directories(AIForgeStudio PRIVATE ${TENSORRT_INCLUDE_DIRS})
    target_compile_definitions(AIForgeStudio PRIVATE HAS_TENSORRT=1)
endif()

if(NVML_LIB)
    target_compile_definitions(AIForgeStudio PRIVATE HAS_NVML=1)
endif()

# ==============================================================================
# Link Libraries
# ==============================================================================
target_link_libraries(AIForgeStudio PRIVATE
    # Qt
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Qml
    Qt6::Quick
    Qt6::Charts
    
    # CUDA
    CUDA::cudart
    CUDA::cuda_driver
    
    # Vulkan
    Vulkan::Vulkan
)

if(TENSORRT_FOUND)
    target_link_libraries(AIForgeStudio PRIVATE
        ${NVINFER_LIB}
        ${NVINFER_PLUGIN_LIB}
        ${NVONNXPARSER_LIB}
    )
endif()

if(NVML_LIB)
    target_link_libraries(AIForgeStudio PRIVATE ${NVML_LIB})
endif()

# ==============================================================================
# CUDA Compilation Settings
# ==============================================================================
set_target_properties(AIForgeStudio PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89;90"  # Ada Lovelace (RTX 40xx) and Blackwell (RTX 50xx)
)

# ==============================================================================
# Installation
# ==============================================================================
include(GNUInstallDirs)

install(TARGETS AIForgeStudio
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Deploy Qt dependencies on Windows
if(WIN32)
    # Get windeployqt path
    get_target_property(_qmake_executable Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${_qt_bin_dir}")
    
    add_custom_command(TARGET AIForgeStudio POST_BUILD
        COMMAND "${WINDEPLOYQT_EXECUTABLE}"
            --no-translations
            --no-system-d3d-compiler
            --no-opengl-sw
            "$<TARGET_FILE:AIForgeStudio>"
        COMMENT "Deploying Qt dependencies..."
    )
endif()

# ==============================================================================
# CPack Installer Configuration
# ==============================================================================
set(CPACK_PACKAGE_NAME "AI Forge Studio")
set(CPACK_PACKAGE_VENDOR "M.3R3")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU-Accelerated AI Development Environment")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "AI Forge Studio")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "AI Forge Studio")
    set(CPACK_NSIS_PACKAGE_NAME "AI Forge Studio")
    set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\AIForgeStudio.exe")
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$DESKTOP\\\\AI Forge Studio.lnk' '$INSTDIR\\\\bin\\\\AIForgeStudio.exe'"
    )
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$DESKTOP\\\\AI Forge Studio.lnk'"
    )
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    
    # Start Menu
    set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/resources/icon.ico")
    set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/resources/icon.ico")
endif()

include(CPack)
