# AI Forge Studio - C++ GPU Integration

## Project Structure

```
cpp/
├── CMakeLists.txt          # Main CMake configuration
├── build.bat               # Windows build script
├── installer.nsi           # NSIS installer script
├── src/
│   ├── main.cpp            # Qt application entry point
│   ├── gpu/
│   │   ├── cuda_inference.h/cpp    # CUDA inference wrapper
│   │   ├── cuda_kernels.cu         # CUDA kernel implementations
│   │   ├── tensorrt_engine.h/cpp   # TensorRT engine manager
│   │   └── vulkan_renderer.h/cpp   # Vulkan rendering pipeline
│   └── monitor/
│       └── gpu_monitor.h/cpp       # NVML-based GPU monitoring
├── shaders/
│   ├── shader.vert         # Vulkan vertex shader
│   ├── shader.frag         # Vulkan fragment shader (PBR)
│   └── inference.comp      # Vulkan compute shader for inference
└── resources/
    └── app.rc              # Windows resource file
```

## Prerequisites

### Required Software

1. **Visual Studio 2022** with "Desktop development with C++"
2. **CMake 3.24+**: https://cmake.org/download/
3. **CUDA Toolkit 12.4+**: https://developer.nvidia.com/cuda-downloads
4. **TensorRT 10.x**: https://developer.nvidia.com/tensorrt
5. **Vulkan SDK 1.3+**: https://vulkan.lunarg.com/sdk/home
6. **Qt 6.6+**: https://www.qt.io/download

### Environment Variables

Ensure these are set in your system:

```powershell
# Example paths - adjust to your installation
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:VULKAN_SDK = "C:\VulkanSDK\1.3.280.0"
$env:Qt6_DIR = "C:\Qt\6.6.2\msvc2019_64\lib\cmake\Qt6"
```

## Building

### Quick Build

```cmd
cd cpp
build.bat
```

### Manual CMake Build

```cmd
cd cpp
mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DTENSORRT_ROOT="C:/TensorRT-10.0"

cmake --build . --config Release --parallel
```

### Compile Shaders

```cmd
cd build
glslc ..\shaders\shader.vert -o shaders\shader.vert.spv
glslc ..\shaders\shader.frag -o shaders\shader.frag.spv
glslc ..\shaders\inference.comp -o shaders\inference.comp.spv
```

## Features

### GPU Monitoring (NVML)

Real-time monitoring of:
- GPU utilization (0-100%)
- VRAM usage
- Temperature
- Power draw
- Clock speeds
- PCIe throughput

### CUDA Inference

- Vector operations
- Matrix multiplication
- Neural network activation functions (ReLU, Sigmoid, GELU)
- Optimized for RTX 5090 (SM 100 - Blackwell architecture)

### TensorRT Integration

- ONNX model loading
- FP16/INT8 quantization
- Dynamic batch inference
- Engine serialization/caching

### Vulkan Rendering

- PBR (Physically Based Rendering) pipeline
- Compute shaders for on-GPU inference
- Multi-light support
- HDR tonemapping

## Qt UI Features

- **GPU Performance Panel**: Live charts showing GPU metrics
- **Inference Controls**: CUDA benchmark, TensorRT loading, Vulkan init
- **Dark Theme**: Matches AI Forge Studio web design
- **High DPI Support**: Works on 4K displays

## Creating the Installer

1. Install NSIS: https://nsis.sourceforge.io/Download
2. Build the project in Release mode
3. Copy required redistributables to `cpp/redist/`
4. Run: `makensis installer.nsi`

## Troubleshooting

### CMake can't find CUDA
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set PATH=%CUDA_PATH%\bin;%PATH%
```

### Qt not found
```cmd
set Qt6_DIR=C:\Qt\6.6.2\msvc2019_64\lib\cmake\Qt6
```

### TensorRT not found
Update `TENSORRT_ROOT` in CMakeLists.txt or pass it:
```cmd
cmake .. -DTENSORRT_ROOT="C:/TensorRT-10.0"
```

### NVML not found
NVML is part of the NVIDIA driver. Ensure you have the latest driver installed.

## License

MIT License - See LICENSE file

## Author

**M.3R3** - AI Forge Studio
