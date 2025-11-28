# AI Forge Studio - C++ Backend

## المتطلبات

- **Windows 11**
- **Visual Studio 2022** مع "Desktop development with C++"
- **CUDA Toolkit 12.x+** (موجود في `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\`)
- **TensorRT 8.x+** (موجود في `C:\Program Files\NVIDIA\TensorRT\`)
- **CMake 3.24+**

## البناء

### الطريقة 1: Developer Command Prompt

```batch
# افتح "Developer Command Prompt for VS 2022"
cd cpp-backend
build.bat
```

### الطريقة 2: CMake مباشرة

```powershell
cd cpp-backend
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## التشغيل

### gpu_info - معلومات الـ GPU
```batch
.\build\bin\Release\gpu_info.exe
```

يعرض:
- معلومات CUDA Device
- GPU metrics حقيقية (NVML)
- اختبار CUDA kernel

### trt_inference - TensorRT Inference
```batch
# عرض المساعدة
.\build\bin\Release\trt_inference.exe --help

# تحميل engine جاهز
.\build\bin\Release\trt_inference.exe --engine model.trt --benchmark

# بناء من ONNX
.\build\bin\Release\trt_inference.exe --onnx model.onnx --save model.trt --benchmark
```

## هيكل المشروع

```
cpp-backend/
├── CMakeLists.txt          # CMake configuration
├── build.bat               # Windows build script
├── README.md               # هذا الملف
├── include/
│   ├── gpu_types.h         # Type definitions
│   └── cuda_helper.h       # CUDA helper functions
└── src/
    ├── gpu_info.cpp        # GPU info tool
    ├── cuda_helper.cu      # CUDA implementations
    └── trt_inference.cpp   # TensorRT inference
```

## الخطوات القادمة

1. ✅ GPU Info tool
2. ✅ TensorRT Inference tool
3. ⏳ REST API server (للربط مع Electron)
4. ⏳ Real-time GPU monitoring daemon
