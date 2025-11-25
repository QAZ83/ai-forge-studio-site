# AI Forge Studio - C++/Qt Integration Guide

## 📚 Overview

هذا الدليل الشامل لبناء تطبيق C++/Qt متكامل مع CUDA, TensorRT, و Vulkan SDK لتشغيل نماذج AI مباشرة على RTX 5090.

## 🛠️ المتطلبات الأساسية

### Software Requirements
- **Qt Framework**: 6.5.0 or later
- **CUDA Toolkit**: 12.4
- **TensorRT**: 10.14.1.48
- **Vulkan SDK**: 1.4.328.1
- **CMake**: 3.18+
- **Visual Studio 2022** (Windows) or **GCC 11+** (Linux)
- **NVIDIA Driver**: 566.03 or later

### Hardware Requirements
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **RAM**: 64GB+ recommended
- **Storage**: 500GB+ SSD for models

## 📁 Project Structure

```
AIForgeStudio/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── MainWindow.h/cpp
│   ├── inference/
│   │   ├── TensorRTEngine.h/cpp
│   │   ├── CudaKernels.cu
│   │   └── ModelLoader.h/cpp
│   ├── graphics/
│   │   ├── VulkanRenderer.h/cpp
│   │   └── Shaders/
│   └── ui/
│       ├── ModelConfigWidget.h/cpp
│       └── PerformanceMonitor.h/cpp
├── models/
├── shaders/
└── resources/
```

## 🔧 CMake Configuration

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(AIForgeStudio VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Qt Configuration
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui)
find_package(CUDA 12.4 REQUIRED)
find_package(Vulkan REQUIRED)

# TensorRT
set(TensorRT_DIR "C:/TensorRT-10.14.1.48")
include_directories(${TensorRT_DIR}/include)
link_directories(${TensorRT_DIR}/lib)

# CUDA Architecture
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 5090 = SM 8.9

# Source Files
set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/inference/TensorRTEngine.cpp
    src/inference/ModelLoader.cpp
    src/graphics/VulkanRenderer.cpp
    src/ui/ModelConfigWidget.cpp
    src/ui/PerformanceMonitor.cpp
)

set(CUDA_SOURCES
    src/inference/CudaKernels.cu
)

# Executable
add_executable(AIForgeStudio ${SOURCES} ${CUDA_SOURCES})

target_link_libraries(AIForgeStudio
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    ${Vulkan_LIBRARIES}
)

target_include_directories(AIForgeStudio PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
)
```

## 💻 Core Implementation

### 1. TensorRT Engine (TensorRTEngine.h)

```cpp
#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();

    // Load model from ONNX
    bool loadModelONNX(const std::string& onnxPath, nvinfer1::DataType precision);

    // Load pre-built TensorRT engine
    bool loadEngine(const std::string& enginePath);

    // Run inference
    bool infer(const float* input, float* output, int batchSize = 1);

    // Get model info
    std::vector<int> getInputShape() const;
    std::vector<int> getOutputShape() const;

    // Performance metrics
    float getLastInferenceTime() const { return mLastInferenceTime; }

private:
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };

    Logger mLogger;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

    void* mDeviceBuffers[2];  // Input and output
    cudaStream_t mStream;
    float mLastInferenceTime;

    bool allocateBuffers();
    void freeBuffers();
};
```

### 2. TensorRT Engine Implementation (TensorRTEngine.cpp)

```cpp
#include "TensorRTEngine.h"
#include <fstream>
#include <iostream>

TensorRTEngine::TensorRTEngine()
    : mLastInferenceTime(0.0f) {

    cudaStreamCreate(&mStream);
    mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
}

TensorRTEngine::~TensorRTEngine() {
    freeBuffers();
    cudaStreamDestroy(mStream);
}

bool TensorRTEngine::loadModelONNX(const std::string& onnxPath,
                                    nvinfer1::DataType precision) {

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(mLogger));

    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, mLogger));

    // Parse ONNX file
    if (!parser->parseFromFile(onnxPath.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        return false;
    }

    // Builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());

    // Set precision mode
    if (precision == nvinfer1::DataType::kHALF) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (precision == nvinfer1::DataType::kINT8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

    // Set memory pool size (4GB workspace)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               4ULL << 30);

    // Build engine
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));

    if (!serializedEngine) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return false;
    }

    // Deserialize engine
    mEngine.reset(mRuntime->deserializeCudaEngine(
        serializedEngine->data(), serializedEngine->size()));

    mContext.reset(mEngine->createExecutionContext());

    return allocateBuffers();
}

bool TensorRTEngine::infer(const float* input, float* output, int batchSize) {
    // Copy input to device
    cudaMemcpyAsync(mDeviceBuffers[0], input,
                    getInputSize() * sizeof(float),
                    cudaMemcpyHostToDevice, mStream);

    // Execute inference
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, mStream);

    void* bindings[] = {mDeviceBuffers[0], mDeviceBuffers[1]};
    mContext->enqueueV2(bindings, mStream, nullptr);

    cudaEventRecord(end, mStream);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&mLastInferenceTime, start, end);

    // Copy output to host
    cudaMemcpyAsync(output, mDeviceBuffers[1],
                    getOutputSize() * sizeof(float),
                    cudaMemcpyDeviceToHost, mStream);

    cudaStreamSynchronize(mStream);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return true;
}

bool TensorRTEngine::allocateBuffers() {
    size_t inputSize = getInputSize() * sizeof(float);
    size_t outputSize = getOutputSize() * sizeof(float);

    cudaMalloc(&mDeviceBuffers[0], inputSize);
    cudaMalloc(&mDeviceBuffers[1], outputSize);

    return true;
}

void TensorRTEngine::freeBuffers() {
    cudaFree(mDeviceBuffers[0]);
    cudaFree(mDeviceBuffers[1]);
}
```

### 3. Qt Main Window (MainWindow.h)

```cpp
#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QTextEdit>
#include "inference/TensorRTEngine.h"
#include "graphics/VulkanRenderer.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onLoadModelClicked();
    void onRunInferenceClicked();
    void onModelSelected(int index);
    void updatePerformanceMetrics();

private:
    void setupUI();
    void setupConnections();

    // UI Components
    QComboBox* mModelSelector;
    QPushButton* mLoadModelBtn;
    QPushButton* mRunInferenceBtn;
    QTextEdit* mOutputDisplay;
    QLabel* mLatencyLabel;
    QLabel* mThroughputLabel;

    // Engine
    std::unique_ptr<TensorRTEngine> mEngine;
    std::unique_ptr<VulkanRenderer> mRenderer;

    // Performance
    QTimer* mMetricsTimer;
};
```

### 4. CUDA Kernels (CudaKernels.cu)

```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Preprocessing kernel
__global__ void preprocessKernel(const unsigned char* input,
                                  float* output,
                                  int width, int height,
                                  float mean, float std) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * 3;

    if (idx < total) {
        output[idx] = (static_cast<float>(input[idx]) - mean) / std;
    }
}

// Postprocessing kernel
__global__ void postprocessKernel(const float* input,
                                   float* output,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Apply softmax or other operations
        output[idx] = expf(input[idx]);
    }
}

// Host function to launch preprocessing
extern "C"
void launchPreprocessing(const unsigned char* d_input,
                         float* d_output,
                         int width, int height,
                         float mean, float std) {

    int total = width * height * 3;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    preprocessKernel<<<gridSize, blockSize>>>(
        d_input, d_output, width, height, mean, std);

    cudaDeviceSynchronize();
}
```

## 🎮 Qt UI Integration

### Main Window Implementation (MainWindow.cpp)

```cpp
#include "MainWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent) {

    setupUI();
    setupConnections();

    mEngine = std::make_unique<TensorRTEngine>();

    // Setup performance monitoring
    mMetricsTimer = new QTimer(this);
    connect(mMetricsTimer, &QTimer::timeout,
            this, &MainWindow::updatePerformanceMetrics);
    mMetricsTimer->start(1000);  // Update every second
}

void MainWindow::setupUI() {
    auto centralWidget = new QWidget(this);
    auto mainLayout = new QVBoxLayout(centralWidget);

    // Model selection
    auto modelLayout = new QHBoxLayout();
    mModelSelector = new QComboBox();
    mModelSelector->addItem("ResNet-50");
    mModelSelector->addItem("YOLO v8");
    mModelSelector->addItem("Llama 3.1 70B");

    mLoadModelBtn = new QPushButton("Load Model");
    modelLayout->addWidget(new QLabel("Model:"));
    modelLayout->addWidget(mModelSelector);
    modelLayout->addWidget(mLoadModelBtn);

    // Inference controls
    mRunInferenceBtn = new QPushButton("Run Inference");
    mRunInferenceBtn->setEnabled(false);

    // Performance metrics
    auto metricsLayout = new QHBoxLayout();
    mLatencyLabel = new QLabel("Latency: -- ms");
    mThroughputLabel = new QLabel("Throughput: -- FPS");
    metricsLayout->addWidget(mLatencyLabel);
    metricsLayout->addWidget(mThroughputLabel);

    // Output display
    mOutputDisplay = new QTextEdit();
    mOutputDisplay->setReadOnly(true);

    // Add to main layout
    mainLayout->addLayout(modelLayout);
    mainLayout->addWidget(mRunInferenceBtn);
    mainLayout->addLayout(metricsLayout);
    mainLayout->addWidget(mOutputDisplay);

    setCentralWidget(centralWidget);
    setWindowTitle("AI Forge Studio - CUDA/TensorRT Inference");
    resize(800, 600);
}

void MainWindow::onLoadModelClicked() {
    QString fileName = QFileDialog::getOpenFileName(
        this, "Open Model", "",
        "Model Files (*.onnx *.trt);;All Files (*)");

    if (!fileName.isEmpty()) {
        mOutputDisplay->append("Loading model: " + fileName);

        bool success = mEngine->loadModelONNX(
            fileName.toStdString(),
            nvinfer1::DataType::kHALF);

        if (success) {
            mOutputDisplay->append("✓ Model loaded successfully!");
            mRunInferenceBtn->setEnabled(true);
        } else {
            QMessageBox::critical(this, "Error", "Failed to load model");
        }
    }
}

void MainWindow::onRunInferenceClicked() {
    mOutputDisplay->append("Running inference...");

    // Prepare dummy input (replace with real data)
    std::vector<float> input(224 * 224 * 3, 0.5f);
    std::vector<float> output(1000);

    bool success = mEngine->infer(input.data(), output.data());

    if (success) {
        float latency = mEngine->getLastInferenceTime();
        mLatencyLabel->setText(QString("Latency: %1 ms").arg(latency, 0, 'f', 2));
        mThroughputLabel->setText(QString("Throughput: %1 FPS")
            .arg(1000.0f / latency, 0, 'f', 0));

        mOutputDisplay->append("✓ Inference completed!");
        mOutputDisplay->append(QString("Latency: %1 ms").arg(latency, 0, 'f', 2));
    }
}
```

## 🚀 Building the Project

### Windows (Visual Studio)

```bash
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

### Linux

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16
```

## 📊 Performance Optimization Tips

1. **Use FP16 Precision**: 2-3x faster on Tensor Cores
2. **Batch Processing**: Maximize GPU utilization
3. **CUDA Streams**: Overlap H2D, compute, and D2H
4. **Memory Pinning**: Use `cudaMallocHost()` for faster transfers
5. **Engine Caching**: Save built TensorRT engines
6. **Dynamic Shapes**: Use optimization profiles for variable batch sizes

## 🔗 Additional Resources

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Qt Documentation](https://doc.qt.io/)

---

**Built with ❤️ by M.3R3 | AI Forge Studio**
