/**
 * AI Forge Studio - CUDA Inference Implementation
 * Author: M.3R3
 */

#include "cuda_inference.h"
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <random>

namespace aiforge {
namespace gpu {

CudaInference::CudaInference()
    : m_initialized(false)
    , m_deviceId(-1)
    , m_stream(nullptr)
{
}

CudaInference::~CudaInference() {
    shutdown();
}

bool CudaInference::checkCudaError(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        m_lastError = std::string(operation) + ": " + cudaGetErrorString(error);
        std::cerr << "[CUDA Error] " << m_lastError << std::endl;
        return false;
    }
    return true;
}

void CudaInference::setError(const std::string& error) {
    m_lastError = error;
    std::cerr << "[CUDA Error] " << error << std::endl;
}

bool CudaInference::initialize(int deviceId) {
    if (m_initialized) {
        shutdown();
    }

    // Check device count
    int deviceCount = 0;
    if (!checkCudaError(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")) {
        return false;
    }

    if (deviceCount == 0) {
        setError("No CUDA-capable devices found");
        return false;
    }

    if (deviceId >= deviceCount) {
        setError("Invalid device ID: " + std::to_string(deviceId));
        return false;
    }

    // Set device
    if (!checkCudaError(cudaSetDevice(deviceId), "cudaSetDevice")) {
        return false;
    }
    m_deviceId = deviceId;

    // Create stream
    if (!checkCudaError(cudaStreamCreate(&m_stream), "cudaStreamCreate")) {
        return false;
    }

    // Get device properties
    cudaDeviceProp prop;
    if (!checkCudaError(cudaGetDeviceProperties(&prop, deviceId), "cudaGetDeviceProperties")) {
        cudaStreamDestroy(m_stream);
        return false;
    }

    std::cout << "[CUDA] Initialized on device " << deviceId << ": " << prop.name << std::endl;
    std::cout << "[CUDA] Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "[CUDA] Total Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
    std::cout << "[CUDA] Multiprocessors: " << prop.multiProcessorCount << std::endl;

    m_initialized = true;
    return true;
}

void CudaInference::shutdown() {
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    
    if (m_initialized) {
        cudaDeviceReset();
        m_initialized = false;
        m_deviceId = -1;
    }
}

CudaDeviceInfo CudaInference::getDeviceInfo() const {
    CudaDeviceInfo info{};
    
    if (!m_initialized) {
        info.isAvailable = false;
        return info;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, m_deviceId) != cudaSuccess) {
        info.isAvailable = false;
        return info;
    }

    info.deviceId = m_deviceId;
    info.name = prop.name;
    info.totalMemory = prop.totalGlobalMem;
    info.computeCapabilityMajor = prop.major;
    info.computeCapabilityMinor = prop.minor;
    info.multiProcessorCount = prop.multiProcessorCount;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.clockRateMHz = prop.clockRate / 1000;
    info.isAvailable = true;

    // Get free memory
    size_t freeMem, totalMem;
    if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess) {
        info.freeMemory = freeMem;
    }

    return info;
}

std::vector<CudaDeviceInfo> CudaInference::enumerateDevices() {
    std::vector<CudaDeviceInfo> devices;
    
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        return devices;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            CudaDeviceInfo info{};
            info.deviceId = i;
            info.name = prop.name;
            info.totalMemory = prop.totalGlobalMem;
            info.computeCapabilityMajor = prop.major;
            info.computeCapabilityMinor = prop.minor;
            info.multiProcessorCount = prop.multiProcessorCount;
            info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
            info.clockRateMHz = prop.clockRate / 1000;
            info.isAvailable = true;
            devices.push_back(info);
        }
    }

    return devices;
}

void* CudaInference::allocateDeviceMemory(size_t sizeBytes) {
    void* ptr = nullptr;
    if (!checkCudaError(cudaMalloc(&ptr, sizeBytes), "cudaMalloc")) {
        return nullptr;
    }
    return ptr;
}

void CudaInference::freeDeviceMemory(void* devicePtr) {
    if (devicePtr) {
        cudaFree(devicePtr);
    }
}

bool CudaInference::copyToDevice(void* dst, const void* src, size_t sizeBytes) {
    return checkCudaError(
        cudaMemcpyAsync(dst, src, sizeBytes, cudaMemcpyHostToDevice, m_stream),
        "cudaMemcpyHostToDevice"
    );
}

bool CudaInference::copyToHost(void* dst, const void* src, size_t sizeBytes) {
    return checkCudaError(
        cudaMemcpyAsync(dst, src, sizeBytes, cudaMemcpyDeviceToHost, m_stream),
        "cudaMemcpyDeviceToHost"
    );
}

void CudaInference::synchronize() {
    cudaStreamSynchronize(m_stream);
}

CudaInferenceResult CudaInference::runVectorAddBenchmark(size_t vectorSize) {
    CudaInferenceResult result{};
    
    if (!m_initialized) {
        result.success = false;
        result.errorMessage = "CUDA not initialized";
        return result;
    }

    size_t bytes = vectorSize * sizeof(float);

    // Allocate host memory
    std::vector<float> h_a(vectorSize);
    std::vector<float> h_b(vectorSize);
    std::vector<float> h_c(vectorSize);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < vectorSize; ++i) {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }

    // Allocate device memory
    float* d_a = static_cast<float*>(allocateDeviceMemory(bytes));
    float* d_b = static_cast<float*>(allocateDeviceMemory(bytes));
    float* d_c = static_cast<float*>(allocateDeviceMemory(bytes));

    if (!d_a || !d_b || !d_c) {
        result.success = false;
        result.errorMessage = "Failed to allocate device memory";
        freeDeviceMemory(d_a);
        freeDeviceMemory(d_b);
        freeDeviceMemory(d_c);
        return result;
    }

    // Copy to device
    copyToDevice(d_a, h_a.data(), bytes);
    copyToDevice(d_b, h_b.data(), bytes);

    // Time the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, m_stream);
    
    // Launch kernel
    launchVectorAdd(d_a, d_b, d_c, static_cast<int>(vectorSize), m_stream);
    
    cudaEventRecord(stop, m_stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    copyToHost(h_c.data(), d_c, bytes);
    synchronize();

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    freeDeviceMemory(d_a);
    freeDeviceMemory(d_b);
    freeDeviceMemory(d_c);

    result.success = true;
    result.executionTimeMs = milliseconds;
    result.outputData = std::move(h_c);

    std::cout << "[CUDA] Vector Add (" << vectorSize << " elements): " 
              << milliseconds << " ms" << std::endl;

    return result;
}

CudaInferenceResult CudaInference::runMatrixMultiplyBenchmark(int matrixSize) {
    CudaInferenceResult result{};
    
    if (!m_initialized) {
        result.success = false;
        result.errorMessage = "CUDA not initialized";
        return result;
    }

    size_t elements = static_cast<size_t>(matrixSize) * matrixSize;
    size_t bytes = elements * sizeof(float);

    // Allocate host memory
    std::vector<float> h_a(elements);
    std::vector<float> h_b(elements);
    std::vector<float> h_c(elements);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < elements; ++i) {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }

    // Allocate device memory
    float* d_a = static_cast<float*>(allocateDeviceMemory(bytes));
    float* d_b = static_cast<float*>(allocateDeviceMemory(bytes));
    float* d_c = static_cast<float*>(allocateDeviceMemory(bytes));

    if (!d_a || !d_b || !d_c) {
        result.success = false;
        result.errorMessage = "Failed to allocate device memory";
        freeDeviceMemory(d_a);
        freeDeviceMemory(d_b);
        freeDeviceMemory(d_c);
        return result;
    }

    // Copy to device
    copyToDevice(d_a, h_a.data(), bytes);
    copyToDevice(d_b, h_b.data(), bytes);

    // Time the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, m_stream);
    
    // Launch kernel
    launchMatrixMultiply(d_a, d_b, d_c, matrixSize, m_stream);
    
    cudaEventRecord(stop, m_stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    copyToHost(h_c.data(), d_c, bytes);
    synchronize();

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    freeDeviceMemory(d_a);
    freeDeviceMemory(d_b);
    freeDeviceMemory(d_c);

    result.success = true;
    result.executionTimeMs = milliseconds;
    result.outputData = std::move(h_c);

    std::cout << "[CUDA] Matrix Multiply (" << matrixSize << "x" << matrixSize << "): " 
              << milliseconds << " ms" << std::endl;

    return result;
}

CudaInferenceResult CudaInference::runNeuralLayerBenchmark(int inputSize, int outputSize, int batchSize) {
    CudaInferenceResult result{};
    
    if (!m_initialized) {
        result.success = false;
        result.errorMessage = "CUDA not initialized";
        return result;
    }

    size_t inputBytes = static_cast<size_t>(batchSize) * inputSize * sizeof(float);
    size_t weightsBytes = static_cast<size_t>(inputSize) * outputSize * sizeof(float);
    size_t biasBytes = static_cast<size_t>(outputSize) * sizeof(float);
    size_t outputBytes = static_cast<size_t>(batchSize) * outputSize * sizeof(float);

    // Allocate host memory
    std::vector<float> h_input(batchSize * inputSize);
    std::vector<float> h_weights(inputSize * outputSize);
    std::vector<float> h_bias(outputSize);
    std::vector<float> h_output(batchSize * outputSize);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& v : h_input) v = dist(gen);
    for (auto& v : h_weights) v = dist(gen) * 0.1f;
    for (auto& v : h_bias) v = dist(gen) * 0.01f;

    // Allocate device memory
    float* d_input = static_cast<float*>(allocateDeviceMemory(inputBytes));
    float* d_weights = static_cast<float*>(allocateDeviceMemory(weightsBytes));
    float* d_bias = static_cast<float*>(allocateDeviceMemory(biasBytes));
    float* d_output = static_cast<float*>(allocateDeviceMemory(outputBytes));

    if (!d_input || !d_weights || !d_bias || !d_output) {
        result.success = false;
        result.errorMessage = "Failed to allocate device memory";
        freeDeviceMemory(d_input);
        freeDeviceMemory(d_weights);
        freeDeviceMemory(d_bias);
        freeDeviceMemory(d_output);
        return result;
    }

    // Copy to device
    copyToDevice(d_input, h_input.data(), inputBytes);
    copyToDevice(d_weights, h_weights.data(), weightsBytes);
    copyToDevice(d_bias, h_bias.data(), biasBytes);

    // Time the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, m_stream);
    
    // Launch kernel
    launchNeuralLayer(d_input, d_weights, d_bias, d_output, 
                      inputSize, outputSize, batchSize, m_stream);
    
    cudaEventRecord(stop, m_stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    copyToHost(h_output.data(), d_output, outputBytes);
    synchronize();

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    freeDeviceMemory(d_input);
    freeDeviceMemory(d_weights);
    freeDeviceMemory(d_bias);
    freeDeviceMemory(d_output);

    result.success = true;
    result.executionTimeMs = milliseconds;
    result.outputData = std::move(h_output);

    std::cout << "[CUDA] Neural Layer (batch=" << batchSize 
              << ", " << inputSize << " -> " << outputSize << "): " 
              << milliseconds << " ms" << std::endl;

    return result;
}

} // namespace gpu
} // namespace aiforge
