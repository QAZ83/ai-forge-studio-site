/**
 * AI Forge Studio - CUDA Inference Module
 * Author: M.3R3
 * 
 * Provides GPU-accelerated computation using CUDA.
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace aiforge {
namespace gpu {

/**
 * CUDA device information structure
 */
struct CudaDeviceInfo {
    int deviceId;
    std::string name;
    size_t totalMemory;
    size_t freeMemory;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int clockRateMHz;
    bool isAvailable;
};

/**
 * CUDA inference result
 */
struct CudaInferenceResult {
    bool success;
    float executionTimeMs;
    std::string errorMessage;
    std::vector<float> outputData;
};

/**
 * CUDA Inference Engine
 * 
 * Manages CUDA context and provides GPU computation capabilities.
 */
class CudaInference {
public:
    CudaInference();
    ~CudaInference();

    // Disable copy
    CudaInference(const CudaInference&) = delete;
    CudaInference& operator=(const CudaInference&) = delete;

    /**
     * Initialize CUDA context on specified device
     * @param deviceId GPU device ID (default: 0)
     * @return true if initialization successful
     */
    bool initialize(int deviceId = 0);

    /**
     * Shutdown CUDA context and free resources
     */
    void shutdown();

    /**
     * Check if CUDA is initialized and ready
     */
    bool isReady() const { return m_initialized; }

    /**
     * Get information about the current CUDA device
     */
    CudaDeviceInfo getDeviceInfo() const;

    /**
     * Get list of all available CUDA devices
     */
    static std::vector<CudaDeviceInfo> enumerateDevices();

    /**
     * Run a simple vector addition benchmark
     * @param vectorSize Size of vectors to add
     * @return Inference result with timing
     */
    CudaInferenceResult runVectorAddBenchmark(size_t vectorSize = 1000000);

    /**
     * Run a matrix multiplication benchmark
     * @param matrixSize Size of square matrices (N x N)
     * @return Inference result with timing
     */
    CudaInferenceResult runMatrixMultiplyBenchmark(int matrixSize = 1024);

    /**
     * Run a simple neural network layer simulation
     * @param inputSize Input dimension
     * @param outputSize Output dimension
     * @param batchSize Batch size
     * @return Inference result with timing
     */
    CudaInferenceResult runNeuralLayerBenchmark(int inputSize = 512, 
                                                  int outputSize = 256, 
                                                  int batchSize = 32);

    /**
     * Allocate GPU memory
     * @param sizeBytes Size in bytes
     * @return Device pointer or nullptr on failure
     */
    void* allocateDeviceMemory(size_t sizeBytes);

    /**
     * Free GPU memory
     * @param devicePtr Pointer to device memory
     */
    void freeDeviceMemory(void* devicePtr);

    /**
     * Copy data from host to device
     */
    bool copyToDevice(void* dst, const void* src, size_t sizeBytes);

    /**
     * Copy data from device to host
     */
    bool copyToHost(void* dst, const void* src, size_t sizeBytes);

    /**
     * Synchronize CUDA stream
     */
    void synchronize();

    /**
     * Get last CUDA error message
     */
    std::string getLastError() const { return m_lastError; }

private:
    bool m_initialized;
    int m_deviceId;
    cudaStream_t m_stream;
    std::string m_lastError;

    void setError(const std::string& error);
    bool checkCudaError(cudaError_t error, const char* operation);
};

// External CUDA kernel declarations (implemented in cuda_kernels.cu)
extern "C" {
    void launchVectorAdd(float* a, float* b, float* c, int n, cudaStream_t stream);
    void launchMatrixMultiply(float* a, float* b, float* c, int n, cudaStream_t stream);
    void launchNeuralLayer(float* input, float* weights, float* bias, float* output,
                           int inputSize, int outputSize, int batchSize, cudaStream_t stream);
}

} // namespace gpu
} // namespace aiforge
