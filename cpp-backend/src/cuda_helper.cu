/**
 * AI Forge Studio - CUDA Helper Implementation
 * CUDA device query and utility functions
 */

#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace aiforge {
namespace cuda {

int getDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

std::string getDeviceName(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    return std::string(prop.name);
}

std::pair<int, int> getComputeCapability(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    return {prop.major, prop.minor};
}

size_t getTotalMemoryMB(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    return prop.totalGlobalMem / (1024 * 1024);
}

size_t getFreeMemoryMB(int deviceId) {
    setDevice(deviceId);
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free / (1024 * 1024);
}

int getSMCount(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    return prop.multiProcessorCount;
}

int getMaxClockMHz(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    // In CUDA 13.0+, use NVML for clock info or return a reasonable default
    // The clockRate field was removed in newer CUDA versions
    // We'll get this from NVML in gpu_info.cpp instead
    return 2520; // Default max clock for RTX 5090 (will be overridden by NVML)
}

void setDevice(int deviceId) {
    CUDA_CHECK(cudaSetDevice(deviceId));
}

void synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool testDevice(int deviceId) {
    try {
        setDevice(deviceId);
        
        // Allocate a small buffer
        float* d_test = nullptr;
        CUDA_CHECK(cudaMalloc(&d_test, sizeof(float) * 1024));
        
        // Free it
        CUDA_CHECK(cudaFree(d_test));
        
        synchronize();
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace cuda
} // namespace aiforge

// =============================================================================
// Simple CUDA Kernel for testing
// =============================================================================

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Export a test function that runs a simple kernel
extern "C" bool runSimpleKernelTest() {
    const int N = 1024;
    const size_t size = N * sizeof(float);
    
    // Allocate host memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Copy back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            break;
        }
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    return success;
}
