/**
 * AI Forge Studio - CUDA Helper Functions
 * Device query and utility functions
 */

#pragma once

#include <cuda_runtime.h>
#include <string>

namespace aiforge {
namespace cuda {

// Check CUDA error and throw if failed
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// Get device count
int getDeviceCount();

// Get device name
std::string getDeviceName(int deviceId = 0);

// Get compute capability
std::pair<int, int> getComputeCapability(int deviceId = 0);

// Get total memory in MB
size_t getTotalMemoryMB(int deviceId = 0);

// Get free memory in MB
size_t getFreeMemoryMB(int deviceId = 0);

// Get SM count
int getSMCount(int deviceId = 0);

// Get max clock rate in MHz
int getMaxClockMHz(int deviceId = 0);

// Set device
void setDevice(int deviceId);

// Synchronize device
void synchronize();

// Simple device test - returns true if CUDA works
bool testDevice(int deviceId = 0);

} // namespace cuda
} // namespace aiforge
