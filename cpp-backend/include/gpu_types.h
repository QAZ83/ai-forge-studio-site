/**
 * AI Forge Studio - GPU Types and Structures
 * Common type definitions for GPU monitoring
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace aiforge {

// GPU Information Structure
struct GPUInfo {
    int deviceId;
    std::string name;
    std::string driverVersion;
    std::string cudaVersion;
    
    // Memory (in MB)
    uint64_t totalMemoryMB;
    uint64_t freeMemoryMB;
    uint64_t usedMemoryMB;
    
    // Compute capability
    int computeMajor;
    int computeMinor;
    
    // Cores
    int cudaCores;
    int tensorCores;
    int smCount;
    
    // Clocks (MHz)
    int gpuClockMHz;
    int memoryClockMHz;
    int maxGpuClockMHz;
    int maxMemoryClockMHz;
    
    // Temperature & Power
    int temperatureC;
    int powerDrawW;
    int powerLimitW;
    
    // Utilization (%)
    int gpuUtilization;
    int memoryUtilization;
};

// TensorRT Engine Info
struct TRTEngineInfo {
    std::string name;
    std::string precision;  // FP32, FP16, INT8
    size_t deviceMemoryMB;
    
    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        std::string dtype;
    };
    
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
};

// Inference Result
struct InferenceResult {
    bool success;
    float latencyMs;
    float throughputFPS;
    std::string errorMessage;
};

// Benchmark Result
struct BenchmarkResult {
    int iterations;
    int warmupIterations;
    float avgLatencyMs;
    float minLatencyMs;
    float maxLatencyMs;
    float p95LatencyMs;
    float throughputFPS;
};

} // namespace aiforge
