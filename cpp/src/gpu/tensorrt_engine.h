/**
 * AI Forge Studio - TensorRT Engine Module
 * Author: M.3R3
 * 
 * High-performance inference using NVIDIA TensorRT with advanced features:
 * - CUDA Graph support for reduced launch overhead
 * - Dynamic output allocation for variable-size outputs
 * - Pre-allocated output buffers for maximum performance
 * - Comprehensive profiling and benchmarking
 * - Engine serialization with metadata
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <functional>
#include <filesystem>
#include <tuple>

#ifdef HAS_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#endif

namespace aiforge {
namespace gpu {

// =============================================================================
// Version and Serialization Format
// =============================================================================
constexpr const char* TENSORRT_ABI_VERSION = "1.0.0";
constexpr char BINDING_DELIM = '%';

/**
 * Flattened state for engine serialization
 * Contains all metadata needed to reconstruct the engine state
 */
using FlattenedState = std::tuple<
    std::tuple<std::string, std::string>,  // ABI_VERSION
    std::tuple<std::string, std::string>,  // name
    std::tuple<std::string, std::string>,  // device
    std::tuple<std::string, std::string>,  // engine data
    std::tuple<std::string, std::string>,  // input binding names
    std::tuple<std::string, std::string>,  // output binding names
    std::tuple<std::string, std::string>,  // hardware compatibility
    std::tuple<std::string, std::string>,  // requires_output_allocator
    std::tuple<std::string, std::string>,  // serialized metadata
    std::tuple<std::string, std::string>>; // platform info

// =============================================================================
// Runtime State Management
// =============================================================================

/**
 * Manages runtime states for optimizations like CUDA Graphs and pre-allocated outputs
 */
struct TensorRTRuntimeStates {
    bool old_cudagraphs = false;
    bool old_pre_allocated_outputs = false;
    bool context_changed = false;

    /**
     * Evaluate conditions for CUDA Graph recording/reset and output reuse
     * @param new_cudagraphs Whether CUDA graphs should be enabled
     * @param new_pre_allocated_output Whether pre-allocated outputs should be used
     * @param shape_changed Whether input shapes have changed
     * @return Tuple of (need_cudagraphs_record, can_use_pre_allocated_outputs, need_cudagraphs_reset)
     */
    std::tuple<bool, bool, bool> setRuntimeStates(
        bool new_cudagraphs,
        bool new_pre_allocated_output,
        bool shape_changed) 
    {
        bool need_cudagraphs_record = false;
        bool can_use_pre_allocated_outputs = false;
        bool need_cudagraphs_reset = false;

        // CUDA graph record is required if enabled and (was disabled, shape changed, or context changed)
        if (new_cudagraphs && (!old_cudagraphs || shape_changed || context_changed)) {
            need_cudagraphs_record = true;
        }
        
        // Pre-allocated output can be used when both previous and current state are true without shape change
        if (old_pre_allocated_outputs && new_pre_allocated_output && !shape_changed) {
            can_use_pre_allocated_outputs = true;
        }
        
        // Reset CUDA graph if disabled, shape changed, or context changed
        if (!new_cudagraphs || shape_changed || context_changed) {
            need_cudagraphs_reset = true;
        }

        old_cudagraphs = new_cudagraphs;
        old_pre_allocated_outputs = new_pre_allocated_output;
        context_changed = false;

        return {need_cudagraphs_record, can_use_pre_allocated_outputs, need_cudagraphs_reset};
    }

    void markContextChanged() { context_changed = true; }
    void reset() { old_cudagraphs = false; old_pre_allocated_outputs = false; context_changed = false; }
};

// =============================================================================
// Profiler
// =============================================================================

/**
 * TensorRT engine profiler for detailed layer-by-layer timing
 */
struct TensorRTProfilerData {
    std::string layerName;
    float timeMs;
    size_t executionCount;
};

// =============================================================================
// TensorRT Inference Result
// =============================================================================

/**
 * TensorRT inference result with detailed timing and output data
 */
struct TensorRTResult {
    bool success = false;
    float inferenceTimeMs = 0.0f;
    float preprocessTimeMs = 0.0f;
    float postprocessTimeMs = 0.0f;
    float h2dTransferTimeMs = 0.0f;  // Host to device transfer time
    float d2hTransferTimeMs = 0.0f;  // Device to host transfer time
    std::string errorMessage;
    std::unordered_map<std::string, std::vector<float>> outputs;
    std::unordered_map<std::string, std::vector<int>> outputShapes;  // Dynamic shapes
    
    // Performance metrics
    size_t peakMemoryUsageBytes = 0;
    bool usedCudaGraph = false;
    bool usedPreAllocatedOutputs = false;
};

/**
 * TensorRT engine configuration with advanced options
 */
struct TensorRTConfig {
    std::string modelPath;                      // Path to ONNX or TensorRT engine file
    std::string engineName = "default";         // Engine identifier
    int maxBatchSize = 1;
    size_t maxWorkspaceSize = 1ULL << 30;       // 1 GB default
    bool fp16Mode = true;                       // Use FP16 precision
    bool int8Mode = false;                      // Use INT8 precision (requires calibration)
    bool bf16Mode = false;                      // Use BF16 precision (Ampere+)
    bool strictTypeConstraints = false;
    int dlaCore = -1;                           // DLA core (-1 = disabled)
    std::string calibrationCache;               // INT8 calibration cache file
    
    // Advanced options
    bool enableCudaGraphs = false;              // Enable CUDA Graph capture
    bool enablePreAllocatedOutputs = false;     // Pre-allocate output buffers
    bool enableDynamicOutputAllocator = false;  // Use dynamic output allocator
    bool hardwareCompatible = false;            // Build for hardware compatibility
    bool enableSparsity = false;                // Enable structured sparsity (Ampere+)
    bool enableTiming = true;                   // Enable kernel timing/selection
    int builderOptimizationLevel = 3;           // Optimization level (0-5)
    std::string profilePath;                    // Path for profiling data
    
    // Dynamic shape support
    std::unordered_map<std::string, std::vector<int>> minShapes;
    std::unordered_map<std::string, std::vector<int>> optShapes;
    std::unordered_map<std::string, std::vector<int>> maxShapes;
};

#ifdef HAS_TENSORRT

/**
 * TensorRT Logger with configurable verbosity
 */
class TensorRTLogger : public nvinfer1::ILogger {
public:
    using LogCallback = std::function<void(Severity, const std::string&)>;
    
    void log(Severity severity, const char* msg) noexcept override;
    void setLogLevel(Severity level) { m_logLevel = level; }
    Severity getLogLevel() const { return m_logLevel; }
    void setCallback(LogCallback callback) { m_callback = callback; }
    
private:
    Severity m_logLevel = Severity::kWARNING;
    LogCallback m_callback = nullptr;
};

/**
 * TensorRT Engine Profiler for layer-by-layer analysis
 */
class TensorRTEngineProfiler : public nvinfer1::IProfiler {
public:
    void reportLayerTime(const char* layerName, float timeMs) noexcept override;
    
    std::vector<TensorRTProfilerData> getProfilingData() const;
    void reset();
    float getTotalTime() const;
    std::string getSummary() const;
    void saveToFile(const std::string& path) const;
    
private:
    std::vector<TensorRTProfilerData> m_data;
    mutable std::mutex m_mutex;
};

/**
 * Dynamic Output Allocator for variable-size outputs
 * Implements TensorRT's IOutputAllocator interface
 */
class DynamicOutputAllocator : public nvinfer1::IOutputAllocator {
public:
    explicit DynamicOutputAllocator(
        const std::unordered_map<std::string, nvinfer1::DataType>& outputDtypes);
    ~DynamicOutputAllocator();

    void* reallocateOutputAsync(
        char const* tensorName,
        void* currentMemory,
        uint64_t size,
        uint64_t alignment,
        cudaStream_t stream) override;

    void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override;

    const std::unordered_map<std::string, void*>& getBuffers() const { return m_buffers; }
    const std::unordered_map<std::string, nvinfer1::Dims>& getShapes() const { return m_shapes; }
    const std::unordered_map<std::string, size_t>& getSizes() const { return m_sizes; }
    
    void reset();

private:
    std::unordered_map<std::string, nvinfer1::DataType> m_dtypes;
    std::unordered_map<std::string, void*> m_buffers;
    std::unordered_map<std::string, nvinfer1::Dims> m_shapes;
    std::unordered_map<std::string, size_t> m_sizes;
    std::unordered_map<std::string, size_t> m_allocatedSizes;
};

/**
 * CUDA Graph wrapper for TensorRT execution
 */
class CudaGraphExecutor {
public:
    CudaGraphExecutor() = default;
    ~CudaGraphExecutor();
    
    bool capture(nvinfer1::IExecutionContext* context, cudaStream_t stream);
    bool execute(cudaStream_t stream);
    void reset();
    bool isValid() const { return m_graphExec != nullptr; }
    
private:
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_captured = false;
};

/**
 * TensorRT Engine - High-performance inference engine
 * 
 * Features:
 * - ONNX model import and optimization
 * - Engine serialization/deserialization
 * - CUDA Graph support for reduced latency
 * - Dynamic output allocation
 * - Comprehensive profiling
 * - Thread-safe execution
 */
class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();

    // Disable copy
    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;
    
    // Enable move
    TensorRTEngine(TensorRTEngine&&) noexcept;
    TensorRTEngine& operator=(TensorRTEngine&&) noexcept;

    // =========================================================================
    // Engine Building and Loading
    // =========================================================================
    
    /**
     * Build engine from ONNX model
     */
    bool buildFromONNX(const TensorRTConfig& config);

    /**
     * Load pre-built TensorRT engine from file
     */
    bool loadEngine(const std::string& enginePath);
    
    /**
     * Load engine from memory buffer
     */
    bool loadEngineFromBuffer(const std::vector<char>& buffer);

    /**
     * Save built engine to file
     */
    bool saveEngine(const std::string& enginePath);
    
    /**
     * Serialize engine to memory buffer
     */
    std::vector<char> serializeEngine();

    // =========================================================================
    // Engine State and Info
    // =========================================================================
    
    bool isReady() const { return m_ready; }
    std::string getName() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }
    
    std::vector<std::pair<std::string, std::vector<int>>> getInputInfo() const;
    std::vector<std::pair<std::string, std::vector<int>>> getOutputInfo() const;
    std::string getEngineInfo() const;
    std::string getLastError() const { return m_lastError; }
    
    /**
     * Get detailed layer information from the engine
     */
    std::string getEngineLayerInfo() const;
    void dumpEngineLayerInfo(const std::string& path = "") const;

    // =========================================================================
    // Inference
    // =========================================================================
    
    /**
     * Run inference with named inputs
     */
    TensorRTResult infer(const std::unordered_map<std::string, std::vector<float>>& inputs);
    
    /**
     * Run inference with dynamic shapes
     */
    TensorRTResult inferWithShapes(
        const std::unordered_map<std::string, std::vector<float>>& inputs,
        const std::unordered_map<std::string, std::vector<int>>& inputShapes);

    /**
     * Run inference with raw device pointers (for advanced use)
     */
    float inferRaw(void** inputBuffers, void** outputBuffers);
    
    /**
     * Infer output shapes without running inference (for pre-allocation)
     */
    std::unordered_map<std::string, std::vector<int>> inferOutputShapes(
        const std::unordered_map<std::string, std::vector<int>>& inputShapes);

    // =========================================================================
    // Performance Optimization
    // =========================================================================
    
    /**
     * Enable/disable CUDA Graph capture for this engine
     */
    void setCudaGraphsEnabled(bool enable);
    bool getCudaGraphsEnabled() const { return m_useCudaGraphs; }
    
    /**
     * Enable/disable pre-allocated output buffers
     */
    void setPreAllocatedOutputs(bool enable);
    bool getPreAllocatedOutputs() const { return m_usePreAllocatedOutputs; }
    
    /**
     * Reset captured CUDA graph (required when shapes change)
     */
    void resetCudaGraph();
    
    /**
     * Set device memory budget for the engine
     */
    bool setDeviceMemoryBudget(int64_t budget);
    int64_t getDeviceMemoryBudget() const;
    int64_t getStreamableDeviceMemoryBudget() const;
    
    // =========================================================================
    // Profiling
    // =========================================================================
    
    void enableProfiling(const std::string& profilePath = "");
    void disableProfiling();
    bool isProfilingEnabled() const { return m_profilingEnabled; }
    std::vector<TensorRTProfilerData> getProfilingData() const;
    void saveProfilingData(const std::string& path) const;

    /**
     * Run benchmark with random data
     */
    float benchmark(int iterations = 100, int warmupIterations = 10);

    // =========================================================================
    // Serialization
    // =========================================================================
    
    /**
     * Flatten engine state for serialization
     */
    FlattenedState flatten() const;
    
    /**
     * Serialize engine and metadata to string vector
     */
    std::vector<std::string> serialize() const;
    
    /**
     * Verify serialization format
     */
    static bool verifySerializationFormat(const std::vector<std::string>& data);

    // =========================================================================
    // Runtime State
    // =========================================================================
    
    TensorRTRuntimeStates& getRuntimeStates() { return m_runtimeStates; }
    const TensorRTRuntimeStates& getRuntimeStates() const { return m_runtimeStates; }

private:
    bool m_ready = false;
    std::string m_name = "default";
    std::string m_lastError;
    mutable std::mutex m_mutex;
    
    // TensorRT objects
    TensorRTLogger m_logger;
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    
    // CUDA resources
    cudaStream_t m_stream = nullptr;
    
    // Tensor information
    struct TensorInfo {
        std::string name;
        nvinfer1::Dims dims;
        nvinfer1::DataType dataType;
        size_t sizeBytes;
        bool isInput;
        int bindingIndex;
    };
    std::vector<TensorInfo> m_tensors;
    std::unordered_map<std::string, int> m_tensorNameToIndex;
    
    // Device buffers
    std::vector<void*> m_deviceBuffers;
    
    // CUDA Graph support
    bool m_useCudaGraphs = false;
    std::unique_ptr<CudaGraphExecutor> m_cudaGraphExecutor;
    std::string m_shapeKey = "None";
    std::vector<void*> m_graphInputBuffers;
    std::vector<void*> m_graphOutputBuffers;
    
    // Pre-allocated outputs
    bool m_usePreAllocatedOutputs = false;
    std::vector<void*> m_preAllocatedOutputs;
    
    // Dynamic output allocator
    bool m_useDynamicAllocator = false;
    std::unique_ptr<DynamicOutputAllocator> m_outputAllocator;
    
    // Runtime state tracking
    TensorRTRuntimeStates m_runtimeStates;
    
    // Profiling
    bool m_profilingEnabled = false;
    std::string m_profilePath;
    std::unique_ptr<TensorRTEngineProfiler> m_profiler;
    
    // Configuration
    TensorRTConfig m_config;
    bool m_hardwareCompatible = false;
    std::string m_serializedMetadata;
    
    // Private methods
    void cleanup();
    bool allocateBuffers();
    void freeBuffers();
    size_t getDataTypeSize(nvinfer1::DataType type) const;
    size_t getTensorSize(const TensorInfo& info) const;
    std::string computeShapeKey(const std::unordered_map<std::string, std::vector<int>>& shapes) const;
    bool updateInputShapes(const std::unordered_map<std::string, std::vector<int>>& shapes);
};

#else // !HAS_TENSORRT

// =============================================================================
// Stub implementations when TensorRT is not available
// =============================================================================

struct TensorRTRuntimeStates {
    bool old_cudagraphs = false;
    bool old_pre_allocated_outputs = false;
    bool context_changed = false;
    std::tuple<bool, bool, bool> setRuntimeStates(bool, bool, bool) { return {false, false, false}; }
    void markContextChanged() {}
    void reset() {}
};

struct TensorRTProfilerData {
    std::string layerName;
    float timeMs = 0;
    size_t executionCount = 0;
};

/**
 * Stub TensorRT Engine (when TensorRT is not available)
 */
class TensorRTEngine {
public:
    TensorRTEngine() = default;
    ~TensorRTEngine() = default;

    bool buildFromONNX(const TensorRTConfig&) { 
        m_lastError = "TensorRT not available - install NVIDIA TensorRT SDK"; 
        return false; 
    }
    bool loadEngine(const std::string&) { 
        m_lastError = "TensorRT not available"; 
        return false; 
    }
    bool loadEngineFromBuffer(const std::vector<char>&) { return false; }
    bool saveEngine(const std::string&) { return false; }
    std::vector<char> serializeEngine() { return {}; }
    bool isReady() const { return false; }
    std::string getName() const { return "stub"; }
    void setName(const std::string&) {}
    
    std::vector<std::pair<std::string, std::vector<int>>> getInputInfo() const { return {}; }
    std::vector<std::pair<std::string, std::vector<int>>> getOutputInfo() const { return {}; }
    std::string getEngineInfo() const { return "TensorRT not available"; }
    std::string getEngineLayerInfo() const { return ""; }
    void dumpEngineLayerInfo(const std::string& = "") const {}
    
    TensorRTResult infer(const std::unordered_map<std::string, std::vector<float>>&) {
        return { false, 0, 0, 0, 0, 0, "TensorRT not available", {}, {} };
    }
    TensorRTResult inferWithShapes(
        const std::unordered_map<std::string, std::vector<float>>&,
        const std::unordered_map<std::string, std::vector<int>>&) {
        return { false, 0, 0, 0, 0, 0, "TensorRT not available", {}, {} };
    }
    float inferRaw(void**, void**) { return 0; }
    std::unordered_map<std::string, std::vector<int>> inferOutputShapes(
        const std::unordered_map<std::string, std::vector<int>>&) { return {}; }
    
    void setCudaGraphsEnabled(bool) {}
    bool getCudaGraphsEnabled() const { return false; }
    void setPreAllocatedOutputs(bool) {}
    bool getPreAllocatedOutputs() const { return false; }
    void resetCudaGraph() {}
    bool setDeviceMemoryBudget(int64_t) { return false; }
    int64_t getDeviceMemoryBudget() const { return 0; }
    int64_t getStreamableDeviceMemoryBudget() const { return 0; }
    
    void enableProfiling(const std::string& = "") {}
    void disableProfiling() {}
    bool isProfilingEnabled() const { return false; }
    std::vector<TensorRTProfilerData> getProfilingData() const { return {}; }
    void saveProfilingData(const std::string&) const {}
    float benchmark(int = 100, int = 10) { return 0; }
    
    std::string getLastError() const { return m_lastError; }
    
    TensorRTRuntimeStates& getRuntimeStates() { return m_runtimeStates; }
    const TensorRTRuntimeStates& getRuntimeStates() const { return m_runtimeStates; }

private:
    std::string m_lastError = "TensorRT not available";
    TensorRTRuntimeStates m_runtimeStates;
};

#endif // HAS_TENSORRT

} // namespace gpu
} // namespace aiforge
