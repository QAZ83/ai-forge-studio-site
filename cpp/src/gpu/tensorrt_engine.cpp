/**
 * AI Forge Studio - TensorRT Engine Implementation
 * Author: M.3R3
 * 
 * Advanced TensorRT inference engine with:
 * - CUDA Graph support
 * - Dynamic output allocation
 * - Comprehensive profiling
 * - Engine serialization
 */

#include "tensorrt_engine.h"

#ifdef HAS_TENSORRT

#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

namespace aiforge {
namespace gpu {

// =============================================================================
// TensorRT Logger Implementation
// =============================================================================

void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity > m_logLevel) return;
    
    const char* levelStr = "";
    switch (severity) {
        case Severity::kINTERNAL_ERROR: levelStr = "[INTERNAL ERROR]"; break;
        case Severity::kERROR:          levelStr = "[ERROR]"; break;
        case Severity::kWARNING:        levelStr = "[WARNING]"; break;
        case Severity::kINFO:           levelStr = "[INFO]"; break;
        case Severity::kVERBOSE:        levelStr = "[VERBOSE]"; break;
    }
    
    std::string message = std::string("[TensorRT] ") + levelStr + " " + msg;
    
    if (m_callback) {
        m_callback(severity, message);
    } else {
        std::cout << message << std::endl;
    }
}

// =============================================================================
// TensorRT Engine Profiler Implementation
// =============================================================================

void TensorRTEngineProfiler::reportLayerTime(const char* layerName, float timeMs) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Find existing entry or create new one
    auto it = std::find_if(m_data.begin(), m_data.end(),
        [layerName](const TensorRTProfilerData& d) { return d.layerName == layerName; });
    
    if (it != m_data.end()) {
        it->timeMs += timeMs;
        it->executionCount++;
    } else {
        m_data.push_back({layerName, timeMs, 1});
    }
}

std::vector<TensorRTProfilerData> TensorRTEngineProfiler::getProfilingData() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_data;
}

void TensorRTEngineProfiler::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_data.clear();
}

float TensorRTEngineProfiler::getTotalTime() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    float total = 0;
    for (const auto& d : m_data) {
        total += d.timeMs;
    }
    return total;
}

std::string TensorRTEngineProfiler::getSummary() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "TensorRT Layer Profiling Summary\n";
    oss << "================================\n";
    
    float total = 0;
    for (const auto& d : m_data) {
        total += d.timeMs;
    }
    
    for (const auto& d : m_data) {
        float avgTime = d.timeMs / d.executionCount;
        float percent = (d.timeMs / total) * 100.0f;
        oss << std::setw(40) << std::left << d.layerName 
            << " | " << std::setw(8) << std::right << avgTime << " ms"
            << " | " << std::setw(5) << percent << "%"
            << " | x" << d.executionCount << "\n";
    }
    
    oss << "--------------------------------\n";
    oss << "Total: " << total << " ms\n";
    
    return oss.str();
}

void TensorRTEngineProfiler::saveToFile(const std::string& path) const {
    std::ofstream file(path);
    if (file.is_open()) {
        file << getSummary();
        file.close();
    }
}

// =============================================================================
// Dynamic Output Allocator Implementation
// =============================================================================

DynamicOutputAllocator::DynamicOutputAllocator(
    const std::unordered_map<std::string, nvinfer1::DataType>& outputDtypes)
    : m_dtypes(outputDtypes)
{
}

DynamicOutputAllocator::~DynamicOutputAllocator() {
    reset();
}

void* DynamicOutputAllocator::reallocateOutputAsync(
    char const* tensorName,
    void* currentMemory,
    uint64_t size,
    uint64_t alignment,
    cudaStream_t stream)
{
    std::string name(tensorName);
    
    // Check if we need to reallocate
    auto allocIt = m_allocatedSizes.find(name);
    if (allocIt != m_allocatedSizes.end() && allocIt->second >= size) {
        // Current buffer is large enough
        m_sizes[name] = size;
        return m_buffers[name];
    }
    
    // Need to allocate new buffer
    void* newBuffer = nullptr;
    
    // Add some extra space to avoid frequent reallocations
    uint64_t allocSize = size + (size / 4);  // 25% extra
    
    cudaError_t err = cudaMallocAsync(&newBuffer, allocSize, stream);
    if (err != cudaSuccess) {
        std::cerr << "[TensorRT] Failed to allocate " << allocSize 
                  << " bytes for output " << name << std::endl;
        return nullptr;
    }
    
    // Free old buffer if exists
    auto bufIt = m_buffers.find(name);
    if (bufIt != m_buffers.end() && bufIt->second != nullptr) {
        cudaFreeAsync(bufIt->second, stream);
    }
    
    m_buffers[name] = newBuffer;
    m_sizes[name] = size;
    m_allocatedSizes[name] = allocSize;
    
    return newBuffer;
}

void DynamicOutputAllocator::notifyShape(
    char const* tensorName, 
    nvinfer1::Dims const& dims) noexcept
{
    m_shapes[tensorName] = dims;
}

void DynamicOutputAllocator::reset() {
    for (auto& [name, buffer] : m_buffers) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    m_buffers.clear();
    m_shapes.clear();
    m_sizes.clear();
    m_allocatedSizes.clear();
}

// =============================================================================
// CUDA Graph Executor Implementation
// =============================================================================

CudaGraphExecutor::~CudaGraphExecutor() {
    reset();
}

bool CudaGraphExecutor::capture(nvinfer1::IExecutionContext* context, cudaStream_t stream) {
    reset();
    
    // Begin capture
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Graph] Failed to begin capture: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Execute inference (this gets captured)
    if (!context->enqueueV3(stream)) {
        cudaStreamEndCapture(stream, &m_graph);
        std::cerr << "[CUDA Graph] Failed to enqueue during capture" << std::endl;
        return false;
    }
    
    // End capture
    err = cudaStreamEndCapture(stream, &m_graph);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Graph] Failed to end capture: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Instantiate the graph
    err = cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Graph] Failed to instantiate: " << cudaGetErrorString(err) << std::endl;
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
        return false;
    }
    
    m_captured = true;
    std::cout << "[CUDA Graph] Successfully captured execution graph" << std::endl;
    
    return true;
}

bool CudaGraphExecutor::execute(cudaStream_t stream) {
    if (!m_captured || !m_graphExec) {
        return false;
    }
    
    cudaError_t err = cudaGraphLaunch(m_graphExec, stream);
    return err == cudaSuccess;
}

void CudaGraphExecutor::reset() {
    if (m_graphExec) {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }
    if (m_graph) {
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
    }
    m_captured = false;
}

bool TensorRTEngine::buildFromONNX(const TensorRTConfig& config) {
    cleanup();
    m_config = config;
    m_name = config.engineName;
    
    std::cout << "[TensorRT] Building engine from ONNX: " << config.modelPath << std::endl;
    
    // Check if file exists
    std::ifstream file(config.modelPath, std::ios::binary);
    if (!file.good()) {
        m_lastError = "ONNX file not found: " + config.modelPath;
        std::cerr << "[TensorRT] " << m_lastError << std::endl;
        return false;
    }
    file.close();
    
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(m_logger)
    );
    if (!builder) {
        m_lastError = "Failed to create TensorRT builder";
        return false;
    }
    
    // Create network
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
    );
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch)
    );
    if (!network) {
        m_lastError = "Failed to create network definition";
        return false;
    }
    
    // Create parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, m_logger)
    );
    if (!parser) {
        m_lastError = "Failed to create ONNX parser";
        return false;
    }
    
    // Parse ONNX file
    if (!parser->parseFromFile(config.modelPath.c_str(), 
                                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        m_lastError = "Failed to parse ONNX file";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "[TensorRT] Parser error: " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    
    std::cout << "[TensorRT] ONNX parsed successfully" << std::endl;
    std::cout << "[TensorRT] Inputs: " << network->getNbInputs() << std::endl;
    std::cout << "[TensorRT] Outputs: " << network->getNbOutputs() << std::endl;
    
    // Create builder config
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig()
    );
    if (!builderConfig) {
        m_lastError = "Failed to create builder config";
        return false;
    }
    
    // Set workspace size
    builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.maxWorkspaceSize);
    
    // Enable FP16 if supported
    if (config.fp16Mode && builder->platformHasFastFp16()) {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TensorRT] FP16 mode enabled" << std::endl;
    }
    
    // Enable BF16 if requested (Ampere+)
    if (config.bf16Mode && builder->platformHasFastFp16()) {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kBF16);
        std::cout << "[TensorRT] BF16 mode enabled" << std::endl;
    }
    
    // Enable INT8 if requested
    if (config.int8Mode && builder->platformHasFastInt8()) {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::cout << "[TensorRT] INT8 mode enabled" << std::endl;
    }
    
    // Enable sparsity if requested (Ampere+)
    if (config.enableSparsity) {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
        std::cout << "[TensorRT] Sparse weights enabled" << std::endl;
    }
    
    // Set optimization level
    builderConfig->setBuilderOptimizationLevel(config.builderOptimizationLevel);
    
    // Handle dynamic shapes if specified
    if (!config.minShapes.empty()) {
        auto profile = builder->createOptimizationProfile();
        
        for (int i = 0; i < network->getNbInputs(); ++i) {
            auto* input = network->getInput(i);
            std::string name = input->getName();
            
            auto minIt = config.minShapes.find(name);
            auto optIt = config.optShapes.find(name);
            auto maxIt = config.maxShapes.find(name);
            
            if (minIt != config.minShapes.end() && 
                optIt != config.optShapes.end() && 
                maxIt != config.maxShapes.end()) {
                
                nvinfer1::Dims minDims{}, optDims{}, maxDims{};
                minDims.nbDims = optDims.nbDims = maxDims.nbDims = static_cast<int>(minIt->second.size());
                
                for (int d = 0; d < minDims.nbDims; ++d) {
                    minDims.d[d] = minIt->second[d];
                    optDims.d[d] = optIt->second[d];
                    maxDims.d[d] = maxIt->second[d];
                }
                
                profile->setDimensions(name.c_str(), 
                    nvinfer1::OptProfileSelector::kMIN, minDims);
                profile->setDimensions(name.c_str(), 
                    nvinfer1::OptProfileSelector::kOPT, optDims);
                profile->setDimensions(name.c_str(), 
                    nvinfer1::OptProfileSelector::kMAX, maxDims);
            }
        }
        
        builderConfig->addOptimizationProfile(profile);
    }
    
    // Build engine
    std::cout << "[TensorRT] Building optimized engine (this may take a while)..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *builderConfig)
    );
    
    auto endTime = std::chrono::high_resolution_clock::now();
    float buildTimeSeconds = std::chrono::duration<float>(endTime - startTime).count();
    
    if (!serializedEngine) {
        m_lastError = "Failed to build engine";
        return false;
    }
    
    std::cout << "[TensorRT] Engine built in " << buildTimeSeconds << " seconds" << std::endl;
    
    // Create runtime and deserialize
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) {
        m_lastError = "Failed to create runtime";
        return false;
    }
    
    m_engine.reset(m_runtime->deserializeCudaEngine(
        serializedEngine->data(), serializedEngine->size()
    ));
    if (!m_engine) {
        m_lastError = "Failed to deserialize engine";
        return false;
    }
    
    // Create execution context
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        m_lastError = "Failed to create execution context";
        return false;
    }
    
    // Setup tensor info and allocate buffers
    if (!allocateBuffers()) {
        return false;
    }
    
    // Apply runtime settings from config
    m_useCudaGraphs = config.enableCudaGraphs;
    m_usePreAllocatedOutputs = config.enablePreAllocatedOutputs;
    m_useDynamicAllocator = config.enableDynamicOutputAllocator;
    m_hardwareCompatible = config.hardwareCompatible;
    
    m_ready = true;
    std::cout << "[TensorRT] Engine built successfully!" << std::endl;
    
    return true;
}

bool TensorRTEngine::loadEngine(const std::string& enginePath) {
    std::cout << "[TensorRT] Loading engine: " << enginePath << std::endl;
    
    // Read engine file
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        m_lastError = "Engine file not found: " + enginePath;
        return false;
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        m_lastError = "Failed to read engine file";
        return false;
    }
    file.close();
    
    return loadEngineFromBuffer(buffer);
}

bool TensorRTEngine::loadEngineFromBuffer(const std::vector<char>& buffer) {
    cleanup();
    
    // Create runtime
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) {
        m_lastError = "Failed to create runtime";
        return false;
    }
    
    // Deserialize engine
    m_engine.reset(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        m_lastError = "Failed to deserialize engine";
        return false;
    }
    
    // Create execution context
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        m_lastError = "Failed to create execution context";
        return false;
    }
    
    // Setup tensor info and allocate buffers
    if (!allocateBuffers()) {
        return false;
    }
    
    m_ready = true;
    std::cout << "[TensorRT] Engine loaded successfully!" << std::endl;
    
    return true;
}

bool TensorRTEngine::saveEngine(const std::string& enginePath) {
    if (!m_engine) {
        m_lastError = "No engine to save";
        return false;
    }
    
    auto buffer = serializeEngine();
    if (buffer.empty()) {
        return false;
    }
    
    std::ofstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        m_lastError = "Failed to open output file";
        return false;
    }
    
    file.write(buffer.data(), buffer.size());
    file.close();
    
    std::cout << "[TensorRT] Engine saved to: " << enginePath << std::endl;
    return true;
}

std::vector<char> TensorRTEngine::serializeEngine() {
    if (!m_engine) {
        m_lastError = "No engine to serialize";
        return {};
    }
    
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(m_engine->serialize());
    if (!serialized) {
        m_lastError = "Failed to serialize engine";
        return {};
    }
    
    std::vector<char> buffer(serialized->size());
    std::memcpy(buffer.data(), serialized->data(), serialized->size());
    
    return buffer;
}

bool TensorRTEngine::allocateBuffers() {
    freeBuffers();
    m_tensors.clear();
    m_tensorNameToIndex.clear();
    
    int nbIOTensors = m_engine->getNbIOTensors();
    m_deviceBuffers.resize(nbIOTensors);
    
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = m_engine->getIOTensorName(i);
        nvinfer1::Dims dims = m_engine->getTensorShape(name);
        nvinfer1::DataType dtype = m_engine->getTensorDataType(name);
        bool isInput = m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        
        TensorInfo info;
        info.name = name;
        info.dims = dims;
        info.dataType = dtype;
        info.isInput = isInput;
        info.bindingIndex = i;
        info.sizeBytes = getTensorSize(info);
        
        // Allocate device buffer
        if (cudaMalloc(&m_deviceBuffers[i], info.sizeBytes) != cudaSuccess) {
            m_lastError = "Failed to allocate device buffer for: " + info.name;
            freeBuffers();
            return false;
        }
        
        m_tensors.push_back(info);
        m_tensorNameToIndex[info.name] = static_cast<int>(m_tensors.size() - 1);
        
        std::cout << "[TensorRT] " << (isInput ? "Input" : "Output") 
                  << " '" << name << "': ";
        for (int d = 0; d < dims.nbDims; ++d) {
            std::cout << dims.d[d] << (d < dims.nbDims - 1 ? "x" : "");
        }
        std::cout << " (" << info.sizeBytes << " bytes)" << std::endl;
    }
    
    return true;
}

void TensorRTEngine::freeBuffers() {
    for (auto& buffer : m_deviceBuffers) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    m_deviceBuffers.clear();
    
    for (auto& buffer : m_preAllocatedOutputs) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    m_preAllocatedOutputs.clear();
    
    for (auto& buffer : m_graphInputBuffers) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    m_graphInputBuffers.clear();
    
    for (auto& buffer : m_graphOutputBuffers) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    m_graphOutputBuffers.clear();
    
    if (m_outputAllocator) {
        m_outputAllocator->reset();
    }
}

size_t TensorRTEngine::getDataTypeSize(nvinfer1::DataType type) const {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kFP8:   return 1;
        case nvinfer1::DataType::kBF16:  return 2;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kINT4:  return 1;  // Packed
        default: return 0;
    }
}

size_t TensorRTEngine::getTensorSize(const TensorInfo& info) const {
    size_t size = getDataTypeSize(info.dataType);
    for (int i = 0; i < info.dims.nbDims; ++i) {
        // Handle dynamic dimensions (-1)
        int dim = info.dims.d[i];
        size *= (dim > 0) ? dim : 1;
    }
    return size;
}

std::string TensorRTEngine::computeShapeKey(
    const std::unordered_map<std::string, std::vector<int>>& shapes) const 
{
    std::ostringstream oss;
    for (const auto& [name, shape] : shapes) {
        oss << name << ":";
        for (size_t i = 0; i < shape.size(); ++i) {
            oss << shape[i];
            if (i < shape.size() - 1) oss << "x";
        }
        oss << ";";
    }
    return oss.str();
}

bool TensorRTEngine::updateInputShapes(
    const std::unordered_map<std::string, std::vector<int>>& shapes) 
{
    for (const auto& [name, shape] : shapes) {
        nvinfer1::Dims dims{};
        dims.nbDims = static_cast<int>(shape.size());
        for (int i = 0; i < dims.nbDims; ++i) {
            dims.d[i] = shape[i];
        }
        
        if (!m_context->setInputShape(name.c_str(), dims)) {
            m_lastError = "Failed to set input shape for: " + name;
            return false;
        }
    }
    return true;
}

std::vector<std::pair<std::string, std::vector<int>>> TensorRTEngine::getInputInfo() const {
    std::vector<std::pair<std::string, std::vector<int>>> result;
    for (const auto& tensor : m_tensors) {
        if (tensor.isInput) {
            std::vector<int> shape;
            for (int i = 0; i < tensor.dims.nbDims; ++i) {
                shape.push_back(tensor.dims.d[i]);
            }
            result.emplace_back(tensor.name, shape);
        }
    }
    return result;
}

std::vector<std::pair<std::string, std::vector<int>>> TensorRTEngine::getOutputInfo() const {
    std::vector<std::pair<std::string, std::vector<int>>> result;
    for (const auto& tensor : m_tensors) {
        if (!tensor.isInput) {
            std::vector<int> shape;
            for (int i = 0; i < tensor.dims.nbDims; ++i) {
                shape.push_back(tensor.dims.d[i]);
            }
            result.emplace_back(tensor.name, shape);
        }
    }
    return result;
}

TensorRTResult TensorRTEngine::infer(
    const std::unordered_map<std::string, std::vector<float>>& inputs
) {
    // Build shape map from current tensor dimensions
    std::unordered_map<std::string, std::vector<int>> inputShapes;
    for (const auto& tensor : m_tensors) {
        if (tensor.isInput) {
            std::vector<int> shape;
            for (int i = 0; i < tensor.dims.nbDims; ++i) {
                shape.push_back(tensor.dims.d[i]);
            }
            inputShapes[tensor.name] = shape;
        }
    }
    
    return inferWithShapes(inputs, inputShapes);
}

TensorRTResult TensorRTEngine::inferWithShapes(
    const std::unordered_map<std::string, std::vector<float>>& inputs,
    const std::unordered_map<std::string, std::vector<int>>& inputShapes
) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    TensorRTResult result{};
    
    if (!m_ready) {
        result.success = false;
        result.errorMessage = "Engine not ready";
        return result;
    }
    
    auto startTotal = std::chrono::high_resolution_clock::now();
    
    // Check if shapes changed
    std::string newShapeKey = computeShapeKey(inputShapes);
    bool shapeChanged = (newShapeKey != m_shapeKey);
    m_shapeKey = newShapeKey;
    
    // Update runtime states
    auto [needCudaGraphRecord, canUsePreAllocated, needCudaGraphReset] = 
        m_runtimeStates.setRuntimeStates(m_useCudaGraphs, m_usePreAllocatedOutputs, shapeChanged);
    
    // Reset CUDA graph if needed
    if (needCudaGraphReset && m_cudaGraphExecutor) {
        m_cudaGraphExecutor->reset();
    }
    
    // Update input shapes if dynamic
    if (shapeChanged) {
        if (!updateInputShapes(inputShapes)) {
            result.success = false;
            result.errorMessage = m_lastError;
            return result;
        }
    }
    
    // Copy inputs to device
    auto startH2D = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < m_tensors.size(); ++i) {
        if (m_tensors[i].isInput) {
            auto it = inputs.find(m_tensors[i].name);
            if (it == inputs.end()) {
                result.success = false;
                result.errorMessage = "Missing input: " + m_tensors[i].name;
                return result;
            }
            
            size_t expectedSize = m_tensors[i].sizeBytes / sizeof(float);
            if (it->second.size() != expectedSize) {
                result.success = false;
                result.errorMessage = "Input size mismatch for: " + m_tensors[i].name + 
                    " (expected " + std::to_string(expectedSize) + 
                    ", got " + std::to_string(it->second.size()) + ")";
                return result;
            }
            
            cudaMemcpyAsync(m_deviceBuffers[i], it->second.data(),
                           m_tensors[i].sizeBytes, cudaMemcpyHostToDevice, m_stream);
        }
    }
    
    cudaStreamSynchronize(m_stream);
    auto endH2D = std::chrono::high_resolution_clock::now();
    result.h2dTransferTimeMs = std::chrono::duration<float, std::milli>(endH2D - startH2D).count();
    
    // Set tensor addresses
    for (size_t i = 0; i < m_tensors.size(); ++i) {
        m_context->setTensorAddress(m_tensors[i].name.c_str(), m_deviceBuffers[i]);
    }
    
    // Set profiler if enabled
    if (m_profilingEnabled && m_profiler) {
        m_context->setProfiler(m_profiler.get());
    }
    
    auto startInfer = std::chrono::high_resolution_clock::now();
    
    // Execute inference
    bool inferSuccess = false;
    
    if (m_useCudaGraphs && m_cudaGraphExecutor) {
        if (needCudaGraphRecord) {
            // Capture new CUDA graph
            inferSuccess = m_cudaGraphExecutor->capture(m_context.get(), m_stream);
        } else if (m_cudaGraphExecutor->isValid()) {
            // Execute captured graph
            inferSuccess = m_cudaGraphExecutor->execute(m_stream);
            result.usedCudaGraph = true;
        } else {
            // Fallback to regular execution
            inferSuccess = m_context->enqueueV3(m_stream);
        }
    } else {
        inferSuccess = m_context->enqueueV3(m_stream);
    }
    
    if (!inferSuccess) {
        result.success = false;
        result.errorMessage = "Inference execution failed";
        return result;
    }
    
    cudaStreamSynchronize(m_stream);
    
    auto endInfer = std::chrono::high_resolution_clock::now();
    result.inferenceTimeMs = std::chrono::duration<float, std::milli>(endInfer - startInfer).count();
    
    // Copy outputs to host
    auto startD2H = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < m_tensors.size(); ++i) {
        if (!m_tensors[i].isInput) {
            size_t numElements = m_tensors[i].sizeBytes / sizeof(float);
            std::vector<float> output(numElements);
            cudaMemcpyAsync(output.data(), m_deviceBuffers[i],
                           m_tensors[i].sizeBytes, cudaMemcpyDeviceToHost, m_stream);
            cudaStreamSynchronize(m_stream);
            result.outputs[m_tensors[i].name] = std::move(output);
            
            // Store output shape
            std::vector<int> shape;
            for (int d = 0; d < m_tensors[i].dims.nbDims; ++d) {
                shape.push_back(m_tensors[i].dims.d[d]);
            }
            result.outputShapes[m_tensors[i].name] = shape;
        }
    }
    
    auto endD2H = std::chrono::high_resolution_clock::now();
    result.d2hTransferTimeMs = std::chrono::duration<float, std::milli>(endD2H - startD2H).count();
    
    result.preprocessTimeMs = result.h2dTransferTimeMs;
    result.postprocessTimeMs = result.d2hTransferTimeMs;
    result.usedPreAllocatedOutputs = canUsePreAllocated;
    
    result.success = true;
    return result;
}

std::unordered_map<std::string, std::vector<int>> TensorRTEngine::inferOutputShapes(
    const std::unordered_map<std::string, std::vector<int>>& inputShapes)
{
    std::unordered_map<std::string, std::vector<int>> result;
    
    if (!m_ready) return result;
    
    // Update input shapes
    for (const auto& [name, shape] : inputShapes) {
        nvinfer1::Dims dims{};
        dims.nbDims = static_cast<int>(shape.size());
        for (int i = 0; i < dims.nbDims; ++i) {
            dims.d[i] = shape[i];
        }
        m_context->setInputShape(name.c_str(), dims);
    }
    
    // Get inferred output shapes
    for (const auto& tensor : m_tensors) {
        if (!tensor.isInput) {
            nvinfer1::Dims dims = m_context->getTensorShape(tensor.name.c_str());
            std::vector<int> shape;
            for (int i = 0; i < dims.nbDims; ++i) {
                shape.push_back(dims.d[i]);
            }
            result[tensor.name] = shape;
        }
    }
    
    return result;
}

// CUDA Graph and optimization methods
void TensorRTEngine::setCudaGraphsEnabled(bool enable) {
    m_useCudaGraphs = enable;
    if (!enable && m_cudaGraphExecutor) {
        m_cudaGraphExecutor->reset();
    }
}

void TensorRTEngine::setPreAllocatedOutputs(bool enable) {
    m_usePreAllocatedOutputs = enable;
}

void TensorRTEngine::resetCudaGraph() {
    if (m_cudaGraphExecutor) {
        m_cudaGraphExecutor->reset();
    }
    m_runtimeStates.markContextChanged();
}

bool TensorRTEngine::setDeviceMemoryBudget(int64_t budget) {
    if (!m_context) return false;
    m_context->setDeviceMemoryV2(nullptr, budget);
    return true;
}

int64_t TensorRTEngine::getDeviceMemoryBudget() const {
    if (!m_engine) return 0;
    return m_engine->getDeviceMemorySizeV2();
}

int64_t TensorRTEngine::getStreamableDeviceMemoryBudget() const {
    if (!m_engine) return 0;
    return m_engine->getStreamableWeightsSize();
}

// Profiling methods
void TensorRTEngine::enableProfiling(const std::string& profilePath) {
    m_profilingEnabled = true;
    m_profilePath = profilePath.empty() ? 
        std::filesystem::temp_directory_path().string() + "/tensorrt_profile.txt" : 
        profilePath;
    
    if (m_profiler) {
        m_profiler->reset();
    }
}

void TensorRTEngine::disableProfiling() {
    m_profilingEnabled = false;
}

std::vector<TensorRTProfilerData> TensorRTEngine::getProfilingData() const {
    if (m_profiler) {
        return m_profiler->getProfilingData();
    }
    return {};
}

void TensorRTEngine::saveProfilingData(const std::string& path) const {
    if (m_profiler) {
        m_profiler->saveToFile(path);
    }
}

std::string TensorRTEngine::getEngineLayerInfo() const {
    if (!m_engine) return "";
    
    std::ostringstream oss;
    oss << "TensorRT Engine Layer Info:\n";
    oss << "===========================\n";
    
    int numLayers = m_engine->getNbLayers();
    oss << "Total layers: " << numLayers << "\n\n";
    
    return oss.str();
}

void TensorRTEngine::dumpEngineLayerInfo(const std::string& path) const {
    std::string info = getEngineLayerInfo();
    
    if (path.empty()) {
        std::cout << info << std::endl;
    } else {
        std::ofstream file(path);
        if (file.is_open()) {
            file << info;
            file.close();
        }
    }
}

FlattenedState TensorRTEngine::flatten() const {
    // Build binding name strings
    std::string inputBindings, outputBindings;
    for (const auto& tensor : m_tensors) {
        if (tensor.isInput) {
            if (!inputBindings.empty()) inputBindings += BINDING_DELIM;
            inputBindings += tensor.name;
        } else {
            if (!outputBindings.empty()) outputBindings += BINDING_DELIM;
            outputBindings += tensor.name;
        }
    }
    
    return FlattenedState{
        {"ABI_VERSION", TENSORRT_ABI_VERSION},
        {"name", m_name},
        {"device", "cuda:0"},
        {"engine", ""},  // Engine data stored separately
        {"input_bindings", inputBindings},
        {"output_bindings", outputBindings},
        {"hardware_compatible", m_hardwareCompatible ? "true" : "false"},
        {"requires_output_allocator", m_useDynamicAllocator ? "true" : "false"},
        {"metadata", m_serializedMetadata},
        {"platform", "windows-x64"}
    };
}

std::vector<std::string> TensorRTEngine::serialize() const {
    auto state = flatten();
    
    std::vector<std::string> result;
    result.push_back(std::get<1>(std::get<0>(state)));  // ABI version
    result.push_back(std::get<1>(std::get<1>(state)));  // name
    result.push_back(std::get<1>(std::get<2>(state)));  // device
    
    return result;
}

bool TensorRTEngine::verifySerializationFormat(const std::vector<std::string>& data) {
    if (data.empty()) return false;
    return data[0] == TENSORRT_ABI_VERSION;
}
    
    result.success = true;
    return result;
}

float TensorRTEngine::inferRaw(void** inputBuffers, void** outputBuffers) {
    if (!m_ready) return 0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, m_stream);
    
    // Set tensor addresses
    int inputIdx = 0, outputIdx = 0;
    for (size_t i = 0; i < m_tensors.size(); ++i) {
        if (m_tensors[i].isInput) {
            m_context->setTensorAddress(m_tensors[i].name.c_str(), inputBuffers[inputIdx++]);
        } else {
            m_context->setTensorAddress(m_tensors[i].name.c_str(), outputBuffers[outputIdx++]);
        }
    }
    
    m_context->enqueueV3(m_stream);
    
    cudaEventRecord(stop, m_stream);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms;
}

std::string TensorRTEngine::getEngineInfo() const {
    if (!m_engine) return "No engine loaded";
    
    std::ostringstream oss;
    oss << "TensorRT Engine Info:\n";
    oss << "  Name: " << m_name << "\n";
    oss << "  Tensors: " << m_tensors.size() << "\n";
    oss << "  Device Memory: " << (getDeviceMemoryBudget() / 1024 / 1024) << " MB\n";
    oss << "  CUDA Graphs: " << (m_useCudaGraphs ? "enabled" : "disabled") << "\n";
    oss << "  Pre-allocated Outputs: " << (m_usePreAllocatedOutputs ? "enabled" : "disabled") << "\n";
    
    for (const auto& tensor : m_tensors) {
        oss << "  - " << tensor.name << " (";
        oss << (tensor.isInput ? "input" : "output") << "): ";
        for (int i = 0; i < tensor.dims.nbDims; ++i) {
            oss << tensor.dims.d[i];
            if (i < tensor.dims.nbDims - 1) oss << "x";
        }
        oss << "\n";
    }
    
    return oss.str();
}

float TensorRTEngine::benchmark(int iterations, int warmupIterations) {
    if (!m_ready) return 0;
    
    std::cout << "[TensorRT] Running benchmark..." << std::endl;
    
    // Generate random input data
    std::unordered_map<std::string, std::vector<float>> inputs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (const auto& tensor : m_tensors) {
        if (tensor.isInput) {
            size_t numElements = tensor.sizeBytes / sizeof(float);
            std::vector<float> data(numElements);
            for (auto& v : data) v = dist(gen);
            inputs[tensor.name] = std::move(data);
        }
    }
    
    // Warmup
    std::cout << "[TensorRT] Warmup (" << warmupIterations << " iterations)..." << std::endl;
    for (int i = 0; i < warmupIterations; ++i) {
        infer(inputs);
    }
    
    // Benchmark
    std::cout << "[TensorRT] Benchmarking (" << iterations << " iterations)..." << std::endl;
    
    float totalTime = 0;
    float minTime = std::numeric_limits<float>::max();
    float maxTime = 0;
    
    for (int i = 0; i < iterations; ++i) {
        auto result = infer(inputs);
        totalTime += result.inferenceTimeMs;
        minTime = std::min(minTime, result.inferenceTimeMs);
        maxTime = std::max(maxTime, result.inferenceTimeMs);
    }
    
    float avgTime = totalTime / iterations;
    
    std::cout << "[TensorRT] Benchmark Results:" << std::endl;
    std::cout << "  Average: " << avgTime << " ms" << std::endl;
    std::cout << "  Min: " << minTime << " ms" << std::endl;
    std::cout << "  Max: " << maxTime << " ms" << std::endl;
    std::cout << "  Throughput: " << (1000.0f / avgTime) << " inferences/sec" << std::endl;
    
    return avgTime;
}

} // namespace gpu
} // namespace aiforge

#endif // HAS_TENSORRT
