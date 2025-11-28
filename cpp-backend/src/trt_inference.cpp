/**
 * AI Forge Studio - TensorRT Inference Tool
 * Loads TensorRT engine and runs inference
 * 
 * Build: cmake --build build --target trt_inference
 * Run:   ./build/bin/trt_inference <engine_file.trt>
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <numeric>
#include <algorithm>
#include <iomanip>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "cuda_helper.h"
#include "gpu_types.h"

// =============================================================================
// TensorRT Logger
// =============================================================================

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

// =============================================================================
// TensorRT Engine Wrapper
// =============================================================================

class TRTEngine {
public:
    TRTEngine() : logger_(), runtime_(nullptr), engine_(nullptr), context_(nullptr) {}
    
    ~TRTEngine() {
        // Free CUDA buffers
        for (void* buf : deviceBuffers_) {
            if (buf) cudaFree(buf);
        }
    }
    
    bool loadEngine(const std::string& enginePath) {
        std::cout << "Loading TensorRT engine: " << enginePath << std::endl;
        
        // Read engine file
        std::ifstream file(enginePath, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open engine file: " << enginePath << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();
        
        std::cout << "Engine file size: " << (size / 1024.0 / 1024.0) << " MB" << std::endl;
        
        // Create runtime
        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_) {
            std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        // Deserialize engine
        engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
        if (!engine_) {
            std::cerr << "Error: Failed to deserialize engine" << std::endl;
            return false;
        }
        
        // Create execution context
        context_ = engine_->createExecutionContext();
        if (!context_) {
            std::cerr << "Error: Failed to create execution context" << std::endl;
            return false;
        }
        
        // Allocate buffers
        if (!allocateBuffers()) {
            return false;
        }
        
        std::cout << "✅ Engine loaded successfully!" << std::endl;
        printEngineInfo();
        
        return true;
    }
    
    bool buildFromONNX(const std::string& onnxPath, const std::string& savePath = "") {
        std::cout << "Building TensorRT engine from ONNX: " << onnxPath << std::endl;
        
        // Create builder
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(logger_));
        if (!builder) {
            std::cerr << "Error: Failed to create builder" << std::endl;
            return false;
        }
        
        // Create network
        const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cerr << "Error: Failed to create network" << std::endl;
            return false;
        }
        
        // Create ONNX parser
        auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, logger_));
        if (!parser) {
            std::cerr << "Error: Failed to create ONNX parser" << std::endl;
            return false;
        }
        
        // Parse ONNX file
        if (!parser->parseFromFile(onnxPath.c_str(), 
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Error: Failed to parse ONNX file" << std::endl;
            return false;
        }
        
        std::cout << "ONNX model parsed successfully" << std::endl;
        
        // Create builder config
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
            builder->createBuilderConfig());
        if (!config) {
            std::cerr << "Error: Failed to create builder config" << std::endl;
            return false;
        }
        
        // Set workspace size (1 GB)
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
        
        // Enable FP16 if supported
        if (builder->platformHasFastFp16()) {
            std::cout << "Enabling FP16 mode (Tensor Cores)" << std::endl;
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        
        // Build engine
        std::cout << "Building optimized engine (this may take a few minutes)..." << std::endl;
        auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *config));
        if (!serializedEngine) {
            std::cerr << "Error: Failed to build engine" << std::endl;
            return false;
        }
        
        // Save engine if path provided
        if (!savePath.empty()) {
            std::ofstream outFile(savePath, std::ios::binary);
            outFile.write(static_cast<const char*>(serializedEngine->data()), 
                         serializedEngine->size());
            std::cout << "Engine saved to: " << savePath << std::endl;
        }
        
        // Create runtime and deserialize
        runtime_ = nvinfer1::createInferRuntime(logger_);
        engine_ = runtime_->deserializeCudaEngine(serializedEngine->data(), 
                                                   serializedEngine->size());
        context_ = engine_->createExecutionContext();
        
        if (!allocateBuffers()) {
            return false;
        }
        
        std::cout << "✅ Engine built successfully!" << std::endl;
        printEngineInfo();
        
        return true;
    }
    
    aiforge::InferenceResult infer(const std::vector<float>& inputData) {
        aiforge::InferenceResult result;
        result.success = false;
        
        if (!context_) {
            result.errorMessage = "No execution context";
            return result;
        }
        
        // Copy input to device
        cudaMemcpy(deviceBuffers_[0], inputData.data(), 
                   inputData.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Run inference with timing
        auto start = std::chrono::high_resolution_clock::now();
        
        bool status = context_->executeV2(deviceBuffers_.data());
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        
        if (!status) {
            result.errorMessage = "Inference execution failed";
            return result;
        }
        
        result.success = true;
        result.latencyMs = std::chrono::duration<float, std::milli>(end - start).count();
        result.throughputFPS = 1000.0f / result.latencyMs;
        
        return result;
    }
    
    aiforge::BenchmarkResult benchmark(int warmupIterations = 10, int iterations = 100) {
        aiforge::BenchmarkResult result;
        result.warmupIterations = warmupIterations;
        result.iterations = iterations;
        
        // Create dummy input
        std::vector<float> dummyInput(inputSize_, 0.5f);
        
        // Warmup
        std::cout << "Warming up (" << warmupIterations << " iterations)..." << std::endl;
        for (int i = 0; i < warmupIterations; i++) {
            infer(dummyInput);
        }
        
        // Benchmark
        std::cout << "Benchmarking (" << iterations << " iterations)..." << std::endl;
        std::vector<float> latencies;
        latencies.reserve(iterations);
        
        for (int i = 0; i < iterations; i++) {
            auto inferResult = infer(dummyInput);
            if (inferResult.success) {
                latencies.push_back(inferResult.latencyMs);
            }
        }
        
        if (latencies.empty()) {
            return result;
        }
        
        // Calculate statistics
        std::sort(latencies.begin(), latencies.end());
        
        result.avgLatencyMs = std::accumulate(latencies.begin(), latencies.end(), 0.0f) 
                              / latencies.size();
        result.minLatencyMs = latencies.front();
        result.maxLatencyMs = latencies.back();
        result.p95LatencyMs = latencies[static_cast<size_t>(latencies.size() * 0.95)];
        result.throughputFPS = 1000.0f / result.avgLatencyMs;
        
        return result;
    }
    
private:
    bool allocateBuffers() {
        int numBindings = engine_->getNbIOTensors();
        deviceBuffers_.resize(numBindings);
        
        for (int i = 0; i < numBindings; i++) {
            const char* name = engine_->getIOTensorName(i);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            nvinfer1::DataType dtype = engine_->getTensorDataType(name);
            
            size_t size = 1;
            for (int j = 0; j < dims.nbDims; j++) {
                size *= dims.d[j];
            }
            
            size_t bytes = size * sizeof(float); // Assuming float32
            
            cudaError_t err = cudaMalloc(&deviceBuffers_[i], bytes);
            if (err != cudaSuccess) {
                std::cerr << "Error allocating GPU memory: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
            
            // Store input size for dummy data generation
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                inputSize_ = size;
            }
        }
        
        return true;
    }
    
    void printEngineInfo() {
        std::cout << "\n--- Engine Info ---" << std::endl;
        
        int numBindings = engine_->getNbIOTensors();
        for (int i = 0; i < numBindings; i++) {
            const char* name = engine_->getIOTensorName(i);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            
            std::cout << (mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output");
            std::cout << " '" << name << "': [";
            for (int j = 0; j < dims.nbDims; j++) {
                std::cout << dims.d[j];
                if (j < dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    TRTLogger logger_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    std::vector<void*> deviceBuffers_;
    size_t inputSize_ = 0;
};

// =============================================================================
// Main
// =============================================================================

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --engine <file.trt>    Load pre-built TensorRT engine" << std::endl;
    std::cout << "  --onnx <file.onnx>     Build engine from ONNX model" << std::endl;
    std::cout << "  --save <file.trt>      Save built engine (with --onnx)" << std::endl;
    std::cout << "  --benchmark            Run performance benchmark" << std::endl;
    std::cout << "  --iterations <N>       Benchmark iterations (default: 100)" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           AI FORGE STUDIO - TensorRT Inference                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    // Parse arguments
    std::string enginePath;
    std::string onnxPath;
    std::string savePath;
    bool runBenchmark = false;
    int iterations = 100;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--engine" && i + 1 < argc) {
            enginePath = argv[++i];
        } else if (arg == "--onnx" && i + 1 < argc) {
            onnxPath = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            savePath = argv[++i];
        } else if (arg == "--benchmark") {
            runBenchmark = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    if (enginePath.empty() && onnxPath.empty()) {
        std::cout << "No model specified. Running in demo mode.\n" << std::endl;
        std::cout << "To load a model, use:" << std::endl;
        std::cout << "  " << argv[0] << " --engine model.trt" << std::endl;
        std::cout << "  " << argv[0] << " --onnx model.onnx --save model.trt" << std::endl;
        std::cout << std::endl;
        
        // Show TensorRT version
        std::cout << "TensorRT Version: " << NV_TENSORRT_MAJOR << "." 
                  << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;
        
        std::cout << "\n✅ TensorRT is properly configured and ready!" << std::endl;
        return 0;
    }
    
    // Create engine
    TRTEngine engine;
    
    if (!onnxPath.empty()) {
        if (!engine.buildFromONNX(onnxPath, savePath)) {
            return 1;
        }
    } else if (!enginePath.empty()) {
        if (!engine.loadEngine(enginePath)) {
            return 1;
        }
    }
    
    // Run benchmark if requested
    if (runBenchmark) {
        std::cout << "\n--- Performance Benchmark ---" << std::endl;
        auto result = engine.benchmark(10, iterations);
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Iterations:    " << result.iterations << std::endl;
        std::cout << "Avg Latency:   " << result.avgLatencyMs << " ms" << std::endl;
        std::cout << "Min Latency:   " << result.minLatencyMs << " ms" << std::endl;
        std::cout << "Max Latency:   " << result.maxLatencyMs << " ms" << std::endl;
        std::cout << "P95 Latency:   " << result.p95LatencyMs << " ms" << std::endl;
        std::cout << "Throughput:    " << result.throughputFPS << " FPS" << std::endl;
    }
    
    std::cout << "\n✅ Done!" << std::endl;
    
    return 0;
}
