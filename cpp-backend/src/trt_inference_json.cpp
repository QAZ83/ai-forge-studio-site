/**
 * AI Forge Studio - TensorRT Inference JSON Output
 * Real-time AI inference metrics on RTX 5090 with CUDA 13.0 + TensorRT 10.14
 * 
 * Author: M.3R3 | AI Forge OPS
 * 
 * Build: cmake --build build --config Release --target trt_inference_json
 * Run:   build/bin/Release/trt_inference_json.exe [options]
 * 
 * Modes:
 *   --cuda                    : Run real CUDA compute benchmark
 *   --mock                    : Run simulated inference
 *   --tensorrt <engine_path>  : Load and run real TensorRT engine
 *   --onnx <onnx_path>        : Convert ONNX to engine and run (saves .engine)
 * 
 * Options:
 *   --warmup <N>              : Number of warmup iterations (default: 10)
 *   --runs <N>                : Number of benchmark iterations (default: 100)
 *   --fp16                    : Use FP16 precision when building from ONNX
 *   --int8                    : Use INT8 precision when building from ONNX
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <thread>
#include <algorithm>
#include <numeric>
#include <memory>

// CUDA headers
#include <cuda_runtime.h>

// NVML for device info
#include <nvml.h>

// TensorRT headers (conditional)
#ifdef HAS_TENSORRT
    #include <NvInfer.h>
    #include <NvInferRuntime.h>
    #include <NvOnnxParser.h>
    #define TRT_AVAILABLE 1
    
    // TensorRT version info
    #define TRT_VERSION_STRING std::to_string(NV_TENSORRT_MAJOR) + "." + \
                               std::to_string(NV_TENSORRT_MINOR) + "." + \
                               std::to_string(NV_TENSORRT_PATCH)
#else
    #define TRT_AVAILABLE 0
    #define TRT_VERSION_STRING "N/A"
#endif

// ============================================================================
// Utility: JSON output helper
// ============================================================================
class JsonBuilder {
    std::ostringstream ss;
    bool first = true;
    
public:
    void start() { ss << "{"; first = true; }
    void end() { ss << "}"; }
    
    template<typename T>
    void add(const std::string& key, T value) {
        if (!first) ss << ",";
        first = false;
        ss << "\"" << key << "\":" << value;
    }
    
    void addString(const std::string& key, const std::string& value) {
        if (!first) ss << ",";
        first = false;
        ss << "\"" << key << "\":\"" << value << "\"";
    }
    
    void addBool(const std::string& key, bool value) {
        if (!first) ss << ",";
        first = false;
        ss << "\"" << key << "\":" << (value ? "true" : "false");
    }
    
    void addRaw(const std::string& key, const std::string& rawJson) {
        if (!first) ss << ",";
        first = false;
        ss << "\"" << key << "\":" << rawJson;
    }
    
    std::string build() { return ss.str(); }
};

// ============================================================================
// GPU Info from NVML
// ============================================================================
struct GPUInfo {
    std::string name = "Unknown GPU";
    std::string driver = "N/A";
    std::string cudaVersion = "N/A";
    std::string computeCapability = "N/A";
    int smCount = 0;
    int cudaCores = 0;
    size_t totalMemoryMB = 0;
    size_t usedMemoryMB = 0;
    size_t freeMemoryMB = 0;
    unsigned int temperature = 0;
    unsigned int powerDraw = 0;
    unsigned int powerLimit = 0;
    unsigned int gpuClock = 0;
    unsigned int memClock = 0;
    unsigned int gpuUtil = 0;
    unsigned int memUtil = 0;
    
    bool init() {
        if (nvmlInit() != NVML_SUCCESS) return false;
        
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) {
            nvmlShutdown();
            return false;
        }
        
        // GPU name
        char nameBuf[256];
        nvmlDeviceGetName(device, nameBuf, sizeof(nameBuf));
        name = nameBuf;
        
        // Driver version
        char driverBuf[80];
        nvmlSystemGetDriverVersion(driverBuf, sizeof(driverBuf));
        driver = driverBuf;
        
        // Memory info
        nvmlMemory_t memInfo;
        nvmlDeviceGetMemoryInfo(device, &memInfo);
        totalMemoryMB = memInfo.total / (1024 * 1024);
        usedMemoryMB = memInfo.used / (1024 * 1024);
        freeMemoryMB = memInfo.free / (1024 * 1024);
        
        // Temperature
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
        
        // Power
        nvmlDeviceGetPowerUsage(device, &powerDraw);
        powerDraw /= 1000; // mW to W
        nvmlDeviceGetPowerManagementLimit(device, &powerLimit);
        powerLimit /= 1000;
        
        // Clocks
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &gpuClock);
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &memClock);
        
        // Utilization
        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
            gpuUtil = util.gpu;
            memUtil = util.memory;
        }
        
        // CUDA info
        int cudaVer = 0;
        cudaRuntimeGetVersion(&cudaVer);
        int major = cudaVer / 1000;
        int minor = (cudaVer % 1000) / 10;
        std::ostringstream vs;
        vs << major << "." << minor;
        cudaVersion = vs.str();
        
        // SM count and compute capability from CUDA
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            smCount = prop.multiProcessorCount;
            std::ostringstream cc;
            cc << prop.major << "." << prop.minor;
            computeCapability = cc.str();
            
            // CUDA cores per SM based on architecture
            int coresPerSM = 128; // Blackwell/Ada default
            if (prop.major == 12) coresPerSM = 128;      // Blackwell
            else if (prop.major == 8 && prop.minor == 9) coresPerSM = 128; // Ada
            else if (prop.major == 8 && prop.minor == 6) coresPerSM = 128; // Ampere GA102
            else if (prop.major == 8 && prop.minor == 0) coresPerSM = 64;  // Ampere A100
            else if (prop.major == 7) coresPerSM = 64;   // Volta/Turing
            
            cudaCores = smCount * coresPerSM;
        }
        
        nvmlShutdown();
        return true;
    }
};

// ============================================================================
// Inference Result
// ============================================================================
struct InferenceResult {
    bool success = false;
    std::string error;
    std::string mode;           // "tensorrt", "cuda", "mock", "onnx"
    std::string modelName;
    std::string modelPath;
    std::string precision;
    std::string tensorrtVersion;
    int batchSize = 1;
    int inputWidth = 224;
    int inputHeight = 224;
    int inputChannels = 3;
    int outputClasses = 1000;
    double latencyMs = 0;
    double minLatencyMs = 0;
    double maxLatencyMs = 0;
    double p95LatencyMs = 0;
    double throughputFPS = 0;
    double inferenceMemoryMB = 0;
    int warmupRuns = 10;
    int benchmarkRuns = 100;
    GPUInfo gpu;
};

// ============================================================================
// TensorRT Logger
// ============================================================================
#if TRT_AVAILABLE
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only print warnings and errors to stderr
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

// ============================================================================
// TensorRT Engine Wrapper
// ============================================================================
class TensorRTEngine {
    TRTLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    std::vector<void*> deviceBuffers;
    std::vector<size_t> bufferSizes;
    std::vector<std::string> tensorNames;
    std::vector<bool> tensorIsInput;
    
    size_t inputSize = 0;
    size_t outputSize = 0;
    
    // Input dimensions
    int batchSize = 1;
    int inputC = 3, inputH = 224, inputW = 224;
    int outputClasses = 1000;
    std::string precision = "FP32";
    
public:
    ~TensorRTEngine() {
        for (void* buf : deviceBuffers) {
            if (buf) cudaFree(buf);
        }
    }
    
    bool loadEngine(const std::string& enginePath) {
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) {
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();
        
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) return false;
        
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        if (!engine) return false;
        
        context.reset(engine->createExecutionContext());
        if (!context) return false;
        
        return allocateBuffers();
    }
    
    bool buildFromONNX(const std::string& onnxPath, const std::string& savePath, bool useFP16, bool useINT8) {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
        if (!builder) return false;
        
        const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(explicitBatch));
        if (!network) return false;
        
        auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, logger));
        if (!parser) return false;
        
        if (!parser->parseFromFile(onnxPath.c_str(), 
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            return false;
        }
        
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) return false;
        
        // Set workspace size (4 GB)
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ULL << 30);
        
        // Set precision
        if (useFP16 && builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            precision = "FP16";
        }
        if (useINT8 && builder->platformHasFastInt8()) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            precision = "INT8";
        }
        
        auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *config));
        if (!serializedEngine) return false;
        
        // Save engine
        if (!savePath.empty()) {
            std::ofstream outFile(savePath, std::ios::binary);
            outFile.write(static_cast<const char*>(serializedEngine->data()), 
                         serializedEngine->size());
        }
        
        runtime.reset(nvinfer1::createInferRuntime(logger));
        engine.reset(runtime->deserializeCudaEngine(serializedEngine->data(), 
                                                     serializedEngine->size()));
        context.reset(engine->createExecutionContext());
        
        return allocateBuffers();
    }
    
    bool allocateBuffers() {
        int numTensors = engine->getNbIOTensors();
        deviceBuffers.resize(numTensors, nullptr);
        bufferSizes.resize(numTensors);
        tensorNames.resize(numTensors);
        tensorIsInput.resize(numTensors);
        
        for (int i = 0; i < numTensors; i++) {
            const char* name = engine->getIOTensorName(i);
            tensorNames[i] = name;
            
            nvinfer1::Dims dims = engine->getTensorShape(name);
            nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
            tensorIsInput[i] = (mode == nvinfer1::TensorIOMode::kINPUT);
            
            size_t size = 1;
            for (int j = 0; j < dims.nbDims; j++) {
                size *= dims.d[j];
            }
            
            // Get data type size
            nvinfer1::DataType dtype = engine->getTensorDataType(name);
            size_t elemSize = sizeof(float);
            if (dtype == nvinfer1::DataType::kHALF) elemSize = 2;
            else if (dtype == nvinfer1::DataType::kINT8) elemSize = 1;
            else if (dtype == nvinfer1::DataType::kINT32) elemSize = 4;
            
            size_t bytes = size * elemSize;
            bufferSizes[i] = bytes;
            
            cudaError_t err = cudaMalloc(&deviceBuffers[i], bytes);
            if (err != cudaSuccess) return false;
            
            // Set tensor address for context
            context->setTensorAddress(name, deviceBuffers[i]);
            
            if (tensorIsInput[i]) {
                inputSize += bytes;
                // Try to extract input dimensions
                if (dims.nbDims >= 4) {
                    batchSize = dims.d[0];
                    inputC = dims.d[1];
                    inputH = dims.d[2];
                    inputW = dims.d[3];
                }
            } else {
                outputSize += bytes;
                if (dims.nbDims >= 2) {
                    outputClasses = dims.d[dims.nbDims - 1];
                }
            }
        }
        
        return true;
    }
    
    InferenceResult benchmark(int warmupRuns, int benchmarkRuns) {
        InferenceResult result;
        result.mode = "tensorrt";
        result.tensorrtVersion = TRT_VERSION_STRING;
        result.precision = precision;
        result.batchSize = batchSize;
        result.inputWidth = inputW;
        result.inputHeight = inputH;
        result.inputChannels = inputC;
        result.outputClasses = outputClasses;
        result.warmupRuns = warmupRuns;
        result.benchmarkRuns = benchmarkRuns;
        result.inferenceMemoryMB = (inputSize + outputSize) / (1024.0 * 1024.0);
        
        // Initialize input with random data
        for (int i = 0; i < tensorNames.size(); i++) {
            if (tensorIsInput[i]) {
                std::vector<float> hostData(bufferSizes[i] / sizeof(float), 0.5f);
                cudaMemcpy(deviceBuffers[i], hostData.data(), bufferSizes[i], cudaMemcpyHostToDevice);
            }
        }
        
        // Warmup
        for (int i = 0; i < warmupRuns; i++) {
            context->enqueueV3(0);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> latencies;
        latencies.reserve(benchmarkRuns);
        
        for (int i = 0; i < benchmarkRuns; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            context->enqueueV3(0);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(ms);
        }
        
        // Calculate statistics
        std::sort(latencies.begin(), latencies.end());
        
        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        result.latencyMs = sum / latencies.size();
        result.minLatencyMs = latencies.front();
        result.maxLatencyMs = latencies.back();
        result.p95LatencyMs = latencies[static_cast<size_t>(latencies.size() * 0.95)];
        result.throughputFPS = 1000.0 / result.latencyMs;
        result.success = true;
        
        return result;
    }
};
#endif // TRT_AVAILABLE

// ============================================================================
// CUDA Compute Kernel for benchmarking
// ============================================================================
__global__ void matrixMulKernel(float* C, const float* A, const float* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

InferenceResult runCudaBenchmark(int warmupRuns = 10, int benchmarkRuns = 100) {
    InferenceResult result;
    result.mode = "cuda";
    result.modelName = "cuda-matmul-benchmark";
    result.precision = "FP32";
    result.batchSize = 1;
    result.inputWidth = 1024;
    result.inputHeight = 1024;
    result.inputChannels = 3;
    result.outputClasses = 0;
    result.warmupRuns = warmupRuns;
    result.benchmarkRuns = benchmarkRuns;
    result.tensorrtVersion = TRT_VERSION_STRING;
    
    const int N = 1024;
    const size_t bytes = N * N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    cudaError_t err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) {
        result.error = "CUDA malloc failed: " + std::string(cudaGetErrorString(err));
        return result;
    }
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    std::vector<float> h_A(N * N, 0.5f);
    std::vector<float> h_B(N * N, 0.5f);
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);
    
    // Warmup
    for (int i = 0; i < warmupRuns; i++) {
        matrixMulKernel<<<gridSize, blockSize>>>(d_C, d_A, d_B, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    std::vector<double> latencies;
    for (int i = 0; i < benchmarkRuns; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        matrixMulKernel<<<gridSize, blockSize>>>(d_C, d_A, d_B, N);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    
    std::sort(latencies.begin(), latencies.end());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    
    result.latencyMs = sum / latencies.size();
    result.minLatencyMs = latencies.front();
    result.maxLatencyMs = latencies.back();
    result.p95LatencyMs = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    result.throughputFPS = 1000.0 / result.latencyMs;
    result.inferenceMemoryMB = (bytes * 3) / (1024.0 * 1024.0);
    result.success = true;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return result;
}

// ============================================================================
// Mock Inference (simulated TensorRT-like performance)
// ============================================================================
InferenceResult runMockInference(const std::string& modelName = "") {
    InferenceResult result;
    result.mode = "mock";
    result.success = true;
    result.modelName = modelName.empty() ? "resnet50-fp16-simulated" : modelName;
    result.precision = "FP16";
    result.batchSize = 1;
    result.inputWidth = 224;
    result.inputHeight = 224;
    result.inputChannels = 3;
    result.outputClasses = 1000;
    result.warmupRuns = 10;
    result.benchmarkRuns = 100;
    result.tensorrtVersion = TRT_VERSION_STRING;
    
    // Simulate realistic RTX 5090 TensorRT performance
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> latencyDist(0.18, 0.32);
    
    result.latencyMs = latencyDist(gen);
    result.minLatencyMs = result.latencyMs * 0.9;
    result.maxLatencyMs = result.latencyMs * 1.2;
    result.p95LatencyMs = result.latencyMs * 1.1;
    result.throughputFPS = 1000.0 / result.latencyMs;
    result.inferenceMemoryMB = 98.5;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    
    return result;
}

// ============================================================================
// Get filename from path
// ============================================================================
std::string getFilename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

// ============================================================================
// Convert Windows path to JSON-safe path (backslash -> forward slash)
// ============================================================================
std::string toJsonSafePath(const std::string& path) {
    std::string result = path;
    std::replace(result.begin(), result.end(), '\\', '/');
    return result;
}

// ============================================================================
// Output JSON result
// ============================================================================
void outputJson(const InferenceResult& result) {
    JsonBuilder json;
    json.start();
    
    json.addBool("success", result.success);
    if (!result.error.empty()) {
        json.addString("error", result.error);
    }
    
    json.addString("mode", result.mode);
    json.addString("model", result.modelName);
    if (!result.modelPath.empty()) {
        // Convert backslashes to forward slashes for valid JSON
        json.addString("modelPath", toJsonSafePath(result.modelPath));
    }
    json.addString("precision", result.precision);
    json.addString("tensorrtVersion", result.tensorrtVersion);
    json.add("batchSize", result.batchSize);
    
    // Input shape
    std::ostringstream inputShape;
    inputShape << "[" << result.batchSize << "," << result.inputChannels << ","
               << result.inputHeight << "," << result.inputWidth << "]";
    json.addRaw("inputShape", inputShape.str());
    
    if (result.outputClasses > 0) {
        json.add("outputClasses", result.outputClasses);
    }
    
    // Performance metrics
    json.add("latencyMs", std::round(result.latencyMs * 1000) / 1000.0);
    json.add("minLatencyMs", std::round(result.minLatencyMs * 1000) / 1000.0);
    json.add("maxLatencyMs", std::round(result.maxLatencyMs * 1000) / 1000.0);
    json.add("p95LatencyMs", std::round(result.p95LatencyMs * 1000) / 1000.0);
    json.add("throughputFPS", std::round(result.throughputFPS * 10) / 10.0);
    json.add("inferenceMemoryMB", std::round(result.inferenceMemoryMB * 10) / 10.0);
    json.add("warmupRuns", result.warmupRuns);
    json.add("benchmarkRuns", result.benchmarkRuns);
    
    // GPU info
    json.addString("device", result.gpu.name);
    json.addString("driver", result.gpu.driver);
    json.addString("cudaVersion", result.gpu.cudaVersion);
    json.addString("computeCapability", result.gpu.computeCapability);
    json.add("smCount", result.gpu.smCount);
    json.add("cudaCores", result.gpu.cudaCores);
    json.add("vramTotalMB", (int)result.gpu.totalMemoryMB);
    json.add("vramUsedMB", (int)result.gpu.usedMemoryMB);
    json.add("vramFreeMB", (int)result.gpu.freeMemoryMB);
    json.add("temperatureC", result.gpu.temperature);
    json.add("powerDrawW", result.gpu.powerDraw);
    json.add("powerLimitW", result.gpu.powerLimit);
    json.add("gpuClockMHz", result.gpu.gpuClock);
    json.add("memClockMHz", result.gpu.memClock);
    json.add("gpuUtilization", result.gpu.gpuUtil);
    
    json.end();
    std::cout << json.build() << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    InferenceResult result;
    
    // Get GPU info first
    if (!result.gpu.init()) {
        result.success = false;
        result.error = "Failed to initialize GPU via NVML";
        result.mode = "error";
        outputJson(result);
        return 1;
    }
    
    std::string enginePath;
    std::string onnxPath;
    bool useCudaBenchmark = false;
    bool useMock = false;
    bool useFP16 = true;  // Default to FP16 for best performance
    bool useINT8 = false;
    int warmupRuns = 10;
    int benchmarkRuns = 100;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--cuda" || arg == "-c") {
            useCudaBenchmark = true;
        } else if (arg == "--mock" || arg == "-m") {
            useMock = true;
        } else if (arg == "--tensorrt" || arg == "-t") {
            if (i + 1 < argc) enginePath = argv[++i];
        } else if (arg == "--onnx" || arg == "-o") {
            if (i + 1 < argc) onnxPath = argv[++i];
        } else if (arg == "--fp16") {
            useFP16 = true;
        } else if (arg == "--int8") {
            useINT8 = true;
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmupRuns = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            benchmarkRuns = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "AI Forge Studio - TensorRT Inference JSON\n"
                      << "Usage: trt_inference_json [options]\n\n"
                      << "Modes:\n"
                      << "  --cuda, -c               Run CUDA compute benchmark\n"
                      << "  --mock, -m               Run simulated TensorRT inference\n"
                      << "  --tensorrt, -t <engine>  Load and run TensorRT .engine file\n"
                      << "  --onnx, -o <model>       Convert ONNX to engine and run\n\n"
                      << "Options:\n"
                      << "  --fp16                   Use FP16 precision (default)\n"
                      << "  --int8                   Use INT8 precision\n"
                      << "  --warmup <N>             Warmup iterations (default: 10)\n"
                      << "  --runs <N>               Benchmark iterations (default: 100)\n"
                      << "  --help, -h               Show this help\n\n"
                      << "TensorRT Available: " << (TRT_AVAILABLE ? "YES" : "NO") << "\n";
            return 0;
        } else if (arg[0] != '-') {
            // Positional argument - treat as engine path
            enginePath = arg;
        }
    }
    
    // Store GPU info
    GPUInfo gpuInfo = result.gpu;
    
    if (useCudaBenchmark) {
        // Run CUDA kernel benchmark
        result = runCudaBenchmark(warmupRuns, benchmarkRuns);
    }
#if TRT_AVAILABLE
    else if (!enginePath.empty()) {
        // Load and run TensorRT engine
        TensorRTEngine engine;
        if (engine.loadEngine(enginePath)) {
            result = engine.benchmark(warmupRuns, benchmarkRuns);
            result.modelName = getFilename(enginePath);
            result.modelPath = enginePath;
        } else {
            result.success = false;
            result.mode = "tensorrt";
            result.error = "Failed to load TensorRT engine: " + enginePath;
        }
    }
    else if (!onnxPath.empty()) {
        // Build from ONNX and run
        TensorRTEngine engine;
        std::string savePath = onnxPath.substr(0, onnxPath.rfind('.')) + ".engine";
        
        if (engine.buildFromONNX(onnxPath, savePath, useFP16, useINT8)) {
            result = engine.benchmark(warmupRuns, benchmarkRuns);
            result.mode = "tensorrt";
            result.modelName = getFilename(onnxPath);
            result.modelPath = savePath;
        } else {
            result.success = false;
            result.mode = "tensorrt";
            result.error = "Failed to build engine from ONNX: " + onnxPath;
        }
    }
#else
    else if (!enginePath.empty() || !onnxPath.empty()) {
        // TensorRT not available
        result.success = false;
        result.mode = "tensorrt";
        result.error = "TensorRT not available. Rebuild with HAS_TENSORRT=1";
    }
#endif
    else if (useMock) {
        // Run mock inference
        result = runMockInference();
    }
    else {
        // Default: mock mode
        result = runMockInference();
    }
    
    // Refresh GPU info after benchmark
    gpuInfo.init();
    result.gpu = gpuInfo;
    
    outputJson(result);
    return result.success ? 0 : 1;
}
