/**
 * AI Forge Studio - GPU Info JSON Output
 * Outputs GPU metrics as JSON for Electron integration
 * 
 * Build: cmake --build build --config Release --target gpu_info_json
 * Run:   ./build/bin/Release/gpu_info_json.exe
 * 
 * Output: Pure JSON to stdout (no extra text)
 */

#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include <nvml.h>

// Simple JSON builder (no external dependencies)
class JsonBuilder {
public:
    void startObject() { ss_ << "{"; first_ = true; }
    void endObject() { ss_ << "}"; }
    void startArray() { ss_ << "["; first_ = true; }
    void endArray() { ss_ << "]"; }
    
    void key(const std::string& k) {
        if (!first_) ss_ << ",";
        first_ = false;
        ss_ << "\"" << k << "\":";
    }
    
    void value(const std::string& v) { ss_ << "\"" << escape(v) << "\""; }
    void value(int v) { ss_ << v; }
    void value(unsigned int v) { ss_ << v; }
    void value(long long v) { ss_ << v; }
    void value(unsigned long long v) { ss_ << v; }
    void value(double v) { ss_ << std::fixed << v; }
    void value(bool v) { ss_ << (v ? "true" : "false"); }
    void nullValue() { ss_ << "null"; }
    
    void keyValue(const std::string& k, const std::string& v) { key(k); value(v); }
    void keyValue(const std::string& k, int v) { key(k); value(v); }
    void keyValue(const std::string& k, unsigned int v) { key(k); value(v); }
    void keyValue(const std::string& k, long long v) { key(k); value(v); }
    void keyValue(const std::string& k, unsigned long long v) { key(k); value(v); }
    void keyValue(const std::string& k, double v) { key(k); value(v); }
    void keyValue(const std::string& k, bool v) { key(k); value(v); }
    
    std::string str() const { return ss_.str(); }
    
private:
    std::stringstream ss_;
    bool first_ = true;
    
    std::string escape(const std::string& s) {
        std::string result;
        for (char c : s) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c;
            }
        }
        return result;
    }
};

// Get GPU info using NVML
struct GPUMetrics {
    bool valid = false;
    std::string errorMessage;
    
    // Device info
    std::string name;
    std::string driverVersion;
    std::string cudaVersion;
    int computeMajor = 0;
    int computeMinor = 0;
    
    // Hardware
    int smCount = 0;
    int cudaCores = 0;
    int memoryBusWidth = 0;
    
    // Memory (MB)
    unsigned long long memoryTotalMB = 0;
    unsigned long long memoryUsedMB = 0;
    unsigned long long memoryFreeMB = 0;
    
    // Utilization (%)
    unsigned int gpuUtilization = 0;
    unsigned int memoryUtilization = 0;
    
    // Clocks (MHz)
    unsigned int gpuClockMHz = 0;
    unsigned int memoryClockMHz = 0;
    unsigned int maxGpuClockMHz = 0;
    unsigned int maxMemoryClockMHz = 0;
    
    // Power & Thermal
    unsigned int temperatureC = 0;
    unsigned int powerDrawW = 0;
    unsigned int powerLimitW = 0;
    
    // Fan
    unsigned int fanSpeedPercent = 0;
};

GPUMetrics getGPUMetrics() {
    GPUMetrics m;
    
    // Initialize NVML
    nvmlReturn_t nvmlResult = nvmlInit();
    if (nvmlResult != NVML_SUCCESS) {
        m.errorMessage = std::string("NVML init failed: ") + nvmlErrorString(nvmlResult);
        return m;
    }
    
    // Get device handle
    nvmlDevice_t device;
    nvmlResult = nvmlDeviceGetHandleByIndex(0, &device);
    if (nvmlResult != NVML_SUCCESS) {
        m.errorMessage = std::string("Failed to get device: ") + nvmlErrorString(nvmlResult);
        nvmlShutdown();
        return m;
    }
    
    // Device name
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    if (nvmlDeviceGetName(device, name, sizeof(name)) == NVML_SUCCESS) {
        m.name = name;
    }
    
    // Driver version
    char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    if (nvmlSystemGetDriverVersion(driver, sizeof(driver)) == NVML_SUCCESS) {
        m.driverVersion = driver;
    }
    
    // CUDA version from NVML
    int cudaVersion;
    if (nvmlSystemGetCudaDriverVersion(&cudaVersion) == NVML_SUCCESS) {
        int major = cudaVersion / 1000;
        int minor = (cudaVersion % 1000) / 10;
        m.cudaVersion = std::to_string(major) + "." + std::to_string(minor);
    }
    
    // Memory info
    nvmlMemory_t memory;
    if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
        m.memoryTotalMB = memory.total / (1024 * 1024);
        m.memoryUsedMB = memory.used / (1024 * 1024);
        m.memoryFreeMB = memory.free / (1024 * 1024);
    }
    
    // Utilization
    nvmlUtilization_t util;
    if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
        m.gpuUtilization = util.gpu;
        m.memoryUtilization = util.memory;
    }
    
    // Clocks
    unsigned int clock;
    if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock) == NVML_SUCCESS) {
        m.gpuClockMHz = clock;
    }
    if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock) == NVML_SUCCESS) {
        m.memoryClockMHz = clock;
    }
    if (nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &clock) == NVML_SUCCESS) {
        m.maxGpuClockMHz = clock;
    }
    if (nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &clock) == NVML_SUCCESS) {
        m.maxMemoryClockMHz = clock;
    }
    
    // Temperature
    unsigned int temp;
    if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
        m.temperatureC = temp;
    }
    
    // Power
    unsigned int power;
    if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
        m.powerDrawW = power / 1000; // mW to W
    }
    unsigned int powerLimit;
    if (nvmlDeviceGetPowerManagementLimit(device, &powerLimit) == NVML_SUCCESS) {
        m.powerLimitW = powerLimit / 1000;
    }
    
    // Fan speed
    unsigned int fan;
    if (nvmlDeviceGetFanSpeed(device, &fan) == NVML_SUCCESS) {
        m.fanSpeedPercent = fan;
    }
    
    // Get CUDA device properties for additional info
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        m.computeMajor = prop.major;
        m.computeMinor = prop.minor;
        m.smCount = prop.multiProcessorCount;
        m.memoryBusWidth = prop.memoryBusWidth;
        
        // Estimate CUDA cores based on architecture
        int coresPerSM = 128; // Default for modern GPUs
        if (prop.major == 10) coresPerSM = 128;      // Blackwell
        else if (prop.major == 8 && prop.minor == 9) coresPerSM = 128; // Ada
        else if (prop.major == 8 && prop.minor == 6) coresPerSM = 128; // Ampere
        else if (prop.major == 7 && prop.minor == 5) coresPerSM = 64;  // Turing
        
        m.cudaCores = prop.multiProcessorCount * coresPerSM;
    }
    
    nvmlShutdown();
    m.valid = true;
    return m;
}

std::string buildJSON(const GPUMetrics& m) {
    JsonBuilder json;
    
    json.startObject();
    
    if (!m.valid) {
        json.keyValue("success", false);
        json.keyValue("error", m.errorMessage);
    } else {
        json.keyValue("success", true);
        
        // Device info
        json.keyValue("gpu", m.name);
        json.keyValue("driverVersion", m.driverVersion);
        json.keyValue("cudaVersion", m.cudaVersion);
        json.keyValue("computeCapability", 
            std::to_string(m.computeMajor) + "." + std::to_string(m.computeMinor));
        
        // Hardware specs
        json.key("hardware");
        json.startObject();
        json.keyValue("smCount", m.smCount);
        json.keyValue("cudaCores", m.cudaCores);
        json.keyValue("memoryBusWidth", m.memoryBusWidth);
        json.endObject();
        
        // Memory metrics
        json.key("memory");
        json.startObject();
        json.keyValue("totalMB", m.memoryTotalMB);
        json.keyValue("usedMB", m.memoryUsedMB);
        json.keyValue("freeMB", m.memoryFreeMB);
        json.keyValue("usagePercent", 
            static_cast<int>((m.memoryUsedMB * 100) / m.memoryTotalMB));
        json.endObject();
        
        // Utilization
        json.key("utilization");
        json.startObject();
        json.keyValue("gpuPercent", m.gpuUtilization);
        json.keyValue("memoryPercent", m.memoryUtilization);
        json.endObject();
        
        // Clocks
        json.key("clocks");
        json.startObject();
        json.keyValue("gpuMHz", m.gpuClockMHz);
        json.keyValue("memoryMHz", m.memoryClockMHz);
        json.keyValue("maxGpuMHz", m.maxGpuClockMHz);
        json.keyValue("maxMemoryMHz", m.maxMemoryClockMHz);
        json.endObject();
        
        // Power & Thermal
        json.key("power");
        json.startObject();
        json.keyValue("drawW", m.powerDrawW);
        json.keyValue("limitW", m.powerLimitW);
        json.keyValue("usagePercent", 
            m.powerLimitW > 0 ? static_cast<int>((m.powerDrawW * 100) / m.powerLimitW) : 0);
        json.endObject();
        
        json.key("thermal");
        json.startObject();
        json.keyValue("temperatureC", m.temperatureC);
        json.keyValue("fanSpeedPercent", m.fanSpeedPercent);
        json.endObject();
    }
    
    json.endObject();
    return json.str();
}

int main(int argc, char* argv[]) {
    // Check for --pretty flag
    bool pretty = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--pretty") {
            pretty = true;
        }
    }
    
    GPUMetrics metrics = getGPUMetrics();
    std::string jsonOutput = buildJSON(metrics);
    
    // Output JSON
    std::cout << jsonOutput << std::endl;
    
    return metrics.valid ? 0 : 1;
}
