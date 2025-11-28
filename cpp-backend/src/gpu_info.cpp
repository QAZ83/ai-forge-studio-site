/**
 * AI Forge Studio - GPU Info Tool
 * Displays real GPU information using CUDA + NVML
 * 
 * Build: cmake --build build --target gpu_info
 * Run:   ./build/bin/gpu_info
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>
#include <nvml.h>

#include "cuda_helper.h"
#include "gpu_types.h"

// External test function from cuda_helper.cu
extern "C" bool runSimpleKernelTest();

// =============================================================================
// NVML Helper Functions
// =============================================================================

class NVMLManager {
public:
    NVMLManager() {
        nvmlReturn_t result = nvmlInit();
        initialized_ = (result == NVML_SUCCESS);
        if (!initialized_) {
            std::cerr << "Warning: NVML initialization failed: " << nvmlErrorString(result) << std::endl;
        }
    }
    
    ~NVMLManager() {
        if (initialized_) {
            nvmlShutdown();
        }
    }
    
    bool isInitialized() const { return initialized_; }
    
    aiforge::GPUInfo getGPUInfo(int deviceIndex = 0) {
        aiforge::GPUInfo info = {};
        info.deviceId = deviceIndex;
        
        if (!initialized_) {
            return info;
        }
        
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex(deviceIndex, &device) != NVML_SUCCESS) {
            return info;
        }
        
        // Name
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        if (nvmlDeviceGetName(device, name, sizeof(name)) == NVML_SUCCESS) {
            info.name = name;
        }
        
        // Driver version
        char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        if (nvmlSystemGetDriverVersion(driver, sizeof(driver)) == NVML_SUCCESS) {
            info.driverVersion = driver;
        }
        
        // CUDA version
        int cudaVersion;
        if (nvmlSystemGetCudaDriverVersion(&cudaVersion) == NVML_SUCCESS) {
            int major = cudaVersion / 1000;
            int minor = (cudaVersion % 1000) / 10;
            info.cudaVersion = std::to_string(major) + "." + std::to_string(minor);
        }
        
        // Memory
        nvmlMemory_t memory;
        if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
            info.totalMemoryMB = memory.total / (1024 * 1024);
            info.freeMemoryMB = memory.free / (1024 * 1024);
            info.usedMemoryMB = memory.used / (1024 * 1024);
        }
        
        // Temperature
        unsigned int temp;
        if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            info.temperatureC = static_cast<int>(temp);
        }
        
        // Power
        unsigned int power;
        if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
            info.powerDrawW = power / 1000; // Convert from mW to W
        }
        
        unsigned int powerLimit;
        if (nvmlDeviceGetPowerManagementLimit(device, &powerLimit) == NVML_SUCCESS) {
            info.powerLimitW = powerLimit / 1000;
        }
        
        // Clocks
        unsigned int clock;
        if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock) == NVML_SUCCESS) {
            info.gpuClockMHz = clock;
        }
        if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock) == NVML_SUCCESS) {
            info.memoryClockMHz = clock;
        }
        if (nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &clock) == NVML_SUCCESS) {
            info.maxGpuClockMHz = clock;
        }
        if (nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &clock) == NVML_SUCCESS) {
            info.maxMemoryClockMHz = clock;
        }
        
        // Utilization
        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
            info.gpuUtilization = util.gpu;
            info.memoryUtilization = util.memory;
        }
        
        return info;
    }
    
private:
    bool initialized_ = false;
};

// =============================================================================
// Print Functions
// =============================================================================

void printHeader() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              AI FORGE STUDIO - GPU INFORMATION                   ║\n";
    std::cout << "║                    Real Hardware Data                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void printSection(const std::string& title) {
    std::cout << "\n┌─────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ " << std::left << std::setw(63) << title << " │\n";
    std::cout << "└─────────────────────────────────────────────────────────────────┘\n";
}

void printKeyValue(const std::string& key, const std::string& value) {
    std::cout << "  " << std::left << std::setw(25) << key << ": " << value << "\n";
}

void printKeyValue(const std::string& key, int value, const std::string& unit = "") {
    std::cout << "  " << std::left << std::setw(25) << key << ": " << value << " " << unit << "\n";
}

void printProgress(const std::string& label, int current, int max, const std::string& unit = "") {
    float percent = (max > 0) ? (static_cast<float>(current) / max * 100.0f) : 0.0f;
    int barWidth = 30;
    int filled = static_cast<int>(percent / 100.0f * barWidth);
    
    std::cout << "  " << std::left << std::setw(20) << label << " [";
    for (int i = 0; i < barWidth; ++i) {
        if (i < filled) std::cout << "█";
        else std::cout << "░";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << percent << "% ";
    std::cout << "(" << current << "/" << max << " " << unit << ")\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printHeader();
    
    // =========================================================================
    // CUDA Device Info (using CUDA Runtime API)
    // =========================================================================
    printSection("CUDA DEVICE INFORMATION");
    
    int deviceCount = aiforge::cuda::getDeviceCount();
    if (deviceCount == 0) {
        std::cerr << "  ❌ No CUDA-capable devices found!\n";
        return 1;
    }
    
    printKeyValue("CUDA Devices Found", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        std::cout << "\n  ═══ Device " << i << " ═══\n";
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printKeyValue("Name", prop.name);
        printKeyValue("Compute Capability", 
            std::to_string(prop.major) + "." + std::to_string(prop.minor));
        printKeyValue("SM Count", prop.multiProcessorCount);
        printKeyValue("Max Threads per SM", prop.maxThreadsPerMultiProcessor);
        printKeyValue("Max Threads per Block", prop.maxThreadsPerBlock);
        printKeyValue("Warp Size", prop.warpSize);
        // Note: clockRate removed in CUDA 13.0+, will get from NVML
        printKeyValue("Memory Bus Width", prop.memoryBusWidth, "bits");
        printKeyValue("L2 Cache Size", prop.l2CacheSize / 1024, "KB");
        printKeyValue("Total Global Memory", 
            std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB");
        printKeyValue("Shared Memory/Block", 
            std::to_string(prop.sharedMemPerBlock / 1024) + " KB");
        
        // CUDA Cores estimation based on SM count and architecture
        int coresPerSM = 128; // Blackwell SM 10.0 has 128 CUDA cores per SM
        if (prop.major == 8 && prop.minor == 9) coresPerSM = 128; // Ada
        else if (prop.major == 8 && prop.minor == 6) coresPerSM = 128; // Ampere
        else if (prop.major == 10) coresPerSM = 128; // Blackwell (estimated)
        
        int estimatedCores = prop.multiProcessorCount * coresPerSM;
        printKeyValue("CUDA Cores (est.)", estimatedCores);
    }
    
    // =========================================================================
    // NVML Real-time Metrics
    // =========================================================================
    printSection("REAL-TIME GPU METRICS (NVML)");
    
    NVMLManager nvml;
    if (nvml.isInitialized()) {
        aiforge::GPUInfo info = nvml.getGPUInfo(0);
        
        printKeyValue("GPU Name", info.name);
        printKeyValue("Driver Version", info.driverVersion);
        printKeyValue("CUDA Version", info.cudaVersion);
        
        std::cout << "\n  --- Memory ---\n";
        printProgress("VRAM Usage", info.usedMemoryMB, info.totalMemoryMB, "MB");
        printKeyValue("Free VRAM", info.freeMemoryMB, "MB");
        
        std::cout << "\n  --- Utilization ---\n";
        printProgress("GPU Utilization", info.gpuUtilization, 100, "%");
        printProgress("Memory Bandwidth", info.memoryUtilization, 100, "%");
        
        std::cout << "\n  --- Power & Thermal ---\n";
        printKeyValue("Temperature", info.temperatureC, "°C");
        printProgress("Power Draw", info.powerDrawW, info.powerLimitW, "W");
        
        std::cout << "\n  --- Clocks ---\n";
        printKeyValue("GPU Clock", std::to_string(info.gpuClockMHz) + " / " + 
                      std::to_string(info.maxGpuClockMHz) + " MHz");
        printKeyValue("Memory Clock", std::to_string(info.memoryClockMHz) + " / " + 
                      std::to_string(info.maxMemoryClockMHz) + " MHz");
    } else {
        std::cout << "  ⚠️  NVML not available - some metrics may be missing\n";
    }
    
    // =========================================================================
    // CUDA Kernel Test
    // =========================================================================
    printSection("CUDA FUNCTIONALITY TEST");
    
    std::cout << "  Running vector addition kernel test...\n";
    
    bool kernelTestPassed = runSimpleKernelTest();
    if (kernelTestPassed) {
        std::cout << "  ✅ CUDA kernel test PASSED!\n";
        std::cout << "  ✅ Your RTX 5090 is ready for AI workloads!\n";
    } else {
        std::cout << "  ❌ CUDA kernel test FAILED!\n";
        return 1;
    }
    
    // =========================================================================
    // Summary
    // =========================================================================
    printSection("SUMMARY");
    
    std::cout << "  ✅ CUDA Runtime: Working\n";
    std::cout << "  ✅ NVML Monitoring: " << (nvml.isInitialized() ? "Working" : "Not Available") << "\n";
    std::cout << "  ✅ GPU Compute: Verified\n";
    std::cout << "\n  🚀 Ready for TensorRT integration!\n";
    
    std::cout << "\n══════════════════════════════════════════════════════════════════\n\n";
    
    return 0;
}
