/**
 * AI Forge Studio - GPU Monitor Implementation
 * Author: M.3R3
 * 
 * NVML-based GPU monitoring implementation.
 */

#include "gpu_monitor.h"
#include <nvml.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "pdh.lib")
#endif

namespace aiforge {
namespace monitor {

//=============================================================================
// GPUMonitor Implementation
//=============================================================================

GPUMonitor::GPUMonitor()
    : m_initialized(false)
    , m_gpuCount(0)
    , m_monitoring(false)
    , m_intervalMs(100)
    , m_tempThreshold(85.0f)  // Alert at 85°C
    , m_memThreshold(95.0f)   // Alert at 95% memory usage
{
}

GPUMonitor::~GPUMonitor()
{
    shutdown();
}

bool GPUMonitor::initialize()
{
    if (m_initialized) {
        return true;
    }

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        m_lastError = "Failed to initialize NVML: " + 
                      std::string(nvmlErrorString(result));
        return false;
    }

    // Get GPU count
    unsigned int deviceCount = 0;
    result = nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS) {
        m_lastError = "Failed to get GPU count: " + 
                      std::string(nvmlErrorString(result));
        nvmlShutdown();
        return false;
    }

    m_gpuCount = static_cast<int>(deviceCount);
    m_devices.resize(m_gpuCount);

    // Get device handles
    for (int i = 0; i < m_gpuCount; ++i) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            m_lastError = "Failed to get device handle for GPU " + 
                          std::to_string(i) + ": " +
                          std::string(nvmlErrorString(result));
            nvmlShutdown();
            return false;
        }
        m_devices[i] = device;
    }

    std::cout << "[GPUMonitor] Initialized with " << m_gpuCount << " GPU(s)\n";
    m_initialized = true;
    return true;
}

void GPUMonitor::shutdown()
{
    stopMonitoring();
    
    if (m_initialized) {
        nvmlShutdown();
        m_initialized = false;
        m_devices.clear();
        m_gpuCount = 0;
    }
}

GPUMetrics GPUMonitor::getMetrics(int deviceIndex)
{
    GPUMetrics metrics = {};
    metrics.deviceIndex = deviceIndex;
    metrics.timestamp = std::chrono::steady_clock::now();

    if (!m_initialized || deviceIndex < 0 || deviceIndex >= m_gpuCount) {
        metrics.status = "Device not available";
        metrics.isHealthy = false;
        return metrics;
    }

    nvmlDevice_t device = static_cast<nvmlDevice_t>(m_devices[deviceIndex]);
    nvmlReturn_t result;

    // Device name
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        metrics.name = name;
    }

    // Driver version
    char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    result = nvmlSystemGetDriverVersion(driverVersion, 
                                        NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        metrics.driverVersion = driverVersion;
    }

    // CUDA version
    int cudaVersion = 0;
    result = nvmlSystemGetCudaDriverVersion(&cudaVersion);
    if (result == NVML_SUCCESS) {
        int major = cudaVersion / 1000;
        int minor = (cudaVersion % 1000) / 10;
        metrics.cudaVersion = std::to_string(major) + "." + std::to_string(minor);
    }

    // GPU utilization
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result == NVML_SUCCESS) {
        metrics.gpuUsagePercent = static_cast<float>(utilization.gpu);
        metrics.memoryUsagePercent = static_cast<float>(utilization.memory);
    }

    // Memory info
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result == NVML_SUCCESS) {
        metrics.memoryTotalMB = memory.total / (1024 * 1024);
        metrics.memoryUsedMB = memory.used / (1024 * 1024);
        metrics.memoryFreeMB = memory.free / (1024 * 1024);
        metrics.memoryUsagePercent = 
            static_cast<float>(memory.used) / static_cast<float>(memory.total) * 100.0f;
    }

    // Temperature
    unsigned int temp;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (result == NVML_SUCCESS) {
        metrics.temperatureCelsius = static_cast<float>(temp);
    }

    // Power draw
    unsigned int power;
    result = nvmlDeviceGetPowerUsage(device, &power);
    if (result == NVML_SUCCESS) {
        metrics.powerDrawWatts = static_cast<float>(power) / 1000.0f; // mW to W
    }

    // Power limit
    unsigned int powerLimit;
    result = nvmlDeviceGetPowerManagementLimit(device, &powerLimit);
    if (result == NVML_SUCCESS) {
        metrics.powerLimitWatts = static_cast<float>(powerLimit) / 1000.0f;
    }

    // Clock speeds
    unsigned int clockSpeed;
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clockSpeed);
    if (result == NVML_SUCCESS) {
        metrics.clockSpeedMHz = static_cast<int>(clockSpeed);
    }

    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clockSpeed);
    if (result == NVML_SUCCESS) {
        metrics.memoryClockMHz = static_cast<int>(clockSpeed);
    }

    // Fan speed
    unsigned int fanSpeed;
    result = nvmlDeviceGetFanSpeed(device, &fanSpeed);
    if (result == NVML_SUCCESS) {
        metrics.fanSpeedPercent = static_cast<int>(fanSpeed);
    }

    // PCIe throughput
    unsigned int txBytes, rxBytes;
    result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &txBytes);
    if (result == NVML_SUCCESS) {
        metrics.pcieTxMBps = static_cast<int>(txBytes / 1024); // KB to MB
    }
    
    result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &rxBytes);
    if (result == NVML_SUCCESS) {
        metrics.pcieRxMBps = static_cast<int>(rxBytes / 1024);
    }

    // PCIe link info
    unsigned int pcieLinkGen, pcieLinkWidth;
    result = nvmlDeviceGetCurrPcieLinkGeneration(device, &pcieLinkGen);
    if (result == NVML_SUCCESS) {
        metrics.pcieGen = static_cast<int>(pcieLinkGen);
    }
    
    result = nvmlDeviceGetCurrPcieLinkWidth(device, &pcieLinkWidth);
    if (result == NVML_SUCCESS) {
        metrics.pcieLanes = static_cast<int>(pcieLinkWidth);
    }

    // SM count
    nvmlDeviceGetNumGpuCores(device, reinterpret_cast<unsigned int*>(&metrics.cudaCores));
    
    // Estimate tensor cores based on architecture (RTX 5090 = Blackwell)
    // Blackwell has ~128 tensor cores per SM
    int smCount = 0;
    nvmlDeviceGetNumGpuCores(device, reinterpret_cast<unsigned int*>(&smCount));
    metrics.smCount = smCount / 128; // Approximate SM count
    metrics.tensorCores = metrics.smCount * 4; // 4th gen tensor cores per SM

    // Health check
    metrics.isHealthy = (metrics.temperatureCelsius < m_tempThreshold) &&
                        (metrics.memoryUsagePercent < m_memThreshold);
    
    metrics.status = metrics.isHealthy ? "OK" : "Warning";

    return metrics;
}

std::vector<GPUMetrics> GPUMonitor::getAllMetrics()
{
    std::vector<GPUMetrics> allMetrics;
    allMetrics.reserve(m_gpuCount);
    
    for (int i = 0; i < m_gpuCount; ++i) {
        allMetrics.push_back(getMetrics(i));
    }
    
    return allMetrics;
}

void GPUMonitor::startMonitoring(int intervalMs)
{
    if (m_monitoring.load()) {
        return;
    }

    m_intervalMs = intervalMs;
    m_monitoring.store(true);
    m_monitorThread = std::thread(&GPUMonitor::monitoringLoop, this);
    
    std::cout << "[GPUMonitor] Started monitoring at " << intervalMs << "ms interval\n";
}

void GPUMonitor::stopMonitoring()
{
    if (!m_monitoring.load()) {
        return;
    }

    m_monitoring.store(false);
    
    if (m_monitorThread.joinable()) {
        m_monitorThread.join();
    }
    
    std::cout << "[GPUMonitor] Stopped monitoring\n";
}

void GPUMonitor::setMetricsCallback(MetricsCallback callback)
{
    std::lock_guard<std::mutex> lock(m_callbackMutex);
    m_metricsCallback = callback;
}

void GPUMonitor::setAlertCallback(AlertCallback callback)
{
    std::lock_guard<std::mutex> lock(m_callbackMutex);
    m_alertCallback = callback;
}

void GPUMonitor::monitoringLoop()
{
    while (m_monitoring.load()) {
        for (int i = 0; i < m_gpuCount; ++i) {
            GPUMetrics metrics = getMetrics(i);
            
            // Fire callback
            {
                std::lock_guard<std::mutex> lock(m_callbackMutex);
                if (m_metricsCallback) {
                    m_metricsCallback(metrics);
                }
            }
            
            // Check for alerts
            checkAlerts(metrics);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(m_intervalMs));
    }
}

void GPUMonitor::checkAlerts(const GPUMetrics& metrics)
{
    std::lock_guard<std::mutex> lock(m_callbackMutex);
    
    if (!m_alertCallback) {
        return;
    }

    if (metrics.temperatureCelsius >= m_tempThreshold) {
        m_alertCallback("GPU " + std::to_string(metrics.deviceIndex) + 
                       " temperature critical: " + 
                       std::to_string(static_cast<int>(metrics.temperatureCelsius)) + "°C");
    }

    if (metrics.memoryUsagePercent >= m_memThreshold) {
        m_alertCallback("GPU " + std::to_string(metrics.deviceIndex) + 
                       " memory usage critical: " + 
                       std::to_string(static_cast<int>(metrics.memoryUsagePercent)) + "%");
    }

    if (metrics.powerDrawWatts > metrics.powerLimitWatts * 0.95f) {
        m_alertCallback("GPU " + std::to_string(metrics.deviceIndex) + 
                       " approaching power limit: " + 
                       std::to_string(static_cast<int>(metrics.powerDrawWatts)) + "W");
    }
}

SystemMetrics GPUMonitor::getSystemMetrics()
{
    SystemMetrics metrics = {};
    
#ifdef _WIN32
    // CPU usage (simplified)
    FILETIME idleTime, kernelTime, userTime;
    if (GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
        static ULONGLONG lastIdle = 0, lastKernel = 0, lastUser = 0;
        
        ULONGLONG idle = (static_cast<ULONGLONG>(idleTime.dwHighDateTime) << 32) | 
                         idleTime.dwLowDateTime;
        ULONGLONG kernel = (static_cast<ULONGLONG>(kernelTime.dwHighDateTime) << 32) | 
                           kernelTime.dwLowDateTime;
        ULONGLONG user = (static_cast<ULONGLONG>(userTime.dwHighDateTime) << 32) | 
                         userTime.dwLowDateTime;
        
        ULONGLONG idleDiff = idle - lastIdle;
        ULONGLONG totalDiff = (kernel - lastKernel) + (user - lastUser);
        
        if (totalDiff > 0) {
            metrics.cpuUsagePercent = 
                100.0f * (1.0f - static_cast<float>(idleDiff) / static_cast<float>(totalDiff));
        }
        
        lastIdle = idle;
        lastKernel = kernel;
        lastUser = user;
    }

    // CPU cores
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    metrics.cpuCores = static_cast<int>(sysInfo.dwNumberOfProcessors);

    // System memory
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        metrics.systemMemoryTotalMB = 
            static_cast<size_t>(memStatus.ullTotalPhys / (1024 * 1024));
        metrics.systemMemoryUsedMB = 
            static_cast<size_t>((memStatus.ullTotalPhys - memStatus.ullAvailPhys) / 
                               (1024 * 1024));
    }
#endif

    return metrics;
}

//=============================================================================
// PerformanceTimer Implementation
//=============================================================================

PerformanceTimer::PerformanceTimer()
    : m_running(false)
{
    reset();
}

void PerformanceTimer::start()
{
    m_startTime = std::chrono::high_resolution_clock::now();
    m_running = true;
}

float PerformanceTimer::stop()
{
    if (!m_running) {
        return 0.0f;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    m_running = false;
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - m_startTime).count();
    
    return static_cast<float>(duration) / 1000.0f; // Return milliseconds
}

float PerformanceTimer::elapsed() const
{
    if (!m_running) {
        return 0.0f;
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        now - m_startTime).count();
    
    return static_cast<float>(duration) / 1000.0f;
}

void PerformanceTimer::recordInference(float latencyMs)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    updateStats(latencyMs);
}

void PerformanceTimer::updateStats(float latencyMs)
{
    m_metrics.latencyMs = latencyMs;
    m_metrics.totalInferences++;
    
    // Update history
    m_metrics.latencyHistory.push_back(latencyMs);
    if (m_metrics.latencyHistory.size() > InferenceMetrics::HISTORY_SIZE) {
        m_metrics.latencyHistory.erase(m_metrics.latencyHistory.begin());
    }
    
    // Calculate statistics
    if (!m_metrics.latencyHistory.empty()) {
        float sum = std::accumulate(m_metrics.latencyHistory.begin(), 
                                    m_metrics.latencyHistory.end(), 0.0f);
        m_metrics.avgLatencyMs = sum / m_metrics.latencyHistory.size();
        
        auto minmax = std::minmax_element(m_metrics.latencyHistory.begin(), 
                                          m_metrics.latencyHistory.end());
        m_metrics.minLatencyMs = *minmax.first;
        m_metrics.maxLatencyMs = *minmax.second;
        
        // P99 latency
        if (m_metrics.latencyHistory.size() >= 10) {
            std::vector<float> sorted = m_metrics.latencyHistory;
            std::sort(sorted.begin(), sorted.end());
            size_t p99Index = static_cast<size_t>(sorted.size() * 0.99);
            m_metrics.p99LatencyMs = sorted[p99Index];
        }
        
        // Throughput
        m_metrics.throughputFPS = 1000.0f / m_metrics.avgLatencyMs;
    }
}

InferenceMetrics PerformanceTimer::getMetrics() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_metrics;
}

void PerformanceTimer::reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_metrics = InferenceMetrics{};
}

float PerformanceTimer::getFPS() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_metrics.throughputFPS;
}

//=============================================================================
// FrameRateCounter Implementation
//=============================================================================

FrameRateCounter::FrameRateCounter(int windowSize)
    : m_windowSize(windowSize)
{
    m_frameTimes.reserve(windowSize);
}

void FrameRateCounter::tick()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    
    auto now = std::chrono::steady_clock::now();
    m_frameTimes.push_back(now);
    
    if (static_cast<int>(m_frameTimes.size()) > m_windowSize) {
        m_frameTimes.erase(m_frameTimes.begin());
    }
}

float FrameRateCounter::getFPS() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_frameTimes.size() < 2) {
        return 0.0f;
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        m_frameTimes.back() - m_frameTimes.front()).count();
    
    if (duration == 0) {
        return 0.0f;
    }
    
    return static_cast<float>(m_frameTimes.size() - 1) * 1000.0f / 
           static_cast<float>(duration);
}

float FrameRateCounter::getFrameTimeMs() const
{
    float fps = getFPS();
    return fps > 0.0f ? 1000.0f / fps : 0.0f;
}

void FrameRateCounter::reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_frameTimes.clear();
}

} // namespace monitor
} // namespace aiforge
