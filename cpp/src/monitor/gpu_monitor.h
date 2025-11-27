/**
 * AI Forge Studio - GPU Monitor Module
 * Author: M.3R3
 * 
 * Real-time GPU monitoring using NVML (NVIDIA Management Library).
 */

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>

namespace aiforge {
namespace monitor {

/**
 * GPU metrics structure
 */
struct GPUMetrics {
    // Identification
    std::string name;
    std::string driverVersion;
    std::string cudaVersion;
    int deviceIndex;
    
    // Utilization
    float gpuUsagePercent;      // 0-100%
    float memoryUsagePercent;   // 0-100%
    
    // Memory
    size_t memoryUsedMB;
    size_t memoryTotalMB;
    size_t memoryFreeMB;
    
    // Performance
    float temperatureCelsius;
    float powerDrawWatts;
    float powerLimitWatts;
    int clockSpeedMHz;
    int memoryClockMHz;
    int fanSpeedPercent;
    
    // PCIe
    int pcieTxMBps;
    int pcieRxMBps;
    int pcieGen;
    int pcieLanes;
    
    // Compute
    int smCount;
    int cudaCores;
    int tensorCores;
    
    // Status
    bool isHealthy;
    std::string status;
    
    // Timestamp
    std::chrono::steady_clock::time_point timestamp;
};

/**
 * Inference timing metrics
 */
struct InferenceMetrics {
    float latencyMs;
    float throughputFPS;
    float avgLatencyMs;
    float minLatencyMs;
    float maxLatencyMs;
    float p99LatencyMs;
    uint64_t totalInferences;
    
    // Rolling averages
    std::vector<float> latencyHistory;
    static constexpr size_t HISTORY_SIZE = 100;
};

/**
 * System-wide metrics
 */
struct SystemMetrics {
    // CPU
    float cpuUsagePercent;
    int cpuCores;
    float cpuTemperature;
    
    // Memory
    size_t systemMemoryUsedMB;
    size_t systemMemoryTotalMB;
    
    // Disk
    size_t diskReadMBps;
    size_t diskWriteMBps;
};

/**
 * Callback types
 */
using MetricsCallback = std::function<void(const GPUMetrics&)>;
using AlertCallback = std::function<void(const std::string& alert)>;

/**
 * GPU Monitor Class
 * 
 * Provides real-time monitoring of NVIDIA GPUs using NVML.
 */
class GPUMonitor {
public:
    GPUMonitor();
    ~GPUMonitor();

    // Disable copy
    GPUMonitor(const GPUMonitor&) = delete;
    GPUMonitor& operator=(const GPUMonitor&) = delete;

    /**
     * Initialize NVML and detect GPUs
     * @return true if initialization successful
     */
    bool initialize();

    /**
     * Shutdown monitoring and cleanup
     */
    void shutdown();

    /**
     * Check if monitor is ready
     */
    bool isReady() const { return m_initialized; }

    /**
     * Get number of detected GPUs
     */
    int getGPUCount() const { return m_gpuCount; }

    /**
     * Get current metrics for a specific GPU
     * @param deviceIndex GPU index (default: 0)
     * @return Current GPU metrics
     */
    GPUMetrics getMetrics(int deviceIndex = 0);

    /**
     * Get metrics for all GPUs
     * @return Vector of GPU metrics
     */
    std::vector<GPUMetrics> getAllMetrics();

    /**
     * Start background monitoring thread
     * @param intervalMs Update interval in milliseconds
     */
    void startMonitoring(int intervalMs = 100);

    /**
     * Stop background monitoring
     */
    void stopMonitoring();

    /**
     * Check if monitoring is active
     */
    bool isMonitoring() const { return m_monitoring.load(); }

    /**
     * Set callback for metrics updates
     */
    void setMetricsCallback(MetricsCallback callback);

    /**
     * Set callback for alerts (temperature, errors, etc.)
     */
    void setAlertCallback(AlertCallback callback);

    /**
     * Set temperature threshold for alerts (Celsius)
     */
    void setTemperatureThreshold(float celsius) { m_tempThreshold = celsius; }

    /**
     * Set memory threshold for alerts (percent)
     */
    void setMemoryThreshold(float percent) { m_memThreshold = percent; }

    /**
     * Get last error message
     */
    std::string getLastError() const { return m_lastError; }

    /**
     * Get system metrics (CPU, system memory)
     */
    SystemMetrics getSystemMetrics();

private:
    bool m_initialized;
    int m_gpuCount;
    std::string m_lastError;
    
    // Monitoring thread
    std::atomic<bool> m_monitoring;
    std::thread m_monitorThread;
    int m_intervalMs;
    
    // Callbacks
    MetricsCallback m_metricsCallback;
    AlertCallback m_alertCallback;
    std::mutex m_callbackMutex;
    
    // Thresholds
    float m_tempThreshold;
    float m_memThreshold;
    
    // NVML handles (opaque pointers)
    std::vector<void*> m_devices;
    
    void monitoringLoop();
    void checkAlerts(const GPUMetrics& metrics);
};

/**
 * Performance Timer
 * 
 * High-precision timing for inference benchmarking.
 */
class PerformanceTimer {
public:
    PerformanceTimer();
    
    /**
     * Start timing
     */
    void start();
    
    /**
     * Stop timing and record result
     * @return Elapsed time in milliseconds
     */
    float stop();
    
    /**
     * Get current elapsed time without stopping
     */
    float elapsed() const;
    
    /**
     * Record an inference timing
     * @param latencyMs Latency in milliseconds
     */
    void recordInference(float latencyMs);
    
    /**
     * Get inference metrics
     */
    InferenceMetrics getMetrics() const;
    
    /**
     * Reset all statistics
     */
    void reset();
    
    /**
     * Get FPS based on recent timings
     */
    float getFPS() const;

private:
    std::chrono::high_resolution_clock::time_point m_startTime;
    bool m_running;
    
    InferenceMetrics m_metrics;
    mutable std::mutex m_mutex;
    
    void updateStats(float latencyMs);
};

/**
 * Frame Rate Counter
 * 
 * Smooth FPS calculation with configurable averaging.
 */
class FrameRateCounter {
public:
    FrameRateCounter(int windowSize = 60);
    
    /**
     * Record a frame
     */
    void tick();
    
    /**
     * Get current FPS
     */
    float getFPS() const;
    
    /**
     * Get frame time in milliseconds
     */
    float getFrameTimeMs() const;
    
    /**
     * Reset counter
     */
    void reset();

private:
    std::vector<std::chrono::steady_clock::time_point> m_frameTimes;
    int m_windowSize;
    mutable std::mutex m_mutex;
};

} // namespace monitor
} // namespace aiforge
