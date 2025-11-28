/**
 * AI Forge Studio - GPU Monitor Service
 * Reads real GPU metrics from C++ backend (gpu_info_json.exe)
 * 
 * This runs in the main process and sends data to renderer via IPC
 */

const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

class GPUMonitor {
    constructor() {
        this.exePath = null;
        this.intervalId = null;
        this.lastMetrics = null;
        this.listeners = [];
        this.updateInterval = 1000; // 1 second
        this.isRunning = false;
        
        this.findExecutable();
    }
    
    /**
     * Find the gpu_info_json executable
     */
    findExecutable() {
        const possiblePaths = [
            // Development paths
            path.join(__dirname, '..', 'cpp-backend', 'build', 'bin', 'Release', 'gpu_info_json.exe'),
            path.join(__dirname, '..', 'cpp-backend', 'build', 'bin', 'Debug', 'gpu_info_json.exe'),
            // Production paths (packaged app)
            path.join(process.resourcesPath || '', 'bin', 'gpu_info_json.exe'),
            path.join(__dirname, '..', 'bin', 'gpu_info_json.exe'),
            // Current directory
            path.join(process.cwd(), 'cpp-backend', 'build', 'bin', 'Release', 'gpu_info_json.exe'),
        ];
        
        for (const exePath of possiblePaths) {
            if (fs.existsSync(exePath)) {
                this.exePath = exePath;
                console.log(`[GPUMonitor] Found executable: ${exePath}`);
                return;
            }
        }
        
        console.warn('[GPUMonitor] gpu_info_json.exe not found. GPU monitoring will use fallback data.');
        console.warn('[GPUMonitor] Searched paths:', possiblePaths);
    }
    
    /**
     * Start monitoring GPU metrics
     */
    start(intervalMs = 1000) {
        if (this.isRunning) {
            console.log('[GPUMonitor] Already running');
            return;
        }
        
        this.updateInterval = intervalMs;
        this.isRunning = true;
        
        console.log(`[GPUMonitor] Starting with ${intervalMs}ms interval`);
        
        // Get initial reading
        this.fetchMetrics();
        
        // Set up interval
        this.intervalId = setInterval(() => {
            this.fetchMetrics();
        }, this.updateInterval);
    }
    
    /**
     * Stop monitoring
     */
    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.isRunning = false;
        console.log('[GPUMonitor] Stopped');
    }
    
    /**
     * Add a listener for metrics updates
     */
    onUpdate(callback) {
        this.listeners.push(callback);
        // Send last known metrics immediately if available
        if (this.lastMetrics) {
            callback(this.lastMetrics);
        }
    }
    
    /**
     * Remove a listener
     */
    removeListener(callback) {
        this.listeners = this.listeners.filter(l => l !== callback);
    }
    
    /**
     * Fetch metrics from C++ executable
     */
    fetchMetrics() {
        if (!this.exePath) {
            // Return fallback/mock data if executable not found
            this.handleMetrics(this.getFallbackMetrics());
            return;
        }
        
        execFile(this.exePath, { timeout: 5000 }, (error, stdout, stderr) => {
            if (error) {
                console.error('[GPUMonitor] Error running executable:', error.message);
                this.handleMetrics(this.getFallbackMetrics());
                return;
            }
            
            try {
                const metrics = JSON.parse(stdout.trim());
                this.handleMetrics(metrics);
            } catch (parseError) {
                console.error('[GPUMonitor] Error parsing JSON:', parseError.message);
                console.error('[GPUMonitor] Raw output:', stdout);
                this.handleMetrics(this.getFallbackMetrics());
            }
        });
    }
    
    /**
     * Handle received metrics
     */
    handleMetrics(metrics) {
        this.lastMetrics = metrics;
        
        // Notify all listeners
        for (const listener of this.listeners) {
            try {
                listener(metrics);
            } catch (e) {
                console.error('[GPUMonitor] Listener error:', e);
            }
        }
    }
    
    /**
     * Get fallback metrics when exe is not available
     */
    getFallbackMetrics() {
        return {
            success: false,
            error: 'GPU monitor executable not found',
            gpu: 'Unknown GPU',
            driverVersion: 'N/A',
            cudaVersion: 'N/A',
            computeCapability: 'N/A',
            hardware: {
                smCount: 0,
                cudaCores: 0,
                memoryBusWidth: 0
            },
            memory: {
                totalMB: 0,
                usedMB: 0,
                freeMB: 0,
                usagePercent: 0
            },
            utilization: {
                gpuPercent: 0,
                memoryPercent: 0
            },
            clocks: {
                gpuMHz: 0,
                memoryMHz: 0,
                maxGpuMHz: 0,
                maxMemoryMHz: 0
            },
            power: {
                drawW: 0,
                limitW: 0,
                usagePercent: 0
            },
            thermal: {
                temperatureC: 0,
                fanSpeedPercent: 0
            }
        };
    }
    
    /**
     * Get current metrics (sync)
     */
    getMetrics() {
        return this.lastMetrics;
    }
    
    /**
     * Get metrics as a promise
     */
    getMetricsAsync() {
        return new Promise((resolve) => {
            if (!this.exePath) {
                resolve(this.getFallbackMetrics());
                return;
            }
            
            execFile(this.exePath, { timeout: 5000 }, (error, stdout) => {
                if (error) {
                    resolve(this.getFallbackMetrics());
                    return;
                }
                
                try {
                    resolve(JSON.parse(stdout.trim()));
                } catch (e) {
                    resolve(this.getFallbackMetrics());
                }
            });
        });
    }
}

// Export singleton instance
const gpuMonitor = new GPUMonitor();

module.exports = { GPUMonitor, gpuMonitor };
