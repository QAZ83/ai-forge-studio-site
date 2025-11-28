/**
 * AI Forge Studio - Real GPU Dashboard
 * Displays real GPU metrics from RTX 5090 via C++ backend
 */

class RealGPUDashboard {
    constructor() {
        this.metrics = null;
        this.isConnected = false;
        this.updateCallbacks = [];
        
        this.init();
    }
    
    async init() {
        // Check if running in Electron
        if (typeof window.electronAPI !== 'undefined' && window.electronAPI.isElectron) {
            console.log('🎮 GPU Dashboard: Running in Electron mode');
            await this.startRealMonitoring();
        } else {
            console.log('🌐 GPU Dashboard: Running in browser mode (mock data)');
            this.startMockMonitoring();
        }
    }
    
    async startRealMonitoring() {
        try {
            // Get initial metrics
            this.metrics = await window.electronAPI.getGpuMetrics();
            this.isConnected = this.metrics.success;
            this.updateUI(this.metrics);
            
            // Start continuous monitoring
            await window.electronAPI.startGpuMonitor(1000);
            
            // Listen for updates
            window.electronAPI.onGpuMetricsUpdate((metrics) => {
                this.metrics = metrics;
                this.isConnected = metrics.success;
                this.updateUI(metrics);
            });
            
            console.log('✅ Real GPU monitoring started');
        } catch (error) {
            console.error('❌ Failed to start GPU monitoring:', error);
            this.startMockMonitoring();
        }
    }
    
    startMockMonitoring() {
        // Fallback mock data for browser testing
        this.metrics = {
            success: true,
            gpu: 'Mock GPU (Browser Mode)',
            driverVersion: 'N/A',
            cudaVersion: 'N/A',
            computeCapability: 'N/A',
            hardware: { smCount: 0, cudaCores: 0, memoryBusWidth: 0 },
            memory: { totalMB: 8192, usedMB: 2048, freeMB: 6144, usagePercent: 25 },
            utilization: { gpuPercent: 15, memoryPercent: 10 },
            clocks: { gpuMHz: 1500, memoryMHz: 7000, maxGpuMHz: 2000, maxMemoryMHz: 8000 },
            power: { drawW: 50, limitW: 200, usagePercent: 25 },
            thermal: { temperatureC: 45, fanSpeedPercent: 30 }
        };
        
        this.updateUI(this.metrics);
        
        // Simulate updates
        setInterval(() => {
            // Add some variance
            this.metrics.utilization.gpuPercent = Math.min(100, Math.max(0, 
                this.metrics.utilization.gpuPercent + (Math.random() - 0.5) * 10));
            this.metrics.thermal.temperatureC = Math.min(90, Math.max(30, 
                this.metrics.thermal.temperatureC + (Math.random() - 0.5) * 2));
            this.metrics.power.drawW = Math.min(this.metrics.power.limitW, Math.max(20, 
                this.metrics.power.drawW + (Math.random() - 0.5) * 10));
            
            this.updateUI(this.metrics);
        }, 2000);
    }
    
    updateUI(metrics) {
        if (!metrics) return;
        
        // Update GPU name
        this.updateElement('.gpu-name', metrics.gpu);
        this.updateElement('#gpu-name', metrics.gpu);
        
        // Update driver/CUDA version
        this.updateElement('.driver-version', metrics.driverVersion);
        this.updateElement('.cuda-version', metrics.cudaVersion);
        this.updateElement('#cuda-version', metrics.cudaVersion);
        
        // Update memory
        if (metrics.memory) {
            const memUsed = (metrics.memory.usedMB / 1024).toFixed(1);
            const memTotal = (metrics.memory.totalMB / 1024).toFixed(0);
            this.updateElement('.vram-used', `${memUsed} GB`);
            this.updateElement('.vram-total', `${memTotal} GB`);
            this.updateElement('.vram-usage', `${metrics.memory.usagePercent}%`);
            this.updateElement('#vram-used', `${memUsed} GB`);
            this.updateElement('#vram-free', `${(metrics.memory.freeMB / 1024).toFixed(1)} GB`);
            
            // Update progress bars
            this.updateProgressBar('#vram-bar', metrics.memory.usagePercent);
            this.updateProgressBar('.vram-progress', metrics.memory.usagePercent);
        }
        
        // Update utilization
        if (metrics.utilization) {
            this.updateElement('.gpu-utilization', `${metrics.utilization.gpuPercent}%`);
            this.updateElement('#gpu-utilization', `${metrics.utilization.gpuPercent}%`);
            this.updateProgressBar('.gpu-util-bar', metrics.utilization.gpuPercent);
            
            // Update circular progress if exists
            this.updateCircularProgress('[data-metric="gpu"]', metrics.utilization.gpuPercent);
        }
        
        // Update thermal
        if (metrics.thermal) {
            this.updateElement('.gpu-temp', `${metrics.thermal.temperatureC}°C`);
            this.updateElement('#gpu-temp', `${metrics.thermal.temperatureC}°C`);
            this.updateElement('.fan-speed', `${metrics.thermal.fanSpeedPercent}%`);
        }
        
        // Update power
        if (metrics.power) {
            this.updateElement('.power-draw', `${metrics.power.drawW}W`);
            this.updateElement('.power-limit', `${metrics.power.limitW}W`);
            this.updateElement('#power-draw', `${metrics.power.drawW}W / ${metrics.power.limitW}W`);
            this.updateProgressBar('.power-bar', metrics.power.usagePercent);
        }
        
        // Update clocks
        if (metrics.clocks) {
            this.updateElement('.gpu-clock', `${metrics.clocks.gpuMHz} MHz`);
            this.updateElement('.memory-clock', `${metrics.clocks.memoryMHz} MHz`);
            this.updateElement('#gpu-clock', `${metrics.clocks.gpuMHz} / ${metrics.clocks.maxGpuMHz} MHz`);
        }
        
        // Update hardware specs
        if (metrics.hardware) {
            this.updateElement('.cuda-cores', metrics.hardware.cudaCores.toLocaleString());
            this.updateElement('.sm-count', metrics.hardware.smCount);
            this.updateElement('#cuda-cores', metrics.hardware.cudaCores.toLocaleString());
        }
        
        // Update compute capability
        this.updateElement('.compute-capability', metrics.computeCapability);
        
        // Notify callbacks
        for (const callback of this.updateCallbacks) {
            try {
                callback(metrics);
            } catch (e) {
                console.error('GPU Dashboard callback error:', e);
            }
        }
    }
    
    updateElement(selector, value) {
        const elements = document.querySelectorAll(selector);
        elements.forEach(el => {
            if (el.tagName === 'INPUT') {
                el.value = value;
            } else {
                el.textContent = value;
            }
        });
    }
    
    updateProgressBar(selector, percent) {
        const bars = document.querySelectorAll(selector);
        bars.forEach(bar => {
            bar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
        });
    }
    
    updateCircularProgress(selector, percent) {
        const circle = document.querySelector(selector);
        if (!circle) return;
        
        const valueEl = circle.querySelector('.progress-value .value');
        if (valueEl) {
            valueEl.textContent = `${Math.round(percent)}%`;
        }
        
        const fill = circle.querySelector('.progress-ring-fill');
        if (fill) {
            const circumference = 2 * Math.PI * 60;
            fill.style.strokeDashoffset = circumference - (percent / 100) * circumference;
        }
    }
    
    onUpdate(callback) {
        this.updateCallbacks.push(callback);
    }
    
    getMetrics() {
        return this.metrics;
    }
}

// Initialize on DOM ready
let gpuDashboard = null;

document.addEventListener('DOMContentLoaded', () => {
    gpuDashboard = new RealGPUDashboard();
    console.log('🎮 Real GPU Dashboard initialized');
});

// Export for use in other scripts
window.gpuDashboard = gpuDashboard;
window.RealGPUDashboard = RealGPUDashboard;
