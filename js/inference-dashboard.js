/**
 * AI Forge Studio - Inference Dashboard
 * Connects to C++ TensorRT/CUDA backend for real inference benchmarks
 * 
 * Author: M.3R3 | AI Forge OPS
 */

class InferenceDashboard {
    constructor() {
        this.lastResult = null;
        this.isRunning = false;
        this.history = [];
        this.maxHistory = 20;
        
        this.init();
    }
    
    async init() {
        // Check if running in Electron
        if (typeof window.electronAPI !== 'undefined' && window.electronAPI.isElectron) {
            console.log('🧠 Inference Dashboard: Running in Electron mode');
            await this.checkAvailability();
            this.setupEventListeners();
        } else {
            console.log('🌐 Inference Dashboard: Running in browser mode');
            this.setupMockMode();
        }
    }
    
    async checkAvailability() {
        try {
            const available = await window.electronAPI.inference.isAvailable();
            this.updateStatus(available ? 'ready' : 'unavailable');
            
            if (available) {
                const modes = await window.electronAPI.inference.getModes();
                this.populateModeSelector(modes);
            }
        } catch (error) {
            console.error('Failed to check inference availability:', error);
            this.updateStatus('error');
        }
    }
    
    setupEventListeners() {
        // Run Benchmark button
        const runBtn = document.querySelector('#run-inference-btn, .run-inference-btn, [data-action="run-benchmark"]');
        if (runBtn) {
            runBtn.addEventListener('click', () => this.runBenchmark());
        }
        
        // Mode selector
        const modeSelect = document.querySelector('#inference-mode, .inference-mode-select');
        if (modeSelect) {
            modeSelect.addEventListener('change', (e) => {
                this.selectedMode = e.target.value;
            });
        }
        
        // Quick mode buttons
        document.querySelectorAll('[data-inference-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.target.dataset.inferenceMode;
                this.runBenchmark(mode);
            });
        });
    }
    
    setupMockMode() {
        // Browser mock mode
        const runBtn = document.querySelector('#run-inference-btn, .run-inference-btn, [data-action="run-benchmark"]');
        if (runBtn) {
            runBtn.addEventListener('click', () => this.runMockBenchmark());
        }
    }
    
    populateModeSelector(modes) {
        const select = document.querySelector('#inference-mode, .inference-mode-select');
        if (!select) return;
        
        select.innerHTML = modes.map(mode => 
            `<option value="${mode.id}" title="${mode.description}">${mode.name}</option>`
        ).join('');
        
        this.selectedMode = modes[0]?.id || 'cuda';
    }
    
    async runBenchmark(mode = null) {
        if (this.isRunning) {
            console.log('Benchmark already running');
            return;
        }
        
        const selectedMode = mode || this.selectedMode || 'cuda';
        
        this.isRunning = true;
        this.updateStatus('running');
        this.updateUI({ status: 'Running benchmark...' });
        
        try {
            console.log(`🚀 Running ${selectedMode} benchmark...`);
            const startTime = performance.now();
            
            const result = await window.electronAPI.inference.runBenchmark(selectedMode);
            
            const elapsedTime = performance.now() - startTime;
            console.log(`✅ Benchmark completed in ${elapsedTime.toFixed(0)}ms`);
            
            this.lastResult = result;
            this.addToHistory(result);
            this.updateUI(result);
            this.updateStatus(result.success ? 'success' : 'error');
            
        } catch (error) {
            console.error('Benchmark failed:', error);
            this.updateStatus('error');
            this.updateUI({ 
                success: false, 
                error: error.message,
                status: 'Benchmark failed' 
            });
        } finally {
            this.isRunning = false;
        }
    }
    
    async runMockBenchmark() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.updateStatus('running');
        
        // Simulate benchmark
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const mockResult = {
            success: true,
            mode: 'mock',
            model: 'ResNet-50 FP16 (Browser Mock)',
            precision: 'FP16',
            batchSize: 1,
            inputShape: [1, 3, 224, 224],
            latencyMs: 0.3 + Math.random() * 0.2,
            throughputFPS: 2500 + Math.random() * 1000,
            inferenceMemoryMB: 98.5,
            device: 'Simulated GPU',
            timestamp: new Date().toISOString()
        };
        
        this.lastResult = mockResult;
        this.updateUI(mockResult);
        this.updateStatus('success');
        this.isRunning = false;
    }
    
    updateUI(result) {
        if (!result) return;
        
        // Update main metrics
        this.updateElement('.inference-latency, #inference-latency', 
            result.latencyMs ? `${result.latencyMs.toFixed(2)} ms` : '--');
        
        this.updateElement('.inference-throughput, #inference-throughput', 
            result.throughputFPS ? `${Math.round(result.throughputFPS)} FPS` : '--');
        
        this.updateElement('.inference-model, #inference-model', 
            result.model || '--');
        
        this.updateElement('.inference-precision, #inference-precision', 
            result.precision || '--');
        
        this.updateElement('.inference-mode, #current-inference-mode', 
            result.mode || '--');
        
        this.updateElement('.inference-device, #inference-device', 
            result.device || '--');
        
        this.updateElement('.inference-batch-size, #inference-batch-size', 
            result.batchSize || '--');
        
        // Input shape
        if (result.inputShape) {
            const shape = Array.isArray(result.inputShape) 
                ? result.inputShape.join(' × ') 
                : result.inputShape;
            this.updateElement('.inference-input-shape, #inference-input-shape', shape);
        }
        
        // Memory usage
        this.updateElement('.inference-memory, #inference-memory', 
            result.inferenceMemoryMB ? `${result.inferenceMemoryMB.toFixed(1)} MB` : '--');
        
        // GPU info
        this.updateElement('.inference-cuda-version, #inference-cuda-version', 
            result.cudaVersion || '--');
        
        this.updateElement('.inference-driver, #inference-driver', 
            result.driver || '--');
        
        this.updateElement('.inference-sm-count, #inference-sm-count', 
            result.smCount || '--');
        
        this.updateElement('.inference-cuda-cores, #inference-cuda-cores', 
            result.cudaCores ? result.cudaCores.toLocaleString() : '--');
        
        // Timestamp
        if (result.timestamp) {
            const time = new Date(result.timestamp).toLocaleTimeString();
            this.updateElement('.inference-timestamp, #inference-timestamp', time);
        }
        
        // Status message
        if (result.status) {
            this.updateElement('.inference-status-text, #inference-status-text', result.status);
        } else if (result.success) {
            this.updateElement('.inference-status-text, #inference-status-text', 'Benchmark completed');
        } else if (result.error) {
            this.updateElement('.inference-status-text, #inference-status-text', `Error: ${result.error}`);
        }
        
        // Update progress bars if present
        this.updateProgressBars(result);
    }
    
    updateProgressBars(result) {
        // Throughput bar (max 5000 FPS for scale)
        const throughputPercent = Math.min(100, (result.throughputFPS / 5000) * 100);
        this.updateProgressBar('.throughput-bar', throughputPercent);
        
        // Latency bar (inverted - lower is better, max 10ms)
        const latencyPercent = Math.min(100, (1 - result.latencyMs / 10) * 100);
        this.updateProgressBar('.latency-bar', latencyPercent);
    }
    
    updateElement(selectors, value) {
        const selectorList = selectors.split(',').map(s => s.trim());
        for (const selector of selectorList) {
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => {
                if (el.tagName === 'INPUT') {
                    el.value = value;
                } else {
                    el.textContent = value;
                }
            });
        }
    }
    
    updateProgressBar(selector, percent) {
        const bars = document.querySelectorAll(selector);
        bars.forEach(bar => {
            bar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
        });
    }
    
    updateStatus(status) {
        const statusEl = document.querySelector('.inference-status, #inference-status');
        if (!statusEl) return;
        
        statusEl.classList.remove('ready', 'running', 'success', 'error', 'unavailable');
        statusEl.classList.add(status);
        
        const statusTexts = {
            'ready': '🟢 Ready',
            'running': '🔄 Running...',
            'success': '✅ Complete',
            'error': '❌ Error',
            'unavailable': '⚠️ Unavailable'
        };
        
        const badge = statusEl.querySelector('.status-badge, .badge');
        if (badge) {
            badge.textContent = statusTexts[status] || status;
        }
    }
    
    addToHistory(result) {
        this.history.unshift({
            ...result,
            id: Date.now()
        });
        
        if (this.history.length > this.maxHistory) {
            this.history.pop();
        }
        
        this.updateHistoryUI();
    }
    
    updateHistoryUI() {
        const container = document.querySelector('.inference-history, #inference-history');
        if (!container) return;
        
        container.innerHTML = this.history.slice(0, 5).map(item => `
            <div class="history-item ${item.success ? 'success' : 'error'}">
                <span class="history-mode">${item.mode}</span>
                <span class="history-fps">${Math.round(item.throughputFPS)} FPS</span>
                <span class="history-latency">${item.latencyMs.toFixed(2)}ms</span>
                <span class="history-time">${new Date(item.timestamp).toLocaleTimeString()}</span>
            </div>
        `).join('');
    }
    
    getLastResult() {
        return this.lastResult;
    }
    
    getHistory() {
        return this.history;
    }
}

// Initialize on DOM ready
let inferenceDashboard = null;

document.addEventListener('DOMContentLoaded', () => {
    // Only initialize on inference page
    if (document.querySelector('.inference-dashboard, #inference-container, [data-page="inference"]') ||
        window.location.pathname.includes('inference')) {
        inferenceDashboard = new InferenceDashboard();
        console.log('🧠 Inference Dashboard initialized');
    }
});

// Export for use in other scripts
window.inferenceDashboard = inferenceDashboard;
window.InferenceDashboard = InferenceDashboard;
