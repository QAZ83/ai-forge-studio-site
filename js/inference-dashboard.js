/**
 * AI Forge Studio - Inference Dashboard
 * Connects to C++ TensorRT/CUDA backend for real inference benchmarks
 * 
 * Supported Modes:
 *   - cuda: Real CUDA compute benchmark (matrix multiplication)
 *   - mock: Simulated TensorRT inference
 *   - tensorrt: Load and run optimized .engine file
 *   - onnx: Convert ONNX model to TensorRT and benchmark
 * 
 * Author: M.3R3 | AI Forge OPS
 */

class InferenceDashboard {
    constructor() {
        this.lastResult = null;
        this.isRunning = false;
        this.history = [];
        this.maxHistory = 20;
        this.selectedMode = 'cuda';
        this.selectedModelPath = null;
        this.status = null;
        
        this.init();
    }
    
    async init() {
        // Check if running in Electron
        if (typeof window.electronAPI !== 'undefined' && window.electronAPI.isElectron) {
            console.log('🧠 Inference Dashboard: Running in Electron mode');
            await this.loadStatus();
            await this.checkAvailability();
            this.setupEventListeners();
        } else {
            console.log('🌐 Inference Dashboard: Running in browser mode');
            this.setupMockMode();
        }
    }
    
    async loadStatus() {
        try {
            this.status = await window.electronAPI.inference.getStatus();
            console.log('📊 Inference Status:', this.status);
            
            // Update status indicators
            const tensorrtIndicator = document.querySelector('#tensorrt-status, .tensorrt-status');
            if (tensorrtIndicator) {
                tensorrtIndicator.textContent = this.status.tensorrtPath ? '✅ TensorRT Ready' : '⚠️ TensorRT Not Found';
                tensorrtIndicator.className = this.status.tensorrtPath ? 'status-ready' : 'status-warning';
            }
        } catch (error) {
            console.error('Failed to load inference status:', error);
        }
    }
    
    async checkAvailability() {
        try {
            const available = await window.electronAPI.inference.isAvailable();
            this.updateStatus(available ? 'ready' : 'unavailable');
            
            if (available) {
                const modes = await window.electronAPI.inference.getModes();
                this.populateModeSelector(modes);
                await this.loadAvailableModels();
            }
        } catch (error) {
            console.error('Failed to check inference availability:', error);
            this.updateStatus('error');
        }
    }
    
    async loadAvailableModels() {
        try {
            const models = await window.electronAPI.inference.listModels();
            this.populateModelSelector(models);
        } catch (error) {
            console.error('Failed to load models:', error);
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
                this.updateModeUI(e.target.value);
            });
        }
        
        // Model selector
        const modelSelect = document.querySelector('#model-select, .model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.selectedModelPath = e.target.value;
            });
        }
        
        // Browse model button
        const browseBtn = document.querySelector('#browse-model-btn, .browse-model-btn');
        if (browseBtn) {
            browseBtn.addEventListener('click', () => this.browseModel());
        }
        
        // Quick mode buttons
        document.querySelectorAll('[data-inference-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.target.dataset.inferenceMode;
                this.runBenchmark(mode);
            });
        });
        
        // Quick ONNX buttons
        document.querySelectorAll('[data-onnx-model]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modelPath = e.target.dataset.onnxModel;
                this.runBenchmark('onnx', { modelPath });
            });
        });
        
        // Quick Engine buttons
        document.querySelectorAll('[data-engine-file]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const enginePath = e.target.dataset.engineFile;
                this.runBenchmark('tensorrt', { enginePath });
            });
        });
    }
    
    updateModeUI(mode) {
        const modelSection = document.querySelector('.model-selection, #model-selection');
        if (modelSection) {
            // Show model selection only for tensorrt and onnx modes
            modelSection.style.display = (mode === 'tensorrt' || mode === 'onnx') ? 'block' : 'none';
        }
        
        // Update file type label
        const fileTypeLabel = document.querySelector('.model-type-label, #model-type-label');
        if (fileTypeLabel) {
            fileTypeLabel.textContent = mode === 'onnx' ? 'ONNX Model (.onnx)' : 'TensorRT Engine (.engine)';
        }
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
    
    populateModelSelector(models) {
        const select = document.querySelector('#model-select, .model-select');
        if (!select) return;
        
        let options = '<option value="">-- Select Model --</option>';
        
        if (models.engines && models.engines.length > 0) {
            options += '<optgroup label="TensorRT Engines (.engine)">';
            models.engines.forEach(m => {
                options += `<option value="${m.path}" data-type="engine">${m.name}</option>`;
            });
            options += '</optgroup>';
        }
        
        if (models.onnx && models.onnx.length > 0) {
            options += '<optgroup label="ONNX Models (.onnx)">';
            models.onnx.forEach(m => {
                options += `<option value="${m.path}" data-type="onnx">${m.name}</option>`;
            });
            options += '</optgroup>';
        }
        
        select.innerHTML = options;
    }
    
    async browseModel() {
        try {
            const type = this.selectedMode === 'onnx' ? 'onnx' : 'engine';
            const result = await window.electronAPI.inference.selectModel(type);
            
            if (result.success && result.path) {
                this.selectedModelPath = result.path;
                
                // Update UI
                const pathDisplay = document.querySelector('#selected-model-path, .selected-model-path');
                if (pathDisplay) {
                    pathDisplay.textContent = result.path.split(/[\\/]/).pop(); // Just filename
                    pathDisplay.title = result.path;
                }
                
                // Add to model select
                const select = document.querySelector('#model-select, .model-select');
                if (select) {
                    const option = new Option(result.path.split(/[\\/]/).pop(), result.path, true, true);
                    select.appendChild(option);
                }
            }
        } catch (error) {
            console.error('Failed to browse for model:', error);
        }
    }
    
    async runBenchmark(mode = null, options = {}) {
        if (this.isRunning) {
            console.log('Benchmark already running');
            return;
        }
        
        const selectedMode = mode || this.selectedMode || 'cuda';
        
        // Prepare options based on mode
        const benchmarkOptions = { ...options };
        
        if (selectedMode === 'tensorrt' && !benchmarkOptions.enginePath) {
            benchmarkOptions.enginePath = this.selectedModelPath;
            if (!benchmarkOptions.enginePath) {
                this.showError('Please select a TensorRT engine file (.engine)');
                return;
            }
        }
        
        if (selectedMode === 'onnx' && !benchmarkOptions.modelPath) {
            benchmarkOptions.modelPath = this.selectedModelPath;
            if (!benchmarkOptions.modelPath) {
                this.showError('Please select an ONNX model file (.onnx)');
                return;
            }
        }
        
        this.isRunning = true;
        this.updateStatus('running');
        this.updateUI({ status: `Running ${selectedMode.toUpperCase()} benchmark...` });
        
        try {
            console.log(`🚀 Running ${selectedMode} benchmark...`, benchmarkOptions);
            const startTime = performance.now();
            
            const result = await window.electronAPI.inference.runBenchmark(selectedMode, benchmarkOptions);
            
            const elapsedTime = performance.now() - startTime;
            console.log(`✅ Benchmark completed in ${elapsedTime.toFixed(0)}ms`);
            console.log('📊 Result:', result);
            
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
    
    showError(message) {
        const errorEl = document.querySelector('.inference-error, #inference-error');
        if (errorEl) {
            errorEl.textContent = message;
            errorEl.style.display = 'block';
            setTimeout(() => {
                errorEl.style.display = 'none';
            }, 5000);
        } else {
            alert(message);
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
            result.mode ? result.mode.toUpperCase() : '--');
        
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
        
        // Advanced latency metrics (TensorRT real mode)
        this.updateElement('.inference-min-latency, #inference-min-latency', 
            result.minLatencyMs ? `${result.minLatencyMs.toFixed(3)} ms` : '--');
        
        this.updateElement('.inference-max-latency, #inference-max-latency', 
            result.maxLatencyMs ? `${result.maxLatencyMs.toFixed(3)} ms` : '--');
        
        this.updateElement('.inference-p95-latency, #inference-p95-latency', 
            result.p95LatencyMs ? `${result.p95LatencyMs.toFixed(3)} ms` : '--');
        
        // TensorRT version
        this.updateElement('.inference-tensorrt-version, #inference-tensorrt-version', 
            result.tensorrtVersion || '--');
        
        // Model path (for TensorRT/ONNX modes)
        this.updateElement('.inference-model-path, #inference-model-path', 
            result.modelPath ? result.modelPath.split(/[\\/]/).pop() : '--');
        
        // Output classes
        this.updateElement('.inference-output-classes, #inference-output-classes', 
            result.outputClasses || '--');
        
        // Benchmark config
        this.updateElement('.inference-warmup-runs, #inference-warmup-runs', 
            result.warmupRuns || '--');
        
        this.updateElement('.inference-benchmark-runs, #inference-benchmark-runs', 
            result.benchmarkRuns || '--');
        
        // GPU info
        this.updateElement('.inference-cuda-version, #inference-cuda-version', 
            result.cudaVersion || '--');
        
        this.updateElement('.inference-driver, #inference-driver', 
            result.driver || '--');
        
        this.updateElement('.inference-sm-count, #inference-sm-count', 
            result.smCount || '--');
        
        this.updateElement('.inference-cuda-cores, #inference-cuda-cores', 
            result.cudaCores ? result.cudaCores.toLocaleString() : '--');
        
        this.updateElement('.inference-compute-capability, #inference-compute-capability', 
            result.computeCapability || '--');
        
        // VRAM info
        this.updateElement('.inference-vram-total, #inference-vram-total', 
            result.vramTotalMB ? `${(result.vramTotalMB / 1024).toFixed(1)} GB` : '--');
        
        this.updateElement('.inference-vram-used, #inference-vram-used', 
            result.vramUsedMB ? `${(result.vramUsedMB / 1024).toFixed(1)} GB` : '--');
        
        this.updateElement('.inference-vram-free, #inference-vram-free', 
            result.vramFreeMB ? `${(result.vramFreeMB / 1024).toFixed(1)} GB` : '--');
        
        // GPU stats
        this.updateElement('.inference-temperature, #inference-temperature', 
            result.temperatureC ? `${result.temperatureC}°C` : '--');
        
        this.updateElement('.inference-power, #inference-power', 
            result.powerDrawW ? `${result.powerDrawW}W / ${result.powerLimitW}W` : '--');
        
        this.updateElement('.inference-gpu-clock, #inference-gpu-clock', 
            result.gpuClockMHz ? `${result.gpuClockMHz} MHz` : '--');
        
        this.updateElement('.inference-mem-clock, #inference-mem-clock', 
            result.memClockMHz ? `${result.memClockMHz} MHz` : '--');
        
        // Timestamp
        if (result.timestamp) {
            const time = new Date(result.timestamp).toLocaleTimeString();
            this.updateElement('.inference-timestamp, #inference-timestamp', time);
        }
        
        // Status message
        if (result.status) {
            this.updateElement('.inference-status-text, #inference-status-text', result.status);
        } else if (result.success) {
            const modeText = result.mode === 'tensorrt' ? 'TensorRT' : 
                            result.mode === 'onnx' ? 'ONNX→TensorRT' :
                            result.mode === 'cuda' ? 'CUDA Compute' : 'Mock';
            this.updateElement('.inference-status-text, #inference-status-text', 
                `${modeText} benchmark completed`);
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
