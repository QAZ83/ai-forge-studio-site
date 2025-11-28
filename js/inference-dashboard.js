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
 * Model Manager:
 *   - Load models from models.json configuration
 *   - Select models via dropdown
 *   - Auto-detect mode based on model type
 * 
 * Author: M.3R3 | AI Forge OPS
 */

class InferenceDashboard {
    constructor() {
        this.lastResult = null;
        this.isRunning = false;
        this.history = [];
        this.maxHistory = 20;
        
        // Model Manager state
        this.models = [];
        this.selectedModelId = null;
        this.selectedMode = 'tensorrt';
        this.status = null;
        
        this.init();
    }
    
    async init() {
        // Check if running in Electron
        if (typeof window.electronAPI !== 'undefined' && window.electronAPI.isElectron) {
            console.log('🧠 Inference Dashboard: Running in Electron mode');
            await this.loadModels();
            await this.loadStatus();
            await this.refreshBenchmarkHistory(); // Load persistent history on startup
            this.setupEventListeners();
        } else {
            console.log('🌐 Inference Dashboard: Running in browser mode');
            this.setupMockMode();
        }
    }
    
    /**
     * Refresh benchmark history from persistent storage
     */
    async refreshBenchmarkHistory() {
        try {
            const runs = await window.electronAPI.inference.getHistory();
            const tbody = document.getElementById('benchmark-history-body');
            if (!tbody) return;
            
            tbody.innerHTML = '';
            
            if (!runs || runs.length === 0) {
                const tr = document.createElement('tr');
                tr.innerHTML = '<td colspan="5" class="empty-history">Run a benchmark to see history</td>';
                tbody.appendChild(tr);
                return;
            }
            
            runs.slice(0, 10).forEach(run => {
                const tr = document.createElement('tr');
                
                const timeCell = document.createElement('td');
                const modelCell = document.createElement('td');
                const modeCell = document.createElement('td');
                const latencyCell = document.createElement('td');
                const fpsCell = document.createElement('td');
                
                const date = new Date(run.timestamp);
                timeCell.textContent = date.toLocaleTimeString();
                
                modelCell.textContent = run.model || '-';
                modeCell.innerHTML = `<span class="mode-badge ${run.mode || ''}">${(run.mode || '').toUpperCase()}</span>`;
                latencyCell.textContent = run.latencyMs != null ? run.latencyMs.toFixed(3) + ' ms' : '--';
                fpsCell.textContent = run.throughputFPS != null ? run.throughputFPS.toFixed(0) + ' FPS' : '--';
                
                tr.appendChild(timeCell);
                tr.appendChild(modelCell);
                tr.appendChild(modeCell);
                tr.appendChild(latencyCell);
                tr.appendChild(fpsCell);
                
                tbody.appendChild(tr);
            });
            
            console.log(`📜 Loaded ${runs.length} history entries`);
        } catch (err) {
            console.error('[Renderer] Failed to refresh benchmark history:', err);
        }
    }
    
    /**
     * Load models from backend
     */
    async loadModels() {
        try {
            const data = await window.electronAPI.inference.getModels();
            console.log('📦 Models loaded:', data);
            
            this.models = data.models || [];
            this.selectedModelId = data.selectedModelId || null;
            this.selectedMode = data.selectedMode || 'tensorrt';
            
            this.populateModelDropdown();
            this.populateModeDropdown();
            this.updateModelInfo();
            
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }
    
    /**
     * Load inference service status
     */
    async loadStatus() {
        try {
            this.status = await window.electronAPI.inference.getStatus();
            console.log('📊 Inference Status:', this.status);
            
            // Update TensorRT status indicator
            const trtStatus = document.querySelector('#tensorrt-status, .tensorrt-status');
            if (trtStatus) {
                if (this.status.tensorrtPath) {
                    trtStatus.innerHTML = '<span class="status-dot ready"></span> TensorRT Ready';
                    trtStatus.className = 'status-indicator ready';
                } else {
                    trtStatus.innerHTML = '<span class="status-dot warning"></span> TensorRT Not Found';
                    trtStatus.className = 'status-indicator warning';
                }
            }
            
            // Update backend status
            const backendStatus = document.querySelector('#backend-status, .backend-status');
            if (backendStatus) {
                if (this.status.available) {
                    backendStatus.innerHTML = '<span class="status-dot ready"></span> Backend Ready';
                    backendStatus.className = 'status-indicator ready';
                } else {
                    backendStatus.innerHTML = '<span class="status-dot error"></span> Backend Unavailable';
                    backendStatus.className = 'status-indicator error';
                }
            }
            
            this.updateStatus(this.status.available ? 'ready' : 'unavailable');
            
        } catch (error) {
            console.error('Failed to load status:', error);
            this.updateStatus('error');
        }
    }
    
    /**
     * Populate model dropdown with available models
     */
    populateModelDropdown() {
        const select = document.querySelector('#model-select, .model-select');
        if (!select) return;
        
        // Group models by type
        const engines = this.models.filter(m => m.type === 'engine');
        const onnx = this.models.filter(m => m.type === 'onnx');
        
        let html = '';
        
        if (engines.length > 0) {
            html += '<optgroup label="🚀 TensorRT Engines (.engine)">';
            engines.forEach(m => {
                const selected = m.id === this.selectedModelId ? 'selected' : '';
                const exists = m.exists !== false ? '' : ' (missing)';
                html += `<option value="${m.id}" data-type="engine" ${selected}>${m.name}${exists}</option>`;
            });
            html += '</optgroup>';
        }
        
        if (onnx.length > 0) {
            html += '<optgroup label="📦 ONNX Models (.onnx)">';
            onnx.forEach(m => {
                const selected = m.id === this.selectedModelId ? 'selected' : '';
                const exists = m.exists !== false ? '' : ' (missing)';
                html += `<option value="${m.id}" data-type="onnx" ${selected}>${m.name}${exists}</option>`;
            });
            html += '</optgroup>';
        }
        
        if (this.models.length === 0) {
            html = '<option value="">No models available</option>';
        }
        
        select.innerHTML = html;
    }
    
    /**
     * Populate mode dropdown
     */
    populateModeDropdown() {
        const select = document.querySelector('#inference-mode, .inference-mode-select');
        if (!select) return;
        
        const modes = [
            { id: 'tensorrt', name: '🚀 TensorRT Engine', description: 'Load optimized .engine file' },
            { id: 'onnx', name: '📦 ONNX → TensorRT', description: 'Convert ONNX and run' },
            { id: 'cuda', name: '💻 CUDA Compute', description: 'Matrix multiplication benchmark' },
            { id: 'mock', name: '🎭 Mock / Simulation', description: 'Simulated inference' }
        ];
        
        select.innerHTML = modes.map(m => {
            const selected = m.id === this.selectedMode ? 'selected' : '';
            return `<option value="${m.id}" ${selected}>${m.name}</option>`;
        }).join('');
    }
    
    /**
     * Update model info display
     */
    updateModelInfo() {
        const model = this.getSelectedModel();
        
        // Model name
        this.updateElement('#selected-model-name, .selected-model-name', 
            model ? model.name : 'No model selected');
        
        // Model type
        this.updateElement('#selected-model-type, .selected-model-type', 
            model ? model.type.toUpperCase() : '--');
        
        // Model description
        this.updateElement('#selected-model-desc, .selected-model-desc', 
            model ? (model.description || 'No description') : '--');
        
        // Model path
        this.updateElement('#selected-model-path, .selected-model-path', 
            model ? model.path : '--');
        
        // Input shape
        if (model?.inputShape) {
            const shape = Array.isArray(model.inputShape) 
                ? model.inputShape.join(' × ') 
                : model.inputShape;
            this.updateElement('#model-input-shape, .model-input-shape', shape);
        }
        
        // Precision
        this.updateElement('#model-precision, .model-precision', 
            model?.precision || '--');
        
        // File exists indicator
        const existsEl = document.querySelector('#model-exists, .model-exists');
        if (existsEl && model) {
            existsEl.textContent = model.exists !== false ? '✅ File exists' : '❌ File missing';
            existsEl.className = model.exists !== false ? 'exists-ok' : 'exists-error';
        }
    }
    
    /**
     * Get currently selected model object
     */
    getSelectedModel() {
        if (!this.selectedModelId) return null;
        return this.models.find(m => m.id === this.selectedModelId) || null;
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Model selector
        const modelSelect = document.querySelector('#model-select, .model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => this.handleModelChange(e.target.value));
        }
        
        // Mode selector
        const modeSelect = document.querySelector('#inference-mode, .inference-mode-select');
        if (modeSelect) {
            modeSelect.addEventListener('change', (e) => this.handleModeChange(e.target.value));
        }
        
        // Run Benchmark button
        const runBtn = document.querySelector('#run-inference-btn, .run-inference-btn, [data-action="run-benchmark"]');
        if (runBtn) {
            runBtn.addEventListener('click', () => this.runBenchmark());
        }
        
        // Browse model button
        const browseBtn = document.querySelector('#browse-model-btn, .browse-model-btn');
        if (browseBtn) {
            browseBtn.addEventListener('click', () => this.browseModel());
        }
        
        // Quick mode buttons
        document.querySelectorAll('[data-inference-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.currentTarget.dataset.inferenceMode;
                this.handleModeChange(mode);
                this.runBenchmark();
            });
        });
        
        // Quick model buttons
        document.querySelectorAll('[data-model-id]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modelId = e.currentTarget.dataset.modelId;
                this.handleModelChange(modelId);
            });
        });
    }
    
    /**
     * Handle model selection change
     */
    async handleModelChange(modelId) {
        console.log(`📦 Selecting model: ${modelId}`);
        
        try {
            const result = await window.electronAPI.inference.selectModel(modelId);
            if (result.success) {
                this.selectedModelId = modelId;
                this.updateModelInfo();
                
                // Auto-switch mode based on model type
                const model = this.getSelectedModel();
                if (model) {
                    const newMode = model.type === 'onnx' ? 'onnx' : 'tensorrt';
                    if (this.selectedMode !== newMode && this.selectedMode !== 'cuda' && this.selectedMode !== 'mock') {
                        this.handleModeChange(newMode);
                    }
                }
            } else {
                console.error('Failed to select model:', result.error);
            }
        } catch (error) {
            console.error('Error selecting model:', error);
        }
    }
    
    /**
     * Handle mode change
     */
    async handleModeChange(mode) {
        console.log(`🔧 Setting mode: ${mode}`);
        this.selectedMode = mode;
        
        try {
            await window.electronAPI.inference.setMode(mode);
            
            // Update mode selector UI
            const modeSelect = document.querySelector('#inference-mode, .inference-mode-select');
            if (modeSelect) {
                modeSelect.value = mode;
            }
            
            // Show/hide model selection based on mode
            const modelSection = document.querySelector('#model-selection, .model-selection');
            if (modelSection) {
                const needsModel = mode === 'tensorrt' || mode === 'onnx';
                modelSection.style.display = needsModel ? 'block' : 'none';
            }
            
        } catch (error) {
            console.error('Error setting mode:', error);
        }
    }
    
    /**
     * Browse for a model file
     */
    async browseModel() {
        try {
            const type = this.selectedMode === 'onnx' ? 'onnx' : 'engine';
            const result = await window.electronAPI.inference.browseModel(type);
            
            if (result.success && result.model) {
                console.log('📂 Added model:', result.model);
                
                // Reload models
                await this.loadModels();
                
                // Select the new model
                if (result.model.id) {
                    this.handleModelChange(result.model.id);
                }
            }
        } catch (error) {
            console.error('Failed to browse for model:', error);
            this.showError('Failed to add model file');
        }
    }
    
    /**
     * Run inference benchmark
     */
    async runBenchmark() {
        if (this.isRunning) {
            console.log('⏳ Benchmark already running');
            return;
        }
        
        // Validate model selection for modes that need it
        if ((this.selectedMode === 'tensorrt' || this.selectedMode === 'onnx') && !this.selectedModelId) {
            this.showError('Please select a model first');
            return;
        }
        
        const model = this.getSelectedModel();
        if (model && model.exists === false) {
            this.showError('Selected model file is missing');
            return;
        }
        
        this.isRunning = true;
        this.updateStatus('running');
        this.updateElement('#inference-status-text, .inference-status-text', 
            `Running ${this.selectedMode.toUpperCase()} benchmark...`);
        
        // Disable run button
        const runBtn = document.querySelector('#run-inference-btn, .run-inference-btn');
        if (runBtn) {
            runBtn.disabled = true;
            runBtn.innerHTML = '<span class="spinner"></span> Running...';
        }
        
        try {
            console.log(`🚀 Running benchmark: mode=${this.selectedMode}, model=${this.selectedModelId}`);
            const startTime = performance.now();
            
            // Call backend - it will use selected model/mode automatically
            const result = await window.electronAPI.inference.run(this.selectedMode);
            
            const elapsedTime = performance.now() - startTime;
            console.log(`✅ Benchmark completed in ${elapsedTime.toFixed(0)}ms`);
            console.log('📊 Result:', result);
            
            this.lastResult = result;
            this.addToHistory(result);
            this.updateUI(result);
            this.updateStatus(result.success ? 'success' : 'error');
            
            // Refresh persistent history display
            await this.refreshBenchmarkHistory();
            
        } catch (error) {
            console.error('❌ Benchmark failed:', error);
            this.updateStatus('error');
            this.updateElement('#inference-status-text, .inference-status-text', 
                `Error: ${error.message}`);
        } finally {
            this.isRunning = false;
            
            // Re-enable run button
            if (runBtn) {
                runBtn.disabled = false;
                runBtn.innerHTML = '▶️ Run Benchmark';
            }
        }
    }
    
    /**
     * Show error message
     */
    showError(message) {
        const errorEl = document.querySelector('#inference-error, .inference-error');
        if (errorEl) {
            errorEl.textContent = message;
            errorEl.style.display = 'block';
            errorEl.classList.add('show');
            setTimeout(() => {
                errorEl.style.display = 'none';
                errorEl.classList.remove('show');
            }, 5000);
        } else {
            console.error('Error:', message);
            // Fallback to status text
            this.updateElement('#inference-status-text, .inference-status-text', `Error: ${message}`);
        }
    }
    
    /**
     * Setup mock mode for browser testing
     */
    setupMockMode() {
        // Populate with mock data
        this.models = [
            { id: 'mock-resnet', name: 'ResNet-50 (Mock)', type: 'engine', exists: true },
            { id: 'mock-squeezenet', name: 'SqueezeNet (Mock)', type: 'onnx', exists: true }
        ];
        this.selectedModelId = 'mock-resnet';
        this.selectedMode = 'mock';
        
        this.populateModelDropdown();
        this.populateModeDropdown();
        this.updateModelInfo();
        
        // Setup event listeners for mock mode
        const runBtn = document.querySelector('#run-inference-btn, .run-inference-btn');
        if (runBtn) {
            runBtn.addEventListener('click', () => this.runMockBenchmark());
        }
        
        const modelSelect = document.querySelector('#model-select, .model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.selectedModelId = e.target.value;
                this.updateModelInfo();
            });
        }
        
        const modeSelect = document.querySelector('#inference-mode, .inference-mode-select');
        if (modeSelect) {
            modeSelect.addEventListener('change', (e) => {
                this.selectedMode = e.target.value;
            });
        }
        
        this.updateStatus('ready');
    }
    
    /**
     * Run mock benchmark for browser testing
     */
    async runMockBenchmark() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.updateStatus('running');
        this.updateElement('#inference-status-text, .inference-status-text', 'Running mock benchmark...');
        
        // Simulate benchmark delay
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 500));
        
        const model = this.getSelectedModel();
        
        const mockResult = {
            success: true,
            mode: this.selectedMode,
            model: model?.name || 'Mock Model',
            precision: 'FP16',
            batchSize: 1,
            inputShape: [1, 3, 224, 224],
            latencyMs: 0.3 + Math.random() * 0.3,
            minLatencyMs: 0.25,
            maxLatencyMs: 0.8,
            p95LatencyMs: 0.6,
            throughputFPS: 2000 + Math.random() * 1500,
            inferenceMemoryMB: 95 + Math.random() * 10,
            device: 'Simulated GPU',
            driver: 'Mock Driver',
            cudaVersion: '12.0',
            tensorrtVersion: '10.0.0',
            computeCapability: '8.9',
            smCount: 128,
            cudaCores: 16384,
            vramTotalMB: 24576,
            vramUsedMB: 4096,
            vramFreeMB: 20480,
            temperatureC: 45,
            powerDrawW: 150,
            powerLimitW: 350,
            gpuClockMHz: 2100,
            memClockMHz: 12000,
            warmupRuns: 10,
            benchmarkRuns: 100,
            timestamp: new Date().toISOString()
        };
        
        this.lastResult = mockResult;
        this.addToHistory(mockResult);
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
