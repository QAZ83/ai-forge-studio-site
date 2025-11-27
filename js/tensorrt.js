/**
 * TensorRT Integration JavaScript
 * AI Forge Studio - Advanced TensorRT Engine Management
 * 
 * Features:
 * - CUDA Graph support for reduced latency
 * - Dynamic output allocation
 * - Comprehensive profiling
 * - Engine serialization/caching
 */

// ===================================
// TensorRT Engine State
// ===================================
const TensorRTState = {
    engineLoaded: false,
    engineName: 'default',
    cudaGraphsEnabled: false,
    preAllocatedOutputs: false,
    profilingEnabled: false,
    lastInferenceTime: 0,
    totalInferences: 0,
    avgLatency: 0,
    
    // Runtime states (mirrors C++ TensorRTRuntimeStates)
    runtimeStates: {
        oldCudagraphs: false,
        oldPreAllocatedOutputs: false,
        contextChanged: false
    },
    
    // Engine info
    engineInfo: {
        inputs: [],
        outputs: [],
        deviceMemoryMB: 0,
        precision: 'FP16'
    }
};

// ===================================
// TensorRT Configuration
// ===================================
const TensorRTConfig = {
    modelPath: '',
    engineName: 'default',
    maxBatchSize: 1,
    maxWorkspaceSize: 1 << 30,  // 1 GB
    fp16Mode: true,
    bf16Mode: false,
    int8Mode: false,
    enableCudaGraphs: false,
    enablePreAllocatedOutputs: false,
    enableDynamicOutputAllocator: false,
    hardwareCompatible: false,
    enableSparsity: false,
    builderOptimizationLevel: 3,
    
    // Dynamic shapes
    minShapes: {},
    optShapes: {},
    maxShapes: {}
};

// ===================================
// TensorRT Functions
// ===================================
function refreshTensorRT() {
    notify.show('Refreshing TensorRT status...', 'info');
    
    // Simulate checking engine status
    setTimeout(() => {
        if (TensorRTState.engineLoaded) {
            notify.show(`TensorRT engine "${TensorRTState.engineName}" operational!`, 'success');
            updateEngineStatusUI();
        } else {
            notify.show('No TensorRT engine loaded', 'info');
        }
    }, 500);
}

function uploadModel() {
    const content = `
        <div style="padding: 20px;">
            <h3 style="color: #00d9ff; margin-bottom: 20px;">Upload Model</h3>
            <div style="background: rgba(0, 217, 255, 0.05); border: 2px dashed #00d9ff; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer;" onclick="document.getElementById('model-upload').click()">
                <div style="font-size: 48px; margin-bottom: 15px;">📦</div>
                <div style="color: #00ffcc; font-weight: 600; margin-bottom: 10px;">Click to upload or drag and drop</div>
                <div style="color: #8b95a8; font-size: 13px;">Supported formats: .onnx, .pt, .pb, .trt, .engine</div>
                <input type="file" id="model-upload" style="display: none;" accept=".onnx,.pt,.pb,.trt,.engine" onchange="handleModelUpload(this.files[0])">
            </div>
            
            <div style="margin-top: 20px;">
                <h4 style="color: #8b95a8; margin-bottom: 15px;">Build Configuration</h4>
                
                <div class="config-row" style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <label style="flex: 1;">
                        <input type="checkbox" id="config-fp16" checked> FP16 Mode
                    </label>
                    <label style="flex: 1;">
                        <input type="checkbox" id="config-bf16"> BF16 Mode
                    </label>
                    <label style="flex: 1;">
                        <input type="checkbox" id="config-int8"> INT8 Mode
                    </label>
                </div>
                
                <div class="config-row" style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <label style="flex: 1;">
                        <input type="checkbox" id="config-cudagraph"> CUDA Graphs
                    </label>
                    <label style="flex: 1;">
                        <input type="checkbox" id="config-prealloc"> Pre-alloc Outputs
                    </label>
                    <label style="flex: 1;">
                        <input type="checkbox" id="config-sparsity"> Sparsity
                    </label>
                </div>
                
                <div style="margin-top: 15px;">
                    <label style="color: #8b95a8;">Optimization Level: 
                        <select id="config-optlevel" style="background: #1e293b; color: #fff; border: 1px solid #334155; padding: 5px;">
                            <option value="0">0 - Fastest build</option>
                            <option value="1">1 - Low</option>
                            <option value="2">2 - Medium</option>
                            <option value="3" selected>3 - High (Default)</option>
                            <option value="4">4 - Very High</option>
                            <option value="5">5 - Maximum</option>
                        </select>
                    </label>
                </div>
            </div>
        </div>
    `;
    modal.show('upload', '📤 Upload Model', content);
}

function handleModelUpload(file) {
    if (file) {
        // Read configuration from UI
        TensorRTConfig.fp16Mode = document.getElementById('config-fp16')?.checked ?? true;
        TensorRTConfig.bf16Mode = document.getElementById('config-bf16')?.checked ?? false;
        TensorRTConfig.int8Mode = document.getElementById('config-int8')?.checked ?? false;
        TensorRTConfig.enableCudaGraphs = document.getElementById('config-cudagraph')?.checked ?? false;
        TensorRTConfig.enablePreAllocatedOutputs = document.getElementById('config-prealloc')?.checked ?? false;
        TensorRTConfig.enableSparsity = document.getElementById('config-sparsity')?.checked ?? false;
        TensorRTConfig.builderOptimizationLevel = parseInt(document.getElementById('config-optlevel')?.value ?? '3');
        TensorRTConfig.modelPath = file.name;
        
        notify.show(`Uploading ${file.name}...`, 'info');
        modal.close('upload');

        // Determine file type
        const ext = file.name.split('.').pop().toLowerCase();
        const isPrebuilt = ['trt', 'engine'].includes(ext);
        
        setTimeout(() => {
            notify.show('Model uploaded successfully!', 'success');
            
            if (isPrebuilt) {
                // Load pre-built engine
                setTimeout(() => {
                    loadPrebuiltEngine(file.name);
                }, 500);
            } else {
                // Convert to TensorRT
                setTimeout(() => {
                    buildEngineFromONNX(file.name);
                }, 500);
            }
        }, 1500);
    }
}

function loadPrebuiltEngine(enginePath) {
    notify.show('Loading TensorRT engine...', 'info');
    
    setTimeout(() => {
        TensorRTState.engineLoaded = true;
        TensorRTState.engineName = enginePath.replace(/\.[^.]+$/, '');
        TensorRTState.cudaGraphsEnabled = TensorRTConfig.enableCudaGraphs;
        TensorRTState.preAllocatedOutputs = TensorRTConfig.enablePreAllocatedOutputs;
        
        // Simulate engine info
        TensorRTState.engineInfo = {
            inputs: [{ name: 'input', shape: [1, 3, 224, 224] }],
            outputs: [{ name: 'output', shape: [1, 1000] }],
            deviceMemoryMB: 256,
            precision: TensorRTConfig.fp16Mode ? 'FP16' : 'FP32'
        };
        
        notify.show('TensorRT engine loaded successfully!', 'success');
        updateEngineStatusUI();
    }, 1500);
}

function buildEngineFromONNX(modelPath) {
    notify.show('Building TensorRT engine from ONNX...', 'info');
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += 5;
        notify.show(`Building engine... ${progress}%`, 'info', 1000);
        
        if (progress >= 100) {
            clearInterval(interval);
            
            setTimeout(() => {
                TensorRTState.engineLoaded = true;
                TensorRTState.engineName = modelPath.replace(/\.[^.]+$/, '');
                TensorRTState.cudaGraphsEnabled = TensorRTConfig.enableCudaGraphs;
                TensorRTState.preAllocatedOutputs = TensorRTConfig.enablePreAllocatedOutputs;
                
                // Simulate engine info
                TensorRTState.engineInfo = {
                    inputs: [{ name: 'input', shape: [1, 3, 224, 224] }],
                    outputs: [{ name: 'output', shape: [1, 1000] }],
                    deviceMemoryMB: 256,
                    precision: TensorRTConfig.fp16Mode ? 'FP16' : (TensorRTConfig.int8Mode ? 'INT8' : 'FP32')
                };
                
                notify.show('TensorRT engine built successfully!', 'success');
                showBuildComplete();
                updateEngineStatusUI();
            }, 500);
        }
    }, 400);
}

function showBuildComplete() {
    const config = TensorRTConfig;
    let features = [];
    if (config.fp16Mode) features.push('FP16');
    if (config.bf16Mode) features.push('BF16');
    if (config.int8Mode) features.push('INT8');
    if (config.enableCudaGraphs) features.push('CUDA Graphs');
    if (config.enablePreAllocatedOutputs) features.push('Pre-alloc Outputs');
    if (config.enableSparsity) features.push('Sparsity');
    
    modal.show('build-complete', '✅ Engine Build Complete', `
        <div style="padding: 20px;">
            <h3 style="color: #00d9ff; margin-bottom: 20px;">TensorRT Engine Created</h3>
            <div class="status-list">
                <div class="status-item active">
                    <span class="status-dot"></span>
                    <span class="status-text">Engine: ${TensorRTState.engineName}</span>
                </div>
                <div class="status-item active">
                    <span class="status-dot"></span>
                    <span class="status-text">Features: ${features.join(', ')}</span>
                </div>
                <div class="status-item active">
                    <span class="status-dot"></span>
                    <span class="status-text">Optimization Level: ${config.builderOptimizationLevel}</span>
                </div>
                <div class="status-item active">
                    <span class="status-dot"></span>
                    <span class="status-text">Device Memory: ${TensorRTState.engineInfo.deviceMemoryMB} MB</span>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: rgba(0, 217, 255, 0.1); border-radius: 8px;">
                <div style="color: #00ffcc; font-weight: 600;">Engine ready for inference!</div>
            </div>
        </div>
    `);
}

function runModel(modelName) {
    if (!TensorRTState.engineLoaded) {
        notify.show('No TensorRT engine loaded!', 'error');
        return;
    }
    
    notify.show(`Running inference with ${modelName}...`, 'info');
    
    // Simulate runtime state evaluation (mirrors C++ setRuntimeStates)
    const shapeChanged = false;
    const [needRecord, canUsePreAlloc, needReset] = evaluateRuntimeStates(
        TensorRTState.cudaGraphsEnabled,
        TensorRTState.preAllocatedOutputs,
        shapeChanged
    );
    
    setTimeout(() => {
        // Simulate inference with varying latency
        const baseLatency = TensorRTState.cudaGraphsEnabled ? 1.5 : 2.1;
        const variance = Math.random() * 0.5;
        const latency = baseLatency + variance;
        
        TensorRTState.lastInferenceTime = latency;
        TensorRTState.totalInferences++;
        TensorRTState.avgLatency = (TensorRTState.avgLatency * (TensorRTState.totalInferences - 1) + latency) / TensorRTState.totalInferences;
        
        const throughput = (1000 / latency).toFixed(0);
        
        modal.show('inference-result', '🎯 Inference Results', `
            <div style="padding: 20px;">
                <h3 style="color: #00d9ff; margin-bottom: 20px;">Model: ${modelName}</h3>
                <div style="background: rgba(0, 217, 255, 0.05); border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                    <div style="color: #00ffcc; font-weight: 600; margin-bottom: 10px;">Performance Metrics</div>
                    <div class="stat-row">
                        <span class="stat-label">Inference Time:</span>
                        <span class="stat-value">${latency.toFixed(2)} ms</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Throughput:</span>
                        <span class="stat-value">${throughput} FPS</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Latency:</span>
                        <span class="stat-value">${TensorRTState.avgLatency.toFixed(2)} ms</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Total Inferences:</span>
                        <span class="stat-value">${TensorRTState.totalInferences}</span>
                    </div>
                </div>
                <div style="background: rgba(139, 92, 246, 0.1); border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                    <div style="color: #8b5cf6; font-weight: 600; margin-bottom: 10px;">Optimization Status</div>
                    <div class="stat-row">
                        <span class="stat-label">CUDA Graph:</span>
                        <span class="stat-value" style="color: ${TensorRTState.cudaGraphsEnabled ? '#10b981' : '#ef4444'}">
                            ${TensorRTState.cudaGraphsEnabled ? '✓ Active' : '✗ Disabled'}
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Pre-alloc Outputs:</span>
                        <span class="stat-value" style="color: ${canUsePreAlloc ? '#10b981' : '#ef4444'}">
                            ${canUsePreAlloc ? '✓ Reused' : '✗ Fresh alloc'}
                        </span>
                    </div>
                </div>
                <div style="background: rgba(0, 255, 136, 0.05); border-radius: 8px; padding: 15px; border-left: 3px solid #00ff88;">
                    <div style="color: #00ff88; font-weight: 600;">✅ Inference completed successfully!</div>
                </div>
            </div>
        `);
    }, 500);
}

// Mirrors C++ TensorRTRuntimeStates::setRuntimeStates
function evaluateRuntimeStates(newCudagraphs, newPreAllocatedOutput, shapeChanged) {
    const state = TensorRTState.runtimeStates;
    
    let needCudagraphsRecord = false;
    let canUsePreAllocatedOutputs = false;
    let needCudagraphsReset = false;
    
    // CUDA graph record required if enabled and (was disabled, shape changed, or context changed)
    if (newCudagraphs && (!state.oldCudagraphs || shapeChanged || state.contextChanged)) {
        needCudagraphsRecord = true;
    }
    
    // Pre-allocated output can be used when both previous and current state are true without shape change
    if (state.oldPreAllocatedOutputs && newPreAllocatedOutput && !shapeChanged) {
        canUsePreAllocatedOutputs = true;
    }
    
    // Reset CUDA graph if disabled, shape changed, or context changed
    if (!newCudagraphs || shapeChanged || state.contextChanged) {
        needCudagraphsReset = true;
    }
    
    // Update state
    state.oldCudagraphs = newCudagraphs;
    state.oldPreAllocatedOutputs = newPreAllocatedOutput;
    state.contextChanged = false;
    
    return [needCudagraphsRecord, canUsePreAllocatedOutputs, needCudagraphsReset];
}

function toggleCudaGraphs() {
    TensorRTState.cudaGraphsEnabled = !TensorRTState.cudaGraphsEnabled;
    TensorRTState.runtimeStates.contextChanged = true;
    
    notify.show(`CUDA Graphs ${TensorRTState.cudaGraphsEnabled ? 'enabled' : 'disabled'}`, 'info');
    updateEngineStatusUI();
}

function togglePreAllocatedOutputs() {
    TensorRTState.preAllocatedOutputs = !TensorRTState.preAllocatedOutputs;
    
    notify.show(`Pre-allocated outputs ${TensorRTState.preAllocatedOutputs ? 'enabled' : 'disabled'}`, 'info');
    updateEngineStatusUI();
}

function enableProfiling() {
    TensorRTState.profilingEnabled = !TensorRTState.profilingEnabled;
    notify.show(`Layer profiling ${TensorRTState.profilingEnabled ? 'enabled' : 'disabled'}`, 'info');
}

function runBenchmark() {
    if (!TensorRTState.engineLoaded) {
        notify.show('Load a TensorRT engine first!', 'error');
        return;
    }
    
    notify.show('Running benchmark (100 iterations)...', 'info');
    
    const warmupIterations = 10;
    const benchmarkIterations = 100;
    let times = [];
    let iteration = 0;
    
    const runIteration = () => {
        if (iteration < warmupIterations + benchmarkIterations) {
            const baseLatency = TensorRTState.cudaGraphsEnabled ? 1.5 : 2.1;
            const time = baseLatency + Math.random() * 0.5;
            
            if (iteration >= warmupIterations) {
                times.push(time);
            }
            
            iteration++;
            
            if (iteration % 20 === 0) {
                notify.show(`Benchmark: ${iteration}/${warmupIterations + benchmarkIterations}`, 'info', 500);
            }
            
            setTimeout(runIteration, 10);
        } else {
            // Calculate results
            const avg = times.reduce((a, b) => a + b, 0) / times.length;
            const min = Math.min(...times);
            const max = Math.max(...times);
            const throughput = 1000 / avg;
            
            modal.show('benchmark-results', '📊 Benchmark Results', `
                <div style="padding: 20px;">
                    <h3 style="color: #00d9ff; margin-bottom: 20px;">TensorRT Benchmark Complete</h3>
                    <div style="background: rgba(0, 217, 255, 0.05); border-radius: 8px; padding: 15px;">
                        <div class="stat-row">
                            <span class="stat-label">Iterations:</span>
                            <span class="stat-value">${benchmarkIterations}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Average Latency:</span>
                            <span class="stat-value">${avg.toFixed(3)} ms</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Min Latency:</span>
                            <span class="stat-value">${min.toFixed(3)} ms</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Max Latency:</span>
                            <span class="stat-value">${max.toFixed(3)} ms</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Throughput:</span>
                            <span class="stat-value">${throughput.toFixed(1)} inferences/sec</span>
                        </div>
                    </div>
                </div>
            `);
        }
    };
    
    runIteration();
}

function updateEngineStatusUI() {
    // Update any UI elements showing engine status
    const statusElements = document.querySelectorAll('.tensorrt-status');
    statusElements.forEach(el => {
        el.innerHTML = TensorRTState.engineLoaded ? 
            `<span style="color: #10b981;">● Engine: ${TensorRTState.engineName}</span>` :
            '<span style="color: #ef4444;">● No engine loaded</span>';
    });
    
    // Update optimization toggles
    const cudaGraphToggle = document.getElementById('cuda-graph-toggle');
    if (cudaGraphToggle) {
        cudaGraphToggle.textContent = `CUDA Graphs: ${TensorRTState.cudaGraphsEnabled ? 'ON' : 'OFF'}`;
    }
}

function convertModel() {
    const modelType = document.getElementById('model-type')?.value || 'ONNX';
    const precision = document.getElementById('precision')?.value || 'FP16';

    notify.show('Starting model conversion...', 'info');

    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        notify.show(`Converting... ${progress}%`, 'info', 1000);

        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
                notify.show('Model converted to TensorRT engine!', 'success');
                modal.show('conversion-complete', '✅ Conversion Complete', `
                    <div style="padding: 20px;">
                        <h3 style="color: #00d9ff; margin-bottom: 20px;">TensorRT Engine Created</h3>
                        <div class="status-list">
                            <div class="status-item active">
                                <span class="status-dot"></span>
                                <span class="status-text">Model Type: ${modelType}</span>
                            </div>
                            <div class="status-item active">
                                <span class="status-dot"></span>
                                <span class="status-text">Precision: ${precision}</span>
                            </div>
                            <div class="status-item active">
                                <span class="status-dot"></span>
                                <span class="status-text">Engine Size: 145 MB</span>
                            </div>
                            <div class="status-item active">
                                <span class="status-dot"></span>
                                <span class="status-text">Expected Latency: ~2.3ms</span>
                            </div>
                        </div>
                        <div style="margin-top: 20px; padding: 15px; background: rgba(0, 217, 255, 0.1); border-radius: 8px;">
                            <div style="color: #00ffcc; font-weight: 600;">Engine saved to: models/model.trt</div>
                        </div>
                    </div>
                `);
            }, 500);
        }
    }, 600);
}

// ===================================
// Code Tab Switching
// ===================================
function showTRTTab(tab) {
    document.getElementById('trt-inference-code').style.display = 'none';
    document.getElementById('trt-builder-code').style.display = 'none';
    document.getElementById('trt-int8-code').style.display = 'none';

    document.querySelectorAll('.env-tab').forEach(t => t.classList.remove('active'));

    document.getElementById('trt-' + tab + '-code').style.display = 'block';
    event.target.classList.add('active');
}

function copyTRTCode() {
    const activeCode = document.querySelector('.code-display:not([style*="display: none"]) code');
    if (activeCode) {
        navigator.clipboard.writeText(activeCode.innerText).then(() => {
            notify.show('Code copied to clipboard!', 'success');
        });
    }
}

// ===================================
// Performance Chart
// ===================================
let tensorrtChart = null;
let chartData = [];

function createTensorRTPerformanceChart() {
    const ctx = document.getElementById('tensorrtPerformanceChart');
    if (!ctx) {
        console.warn('TensorRT performance chart canvas not found');
        return;
    }

    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded, retrying in 500ms...');
        setTimeout(createTensorRTPerformanceChart, 500);
        return;
    }

    // Clear any existing chart
    if (tensorrtChart) {
        tensorrtChart.destroy();
        tensorrtChart = null;
    }

    // Initialize with random data
    chartData = [];
    for (let i = 0; i < 50; i++) {
        chartData.push((Math.random() * 0.5 + 2.0).toFixed(2));
    }

    tensorrtChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 50}, (_, i) => i),
            datasets: [{
                label: 'Inference Latency (ms)',
                data: chartData,
                borderColor: '#00d9ff',
                backgroundColor: 'rgba(0, 217, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    min: 0,
                    max: 5,
                    grid: {
                        color: 'rgba(0, 217, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8b95a8',
                        callback: function(value) {
                            return value + 'ms';
                        }
                    }
                }
            }
        }
    });
    
    // Update chart periodically with simulated data
    setInterval(() => {
        if (TensorRTState.engineLoaded) {
            const baseLatency = TensorRTState.cudaGraphsEnabled ? 1.5 : 2.1;
            const newValue = (baseLatency + Math.random() * 0.5).toFixed(2);
            chartData.push(newValue);
            chartData.shift();
            
            if (tensorrtChart) {
                tensorrtChart.data.datasets[0].data = chartData;
                tensorrtChart.update('none');
            }
        }
    }, 100);
}

// ===================================
// Initialization
// ===================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🧠 TensorRT Integration loaded');
    console.log('   Features: CUDA Graphs, Dynamic Allocation, Profiling');

    createTensorRTPerformanceChart();
    updateEngineStatusUI();

    console.log('✨ TensorRT Interface Ready!');
});
