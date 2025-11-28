/**
 * AI Forge Studio - Electron Preload Script
 * Bridge between renderer and main process
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
    // App info
    getAppVersion: () => ipcRenderer.invoke('get-app-version'),
    getAppPath: () => ipcRenderer.invoke('get-app-path'),
    getSystemInfo: () => ipcRenderer.invoke('get-system-info'),
    getRealtimeStats: () => ipcRenderer.invoke('get-realtime-stats'),

    // GPU Metrics (Real data from C++ backend)
    getGpuMetrics: () => ipcRenderer.invoke('get-gpu-metrics'),
    startGpuMonitor: (intervalMs) => ipcRenderer.invoke('start-gpu-monitor', intervalMs),
    stopGpuMonitor: () => ipcRenderer.invoke('stop-gpu-monitor'),
    onGpuMetricsUpdate: (callback) => {
        ipcRenderer.on('gpu-metrics-update', (event, metrics) => callback(metrics));
    },

    // Inference Benchmark (TensorRT / CUDA from C++ backend)
    inference: {
        // Run benchmark with optional mode and options
        // If mode is null, uses the currently selected mode
        run: (mode = null, options = {}) => ipcRenderer.invoke('inference-run-benchmark', mode, options),
        runBenchmark: (mode = null, options = {}) => ipcRenderer.invoke('inference-run-benchmark', mode, options),
        
        // Get all available models with selection state
        getModels: () => ipcRenderer.invoke('inference-get-models'),
        
        // Select a model by ID
        selectModel: (modelId) => ipcRenderer.invoke('inference-select-model-by-id', modelId),
        
        // Set the benchmark mode (cuda, mock, tensorrt, onnx)
        setMode: (mode) => ipcRenderer.invoke('inference-set-mode', mode),
        
        // Browse for model file and add to list
        browseModel: (type = 'onnx') => ipcRenderer.invoke('inference-browse-model', type),
        
        // Add a model programmatically
        addModel: (model) => ipcRenderer.invoke('inference-add-model', model),
        
        // Get supported modes
        getModes: () => ipcRenderer.invoke('inference-get-modes'),
        
        // Get current status (includes selected model/mode)
        getStatus: () => ipcRenderer.invoke('inference-get-status'),
        
        // Get last benchmark result
        getLastResult: () => ipcRenderer.invoke('inference-get-last-result'),
        
        // Check if inference backend is available
        isAvailable: () => ipcRenderer.invoke('inference-is-available'),
        
        // Legacy: list models
        listModels: () => ipcRenderer.invoke('inference-get-models'),
    },

    // File operations
    saveFile: (data) => ipcRenderer.invoke('save-file', data),
    openFile: () => ipcRenderer.invoke('open-file'),

    // Project operations
    onNewProject: (callback) => ipcRenderer.on('new-project', callback),
    onOpenProject: (callback) => ipcRenderer.on('open-project', (event, path) => callback(path)),

    // Navigation
    navigateTo: (pageName) => ipcRenderer.invoke('navigate-to', pageName),
    getCurrentPage: () => ipcRenderer.invoke('get-current-page'),

    // Window controls
    minimizeWindow: () => ipcRenderer.invoke('window-minimize'),
    maximizeWindow: () => ipcRenderer.invoke('window-maximize'),
    closeWindow: () => ipcRenderer.invoke('window-close'),

    // Platform detection
    platform: process.platform,
    isElectron: true,
});

console.log('✅ Preload script loaded - AI Forge Studio');
