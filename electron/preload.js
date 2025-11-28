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
