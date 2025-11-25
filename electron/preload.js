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

    // File operations
    saveFile: (data) => ipcRenderer.invoke('save-file', data),
    openFile: () => ipcRenderer.invoke('open-file'),

    // Project operations
    onNewProject: (callback) => ipcRenderer.on('new-project', callback),
    onOpenProject: (callback) => ipcRenderer.on('open-project', (event, path) => callback(path)),

    // Platform detection
    platform: process.platform,
    isElectron: true,
});

console.log('✅ Preload script loaded - AI Forge Studio');
