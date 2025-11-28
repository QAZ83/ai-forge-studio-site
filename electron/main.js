/**
 * AI Forge Studio - Electron Main Process
 * Desktop Application Entry Point
 * Designed by: M.3R3
 */

const { app, BrowserWindow, Menu, ipcMain, dialog, shell, globalShortcut, nativeTheme } = require('electron');
const path = require('path');
const fs = require('fs');

// GPU Monitor for real GPU metrics
const { gpuMonitor } = require('./gpu-monitor');

// Inference Monitor for TensorRT benchmarks
const { inferenceMonitor } = require('./inference-monitor');

// Disable hardware acceleration if issues detected
app.disableHardwareAcceleration();

let mainWindow;
let splashWindow;

// Application configuration
const CONFIG = {
    appName: 'AI Forge Studio',
    version: '1.0.0',
    developer: 'M.3R3',
    width: 1920,
    height: 1080,
    minWidth: 1280,
    minHeight: 720,
};

// Window state management
let windowState = {
    isMaximized: false,
    bounds: null
};

// Navigation pages
const PAGES = {
    index: 'index.html',
    dashboard: 'dashboard.html',
    cuda: 'cuda-integration.html',
    tensorrt: 'tensorrt.html',
    vulkan: 'vulkan.html',
    inference: 'inference.html',
    download: 'download.html'
};

/**
 * Create splash screen window
 */
function createSplashWindow() {
    splashWindow = new BrowserWindow({
        width: 500,
        height: 300,
        transparent: true,
        frame: false,
        alwaysOnTop: true,
        center: true,
        resizable: false,
        skipTaskbar: true,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
        },
    });

    splashWindow.loadFile(path.join(__dirname, '../splash.html'));
    splashWindow.on('closed', () => (splashWindow = null));
}

/**
 * Create main application window
 */
function createMainWindow() {
    // Load saved window state
    loadWindowState();

    mainWindow = new BrowserWindow({
        width: windowState.bounds?.width || CONFIG.width,
        height: windowState.bounds?.height || CONFIG.height,
        x: windowState.bounds?.x,
        y: windowState.bounds?.y,
        minWidth: CONFIG.minWidth,
        minHeight: CONFIG.minHeight,
        show: false, // Don't show until ready
        backgroundColor: '#0a0e1a',
        icon: path.join(__dirname, '../assets/icon.png'),
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            enableRemoteModule: false,
            preload: path.join(__dirname, 'preload.js'),
            webSecurity: true,
        },
        autoHideMenuBar: false,
        frame: true,
        titleBarStyle: 'default',
    });

    // Restore maximized state
    if (windowState.isMaximized) {
        mainWindow.maximize();
    }

    // Load the main page
    mainWindow.loadFile(path.join(__dirname, '../index.html'));

    // Show window when ready
    mainWindow.once('ready-to-show', () => {
        // No delay - show immediately
        if (splashWindow) {
            splashWindow.close();
        }
        mainWindow.show();
        mainWindow.focus();

        // Open DevTools in development mode
        if (process.env.NODE_ENV === 'development') {
            mainWindow.webContents.openDevTools();
        }
    });

    // Save window state on close
    mainWindow.on('close', () => {
        saveWindowState();
    });

    // Handle window close
    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Track window state changes
    mainWindow.on('maximize', () => {
        windowState.isMaximized = true;
    });

    mainWindow.on('unmaximize', () => {
        windowState.isMaximized = false;
    });

    // Create application menu
    createMenu();

    // Handle external links
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });

    // Handle navigation errors
    mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
        console.error(`Failed to load: ${validatedURL} - ${errorDescription}`);
        // Fallback to index if page fails to load
        if (!validatedURL.includes('index.html')) {
            mainWindow.loadFile(path.join(__dirname, '../index.html'));
        }
    });
}

/**
 * Create application menu
 */
function createMenu() {
    const template = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'New Project',
                    accelerator: 'CmdOrCtrl+N',
                    click: () => {
                        mainWindow.webContents.send('new-project');
                    },
                },
                {
                    label: 'Open Project',
                    accelerator: 'CmdOrCtrl+O',
                    click: async () => {
                        const result = await dialog.showOpenDialog(mainWindow, {
                            properties: ['openDirectory'],
                        });
                        if (!result.canceled) {
                            mainWindow.webContents.send('open-project', result.filePaths[0]);
                        }
                    },
                },
                { type: 'separator' },
                {
                    label: 'Exit',
                    accelerator: 'CmdOrCtrl+Q',
                    click: () => {
                        app.quit();
                    },
                },
            ],
        },
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectAll' },
            ],
        },
        {
            label: 'View',
            submenu: [
                {
                    label: 'Dashboard',
                    click: () => {
                        mainWindow.loadFile(path.join(__dirname, '../dashboard.html'));
                    },
                },
                {
                    label: 'CUDA Integration',
                    click: () => {
                        mainWindow.loadFile(path.join(__dirname, '../cuda-integration.html'));
                    },
                },
                {
                    label: 'TensorRT',
                    click: () => {
                        mainWindow.loadFile(path.join(__dirname, '../tensorrt.html'));
                    },
                },
                {
                    label: 'Vulkan SDK',
                    click: () => {
                        mainWindow.loadFile(path.join(__dirname, '../vulkan.html'));
                    },
                },
                { type: 'separator' },
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' },
            ],
        },
        {
            label: 'Window',
            submenu: [
                { role: 'minimize' },
                { role: 'zoom' },
                { role: 'close' },
            ],
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'Documentation',
                    click: async () => {
                        await shell.openExternal('https://github.com/QAZ83/ai-forge-studio');
                    },
                },
                {
                    label: 'Report Issue',
                    click: async () => {
                        await shell.openExternal('https://github.com/QAZ83/ai-forge-studio/issues');
                    },
                },
                { type: 'separator' },
                {
                    label: 'About',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About AI Forge Studio',
                            message: `${CONFIG.appName}`,
                            detail: `Version: ${CONFIG.version}\nDeveloped by: ${CONFIG.developer}\n\nA powerful AI development studio with GPU acceleration support.`,
                            buttons: ['OK'],
                        });
                    },
                },
            ],
        },
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

/**
 * App ready event
 */
app.whenReady().then(() => {
    // Skip splash for faster startup - directly show main window
    createMainWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createMainWindow();
        }
    });
});

/**
 * Quit when all windows are closed
 */
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

/**
 * IPC Handlers
 */

// Get app version
ipcMain.handle('get-app-version', () => {
    return CONFIG.version;
});

// Get app path
ipcMain.handle('get-app-path', () => {
    return app.getPath('userData');
});

// Save file
ipcMain.handle('save-file', async (event, data) => {
    const result = await dialog.showSaveDialog(mainWindow, {
        title: 'Save File',
        defaultPath: path.join(app.getPath('documents'), 'untitled.txt'),
        filters: [
            { name: 'Text Files', extensions: ['txt'] },
            { name: 'All Files', extensions: ['*'] },
        ],
    });

    if (!result.canceled) {
        try {
            fs.writeFileSync(result.filePath, data, 'utf-8');
            return { success: true, path: result.filePath };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    return { success: false, canceled: true };
});

// Open file
ipcMain.handle('open-file', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openFile'],
        filters: [
            { name: 'Text Files', extensions: ['txt', 'json', 'md'] },
            { name: 'All Files', extensions: ['*'] },
        ],
    });

    if (!result.canceled) {
        try {
            const data = fs.readFileSync(result.filePaths[0], 'utf-8');
            return { success: true, data, path: result.filePaths[0] };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    return { success: false, canceled: true };
});

// System info
ipcMain.handle('get-system-info', () => {
    const os = require('os');
    const cpus = os.cpus();
    
    // Calculate actual CPU usage
    let totalIdle = 0;
    let totalTick = 0;
    cpus.forEach(cpu => {
        for (let type in cpu.times) {
            totalTick += cpu.times[type];
        }
        totalIdle += cpu.times.idle;
    });
    const cpuUsage = Math.round(100 - (100 * totalIdle / totalTick));
    
    return {
        platform: process.platform,
        arch: process.arch,
        cpus: cpus.length,
        cpuModel: cpus[0]?.model || 'Unknown',
        cpuUsage: cpuUsage,
        totalMemory: os.totalmem(),
        freeMemory: os.freemem(),
        usedMemory: os.totalmem() - os.freemem(),
        memoryUsage: Math.round(((os.totalmem() - os.freemem()) / os.totalmem()) * 100),
        totalMemoryGB: (os.totalmem() / 1024 / 1024 / 1024).toFixed(2) + ' GB',
        freeMemoryGB: (os.freemem() / 1024 / 1024 / 1024).toFixed(2) + ' GB',
        usedMemoryGB: ((os.totalmem() - os.freemem()) / 1024 / 1024 / 1024).toFixed(2) + ' GB',
        hostname: os.hostname(),
        uptime: os.uptime(),
        nodeVersion: process.version,
        electronVersion: process.versions.electron,
        chromeVersion: process.versions.chrome,
    };
});

// Real-time system stats
ipcMain.handle('get-realtime-stats', () => {
    const os = require('os');
    const cpus = os.cpus();
    
    // Calculate actual CPU usage
    let totalIdle = 0;
    let totalTick = 0;
    cpus.forEach(cpu => {
        for (let type in cpu.times) {
            totalTick += cpu.times[type];
        }
        totalIdle += cpu.times.idle;
    });
    const cpuUsage = Math.round(100 - (100 * totalIdle / totalTick));
    const memoryUsage = Math.round(((os.totalmem() - os.freemem()) / os.totalmem()) * 100);
    
    return {
        cpuUsage,
        memoryUsage,
        freeMemory: os.freemem(),
        totalMemory: os.totalmem(),
        uptime: os.uptime()
    };
});

// =============================================================================
// GPU Metrics (Real data from C++ backend)
// =============================================================================

// Get GPU metrics (single request)
ipcMain.handle('get-gpu-metrics', async () => {
    return await gpuMonitor.getMetricsAsync();
});

// Start GPU monitoring with interval updates
ipcMain.handle('start-gpu-monitor', (event, intervalMs = 1000) => {
    gpuMonitor.start(intervalMs);
    
    // Send updates to renderer
    gpuMonitor.onUpdate((metrics) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('gpu-metrics-update', metrics);
        }
    });
    
    return { success: true };
});

// Stop GPU monitoring
ipcMain.handle('stop-gpu-monitor', () => {
    gpuMonitor.stop();
    return { success: true };
});

// =============================================================================
// Inference Benchmark (TensorRT / CUDA from C++ backend)
// =============================================================================

// Run inference benchmark
ipcMain.handle('inference-run-benchmark', async (event, mode = 'cuda', options = {}) => {
    console.log(`[Inference] Running benchmark in ${mode} mode...`);
    const result = await inferenceMonitor.runBenchmark(mode, options);
    console.log(`[Inference] Benchmark complete: ${result.throughputFPS} FPS`);
    return result;
});

// Get supported inference modes
ipcMain.handle('inference-get-modes', () => {
    return inferenceMonitor.getSupportedModes();
});

// Get last inference result
ipcMain.handle('inference-get-last-result', () => {
    return inferenceMonitor.getLastResult();
});

// Check if inference is available
ipcMain.handle('inference-is-available', () => {
    return inferenceMonitor.isAvailable();
});

console.log('🚀 AI Forge Studio - Electron App Started');
console.log(`📦 Version: ${CONFIG.version}`);
console.log(`👨‍💻 Developer: ${CONFIG.developer}`);

/**
 * Window state management functions
 */
function loadWindowState() {
    try {
        const statePath = path.join(app.getPath('userData'), 'window-state.json');
        if (fs.existsSync(statePath)) {
            const data = fs.readFileSync(statePath, 'utf-8');
            windowState = JSON.parse(data);
        }
    } catch (error) {
        console.error('Failed to load window state:', error);
    }
}

function saveWindowState() {
    try {
        if (mainWindow) {
            windowState.isMaximized = mainWindow.isMaximized();
            if (!mainWindow.isMaximized()) {
                windowState.bounds = mainWindow.getBounds();
            }
            const statePath = path.join(app.getPath('userData'), 'window-state.json');
            fs.writeFileSync(statePath, JSON.stringify(windowState), 'utf-8');
        }
    } catch (error) {
        console.error('Failed to save window state:', error);
    }
}

/**
 * Navigate to a specific page
 */
function navigateToPage(pageName) {
    const pageFile = PAGES[pageName];
    if (pageFile && mainWindow) {
        mainWindow.loadFile(path.join(__dirname, '..', pageFile));
    }
}

// IPC handler for navigation
ipcMain.handle('navigate-to', (event, pageName) => {
    navigateToPage(pageName);
    return { success: true };
});

// IPC handler for window controls
ipcMain.handle('window-minimize', () => {
    if (mainWindow) mainWindow.minimize();
});

ipcMain.handle('window-maximize', () => {
    if (mainWindow) {
        if (mainWindow.isMaximized()) {
            mainWindow.unmaximize();
        } else {
            mainWindow.maximize();
        }
    }
    return mainWindow?.isMaximized();
});

ipcMain.handle('window-close', () => {
    if (mainWindow) mainWindow.close();
});

// IPC handler for getting current page
ipcMain.handle('get-current-page', () => {
    if (mainWindow) {
        const url = mainWindow.webContents.getURL();
        for (const [key, value] of Object.entries(PAGES)) {
            if (url.includes(value)) {
                return key;
            }
        }
    }
    return 'index';
});
