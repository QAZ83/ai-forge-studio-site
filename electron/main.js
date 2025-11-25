/**
 * AI Forge Studio - Electron Main Process
 * Desktop Application Entry Point
 * Designed by: M.3R3
 */

const { app, BrowserWindow, Menu, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');

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
    mainWindow = new BrowserWindow({
        width: CONFIG.width,
        height: CONFIG.height,
        minWidth: CONFIG.minWidth,
        minHeight: CONFIG.minHeight,
        show: false, // Don't show until ready
        backgroundColor: '#0a0e1a',
        icon: path.join(__dirname, '../assets/icon.png'),
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: true,
            enableRemoteModule: false,
            preload: path.join(__dirname, 'preload.js'),
        },
        autoHideMenuBar: false,
        frame: true,
        titleBarStyle: 'default',
    });

    // Load the main page
    mainWindow.loadFile(path.join(__dirname, '../index.html'));

    // Show window when ready
    mainWindow.once('ready-to-show', () => {
        setTimeout(() => {
            if (splashWindow) {
                splashWindow.close();
            }
            mainWindow.show();
            mainWindow.focus();

            // Open DevTools in development mode
            if (process.env.NODE_ENV === 'development') {
                mainWindow.webContents.openDevTools();
            }
        }, 2000);
    });

    // Handle window close
    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Create application menu
    createMenu();

    // Handle external links
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
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
    createSplashWindow();
    setTimeout(createMainWindow, 500);

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
    return {
        platform: process.platform,
        arch: process.arch,
        cpus: os.cpus().length,
        totalMemory: (os.totalmem() / 1024 / 1024 / 1024).toFixed(2) + ' GB',
        freeMemory: (os.freemem() / 1024 / 1024 / 1024).toFixed(2) + ' GB',
        version: process.version,
        electronVersion: process.versions.electron,
        chromeVersion: process.versions.chrome,
    };
});

console.log('🚀 AI Forge Studio - Electron App Started');
console.log(`📦 Version: ${CONFIG.version}`);
console.log(`👨‍💻 Developer: ${CONFIG.developer}`);
