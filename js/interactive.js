/**
 * AI Forge Studio - Interactive Features
 * Complete interactivity system
 * Author: M.3R3
 */

// ===================================
// Notification System
// ===================================
class NotificationSystem {
    constructor() {
        this.createContainer();
    }

    createContainer() {
        if (!document.getElementById('notification-container')) {
            const container = document.createElement('div');
            container.id = 'notification-container';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                display: flex;
                flex-direction: column;
                gap: 10px;
                pointer-events: none;
            `;
            document.body.appendChild(container);
        }
    }

    show(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;

        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        const colors = {
            success: '#00ff88',
            error: '#ff3366',
            warning: '#ffd500',
            info: '#00d9ff'
        };

        notification.innerHTML = `
            <span style="font-size: 20px; margin-right: 10px;">${icons[type]}</span>
            <span>${message}</span>
        `;

        notification.style.cssText = `
            background: rgba(0, 0, 0, 0.95);
            border: 2px solid ${colors[type]};
            border-radius: 8px;
            padding: 15px 20px;
            color: #fff;
            font-family: 'Rajdhani', sans-serif;
            font-size: 14px;
            display: flex;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0, 217, 255, 0.3);
            pointer-events: all;
            animation: slideIn 0.3s ease-out;
            min-width: 300px;
        `;

        const container = document.getElementById('notification-container');
        container.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }
}

const notify = new NotificationSystem();

// ===================================
// Modal System
// ===================================
class ModalSystem {
    constructor() {
        this.modals = new Map();
        this.createStyles();
    }

    createStyles() {
        if (!document.getElementById('modal-styles')) {
            const style = document.createElement('style');
            style.id = 'modal-styles';
            style.textContent = `
                .modal-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.8);
                    backdrop-filter: blur(10px);
                    z-index: 9999;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    animation: fadeIn 0.3s ease-out;
                }

                .modal-content {
                    background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
                    border: 2px solid #00d9ff;
                    border-radius: 12px;
                    padding: 30px;
                    max-width: 600px;
                    width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 10px 50px rgba(0, 217, 255, 0.3);
                    animation: scaleIn 0.3s ease-out;
                }

                .modal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 1px solid rgba(0, 217, 255, 0.3);
                }

                .modal-header h2 {
                    color: #00d9ff;
                    font-family: 'Orbitron', sans-serif;
                    font-size: 24px;
                    margin: 0;
                }

                .modal-close {
                    background: transparent;
                    border: none;
                    color: #00d9ff;
                    font-size: 28px;
                    cursor: pointer;
                    padding: 0;
                    width: 30px;
                    height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s;
                }

                .modal-close:hover {
                    color: #00ffcc;
                    transform: rotate(90deg);
                }

                .modal-body {
                    color: #e0e6ed;
                    font-family: 'Rajdhani', sans-serif;
                }

                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }

                @keyframes scaleIn {
                    from { transform: scale(0.9); opacity: 0; }
                    to { transform: scale(1); opacity: 1; }
                }

                @keyframes slideIn {
                    from { transform: translateX(400px); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }

                @keyframes slideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(400px); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }
    }

    create(id, title, content) {
        const overlay = document.createElement('div');
        overlay.className = 'modal-overlay';
        overlay.id = `modal-${id}`;

        overlay.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>${title}</h2>
                    <button class="modal-close" onclick="modal.close('${id}')">×</button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        `;

        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.close(id);
            }
        });

        document.body.appendChild(overlay);
        this.modals.set(id, overlay);
    }

    show(id, title, content) {
        this.close(id); // Close if already exists
        this.create(id, title, content);
    }

    close(id) {
        const modal = document.getElementById(`modal-${id}`);
        if (modal) {
            modal.style.animation = 'fadeIn 0.3s ease-out reverse';
            setTimeout(() => {
                modal.remove();
                this.modals.delete(id);
            }, 300);
        }
    }
}

const modal = new ModalSystem();

// ===================================
// Terminal Simulator
// ===================================
class TerminalSimulator {
    constructor() {
        this.history = [];
        this.historyIndex = -1;
        this.currentPath = '/home/ai-forge';
    }

    show() {
        const content = `
            <div class="terminal-simulator" style="background: #000; border-radius: 8px; padding: 20px; font-family: 'Courier New', monospace;">
                <div id="terminal-output" style="color: #00ff88; max-height: 400px; overflow-y: auto; margin-bottom: 10px;">
                    <div style="color: #00d9ff;">AI Forge Studio Terminal v1.0</div>
                    <div style="color: #8b95a8;">Type 'help' for available commands</div>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="color: #00d9ff; margin-right: 5px;">${this.currentPath} $</span>
                    <input type="text" id="terminal-input"
                           style="flex: 1; background: transparent; border: none; color: #00ff88; font-family: 'Courier New', monospace; outline: none; font-size: 14px;"
                           placeholder="Enter command...">
                </div>
            </div>
        `;

        modal.show('terminal', '🖥️ Terminal', content);

        setTimeout(() => {
            const input = document.getElementById('terminal-input');
            if (input) {
                input.focus();
                input.addEventListener('keydown', (e) => this.handleInput(e));
            }
        }, 100);
    }

    handleInput(e) {
        if (e.key === 'Enter') {
            const input = e.target;
            const command = input.value.trim();

            if (command) {
                this.history.push(command);
                this.historyIndex = this.history.length;
                this.executeCommand(command);
                input.value = '';
            }
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (this.historyIndex > 0) {
                this.historyIndex--;
                e.target.value = this.history[this.historyIndex];
            }
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (this.historyIndex < this.history.length - 1) {
                this.historyIndex++;
                e.target.value = this.history[this.historyIndex];
            } else {
                this.historyIndex = this.history.length;
                e.target.value = '';
            }
        }
    }

    executeCommand(command) {
        const output = document.getElementById('terminal-output');
        if (!output) return;

        // Add command to output
        const cmdLine = document.createElement('div');
        cmdLine.style.color = '#8b95a8';
        cmdLine.innerHTML = `<span style="color: #00d9ff;">${this.currentPath} $</span> ${command}`;
        output.appendChild(cmdLine);

        // Process command
        const parts = command.split(' ');
        const cmd = parts[0].toLowerCase();
        const args = parts.slice(1);

        let response = '';

        switch(cmd) {
            case 'help':
                response = `<div style="color: #00ff88;">
                    Available commands:<br>
                    • help - Show this help message<br>
                    • clear - Clear terminal<br>
                    • status - Show system status<br>
                    • gpu - Show GPU information<br>
                    • train - Start training simulation<br>
                    • build - Build project<br>
                    • deploy - Deploy to cloud<br>
                    • ls - List files<br>
                    • pwd - Print working directory
                </div>`;
                break;

            case 'clear':
                output.innerHTML = '';
                return;

            case 'status':
                response = `<div style="color: #00ff88;">
                    System Status:<br>
                    • GPU Usage: 95%<br>
                    • CPU Usage: 12%<br>
                    • Memory: 28GB / 32GB<br>
                    • Temperature: 72°C<br>
                    • Status: <span style="color: #00ffcc;">Online</span>
                </div>`;
                break;

            case 'gpu':
                response = `<div style="color: #00ff88;">
                    GPU Information:<br>
                    • Device: NVIDIA GeForce RTX 4000<br>
                    • CUDA Version: 12.0<br>
                    • VRAM: 28GB / 24GB<br>
                    • Driver Version: 535.129.03<br>
                    • Compute Capability: 8.9
                </div>`;
                break;

            case 'train':
                response = `<div style="color: #ffd500;">
                    Starting training process...<br>
                    <span style="color: #00ff88;">✓ Loading dataset</span><br>
                    <span style="color: #00ff88;">✓ Initializing model</span><br>
                    <span style="color: #00ff88;">✓ Training started</span><br>
                    Epoch 1/100 - Loss: 0.8523
                </div>`;
                break;

            case 'build':
                response = `<div style="color: #00ff88;">
                    Building project...<br>
                    ✓ Compiling sources<br>
                    ✓ Linking libraries<br>
                    ✓ Build completed successfully
                </div>`;
                break;

            case 'deploy':
                response = `<div style="color: #00d9ff;">
                    Deploying to Cloudflare Workers...<br>
                    ✓ Bundling assets<br>
                    ✓ Uploading to cloud<br>
                    ✓ Deployment complete<br>
                    URL: https://ai-forge-studio.workers.dev
                </div>`;
                break;

            case 'ls':
                response = `<div style="color: #00d9ff;">
                    models/  datasets/  scripts/  configs/  outputs/
                </div>`;
                break;

            case 'pwd':
                response = `<div style="color: #00d9ff;">${this.currentPath}</div>`;
                break;

            default:
                response = `<div style="color: #ff3366;">Command not found: ${cmd}</div>`;
        }

        const responseLine = document.createElement('div');
        responseLine.innerHTML = response;
        output.appendChild(responseLine);

        // Scroll to bottom
        output.scrollTop = output.scrollHeight;
    }
}

const terminal = new TerminalSimulator();

// ===================================
// Search System
// ===================================
class SearchSystem {
    constructor() {
        this.searchIndex = [
            { title: 'GPU Accelerated Output', category: 'Visualization', path: 'index.html#gpu-output' },
            { title: 'System Status', category: 'Monitoring', path: 'index.html#system-status' },
            { title: 'Training & Experimentation', category: 'AI', path: 'index.html#training' },
            { title: 'Development Environment', category: 'Development', path: 'index.html#dev-env' },
            { title: 'Performance Dashboard', category: 'Monitoring', path: 'dashboard.html' },
            { title: 'GPU Tracking', category: 'Monitoring', path: 'dashboard.html#gpu-tracking' },
            { title: 'Training Metrics', category: 'AI', path: 'dashboard.html#training-metrics' },
            { title: 'Cloud API Interface', category: 'Deployment', path: 'dashboard.html#cloud-api' }
        ];
    }

    show() {
        const content = `
            <div class="search-container">
                <input type="text" id="search-input"
                       placeholder="Search features, pages, commands..."
                       style="width: 100%; padding: 15px; background: rgba(0, 217, 255, 0.1); border: 2px solid #00d9ff; border-radius: 8px; color: #fff; font-family: 'Rajdhani', sans-serif; font-size: 16px; outline: none; margin-bottom: 20px;">
                <div id="search-results" style="max-height: 400px; overflow-y: auto;"></div>
            </div>
        `;

        modal.show('search', '🔍 Search', content);

        setTimeout(() => {
            const input = document.getElementById('search-input');
            if (input) {
                input.focus();
                input.addEventListener('input', (e) => this.performSearch(e.target.value));
                this.performSearch(''); // Show all initially
            }
        }, 100);
    }

    performSearch(query) {
        const resultsContainer = document.getElementById('search-results');
        if (!resultsContainer) return;

        const filtered = query.trim() === ''
            ? this.searchIndex
            : this.searchIndex.filter(item =>
                item.title.toLowerCase().includes(query.toLowerCase()) ||
                item.category.toLowerCase().includes(query.toLowerCase())
            );

        if (filtered.length === 0) {
            resultsContainer.innerHTML = '<div style="color: #8b95a8; text-align: center; padding: 20px;">No results found</div>';
            return;
        }

        resultsContainer.innerHTML = filtered.map(item => `
            <div class="search-result-item"
                 style="padding: 15px; margin-bottom: 10px; background: rgba(0, 217, 255, 0.05); border-left: 3px solid #00d9ff; border-radius: 4px; cursor: pointer; transition: all 0.3s;"
                 onclick="window.location.href='${item.path}'; modal.close('search');"
                 onmouseover="this.style.background='rgba(0, 217, 255, 0.1)'"
                 onmouseout="this.style.background='rgba(0, 217, 255, 0.05)'">
                <div style="color: #00ffcc; font-weight: 600; margin-bottom: 5px;">${item.title}</div>
                <div style="color: #8b95a8; font-size: 13px;">
                    <span style="background: rgba(0, 217, 255, 0.2); padding: 2px 8px; border-radius: 3px;">${item.category}</span>
                    <span style="margin-left: 10px;">${item.path}</span>
                </div>
            </div>
        `).join('');
    }
}

const search = new SearchSystem();

// ===================================
// Settings System
// ===================================
class SettingsSystem {
    constructor() {
        this.settings = {
            theme: 'dark',
            notifications: true,
            autoSave: true,
            gpuAcceleration: true,
            realTimeUpdates: true,
            chartAnimations: true
        };
        this.loadSettings();
    }

    loadSettings() {
        const saved = localStorage.getItem('aiforge_settings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
        }
    }

    saveSettings() {
        localStorage.setItem('aiforge_settings', JSON.stringify(this.settings));
        notify.show('Settings saved successfully!', 'success');
    }

    show() {
        const content = `
            <div class="settings-container">
                ${this.createSettingToggle('notifications', 'Enable Notifications', 'Receive system notifications and alerts')}
                ${this.createSettingToggle('autoSave', 'Auto Save', 'Automatically save your work')}
                ${this.createSettingToggle('gpuAcceleration', 'GPU Acceleration', 'Use GPU for rendering and computations')}
                ${this.createSettingToggle('realTimeUpdates', 'Real-time Updates', 'Update metrics in real-time')}
                ${this.createSettingToggle('chartAnimations', 'Chart Animations', 'Enable smooth chart animations')}

                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(0, 217, 255, 0.3);">
                    <button onclick="settings.saveSettings()"
                            style="background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%); border: none; color: #000; padding: 12px 30px; border-radius: 6px; font-family: 'Rajdhani', sans-serif; font-size: 16px; font-weight: 600; cursor: pointer; margin-right: 10px;">
                        Save Settings
                    </button>
                    <button onclick="settings.resetSettings()"
                            style="background: transparent; border: 2px solid #ff3366; color: #ff3366; padding: 12px 30px; border-radius: 6px; font-family: 'Rajdhani', sans-serif; font-size: 16px; font-weight: 600; cursor: pointer;">
                        Reset to Default
                    </button>
                </div>
            </div>
        `;

        modal.show('settings', '⚙️ Settings', content);
    }

    createSettingToggle(key, title, description) {
        return `
            <div style="margin-bottom: 25px; padding-bottom: 20px; border-bottom: 1px solid rgba(0, 217, 255, 0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <div>
                        <div style="color: #00ffcc; font-weight: 600; font-size: 16px; margin-bottom: 5px;">${title}</div>
                        <div style="color: #8b95a8; font-size: 13px;">${description}</div>
                    </div>
                    <label class="toggle-switch" style="position: relative; display: inline-block; width: 60px; height: 30px;">
                        <input type="checkbox" id="setting-${key}" ${this.settings[key] ? 'checked' : ''}
                               onchange="settings.updateSetting('${key}', this.checked)"
                               style="opacity: 0; width: 0; height: 0;">
                        <span style="position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #1a1f2e; transition: .4s; border-radius: 30px; border: 2px solid #00d9ff;"></span>
                    </label>
                </div>
            </div>
        `;
    }

    updateSetting(key, value) {
        this.settings[key] = value;
        console.log(`Setting ${key} updated to ${value}`);
    }

    resetSettings() {
        if (confirm('Are you sure you want to reset all settings to default?')) {
            this.settings = {
                theme: 'dark',
                notifications: true,
                autoSave: true,
                gpuAcceleration: true,
                realTimeUpdates: true,
                chartAnimations: true
            };
            localStorage.removeItem('aiforge_settings');
            notify.show('Settings reset to default', 'info');
            modal.close('settings');
            setTimeout(() => this.show(), 300);
        }
    }
}

const settings = new SettingsSystem();

// ===================================
// Project Manager
// ===================================
class ProjectManager {
    constructor() {
        this.currentProject = {
            name: 'Project Omega',
            files: [],
            lastModified: new Date()
        };
    }

    showLoadDialog() {
        const content = `
            <div style="text-align: center; padding: 20px;">
                <div style="margin-bottom: 20px;">
                    <div style="font-size: 60px; margin-bottom: 10px;">📂</div>
                    <div style="color: #00d9ff; font-size: 18px; margin-bottom: 10px;">Load Project</div>
                    <div style="color: #8b95a8;">Select a project file or recent project</div>
                </div>

                <div style="margin: 30px 0;">
                    <input type="file" id="project-file-input" accept=".json,.aiforge"
                           style="display: none;" onchange="projectManager.loadFile(this.files[0])">
                    <button onclick="document.getElementById('project-file-input').click()"
                            style="background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%); border: none; color: #000; padding: 15px 40px; border-radius: 8px; font-family: 'Rajdhani', sans-serif; font-size: 16px; font-weight: 600; cursor: pointer; margin-right: 10px;">
                        Browse Files
                    </button>
                </div>

                <div style="margin-top: 30px; text-align: left;">
                    <div style="color: #00d9ff; margin-bottom: 15px; font-weight: 600;">Recent Projects:</div>
                    ${this.getRecentProjects()}
                </div>
            </div>
        `;

        modal.show('load', '📁 Load Project', content);
    }

    getRecentProjects() {
        const recent = [
            { name: 'Project Omega', date: '2025-11-24', path: '/projects/omega' },
            { name: 'Neural Network v2', date: '2025-11-20', path: '/projects/neural-v2' },
            { name: 'Image Classifier', date: '2025-11-15', path: '/projects/classifier' }
        ];

        return recent.map(project => `
            <div style="padding: 12px; background: rgba(0, 217, 255, 0.05); border-left: 3px solid #00d9ff; border-radius: 4px; margin-bottom: 10px; cursor: pointer; transition: all 0.3s;"
                 onclick="projectManager.loadProject('${project.name}')"
                 onmouseover="this.style.background='rgba(0, 217, 255, 0.1)'"
                 onmouseout="this.style.background='rgba(0, 217, 255, 0.05)'">
                <div style="color: #00ffcc; font-weight: 600;">${project.name}</div>
                <div style="color: #8b95a8; font-size: 13px; margin-top: 5px;">
                    <span>📅 ${project.date}</span>
                    <span style="margin-left: 15px;">📍 ${project.path}</span>
                </div>
            </div>
        `).join('');
    }

    loadFile(file) {
        if (file) {
            notify.show(`Loading ${file.name}...`, 'info');
            setTimeout(() => {
                notify.show('Project loaded successfully!', 'success');
                modal.close('load');
            }, 1000);
        }
    }

    loadProject(name) {
        notify.show(`Loading ${name}...`, 'info');
        setTimeout(() => {
            notify.show('Project loaded successfully!', 'success');
            modal.close('load');
        }, 1000);
    }
}

const projectManager = new ProjectManager();

// ===================================
// Button Handlers Setup
// ===================================
function setupInteractiveButtons() {
    // Search button
    const searchBtn = document.querySelector('.search-btn');
    if (searchBtn) {
        searchBtn.addEventListener('click', () => search.show());
    }

    // Terminal button
    const terminalBtn = document.querySelector('.terminal-btn');
    if (terminalBtn) {
        terminalBtn.addEventListener('click', () => terminal.show());
    }

    // Load button
    const loadBtn = document.querySelector('.load-btn');
    if (loadBtn) {
        loadBtn.addEventListener('click', () => projectManager.showLoadDialog());
    }

    // Settings buttons
    const settingsBtns = document.querySelectorAll('.nav-btn[data-view="settings"]');
    settingsBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            settings.show();
        });
    });

    // Visualize Web Output buttons
    const visualizeBtns = document.querySelectorAll('.action-button');
    visualizeBtns.forEach(btn => {
        if (btn.textContent.includes('Visualize')) {
            btn.addEventListener('click', () => {
                notify.show('Generating visualization...', 'info');
                setTimeout(() => {
                    notify.show('Visualization complete!', 'success');
                    window.open('https://ai-forge-output.demo', '_blank');
                }, 2000);
            });
        }
    });

    // Panel action buttons
    const panelActions = document.querySelectorAll('.panel-action');
    panelActions.forEach(btn => {
        btn.addEventListener('click', () => {
            const panel = btn.closest('.panel');
            const title = panel.querySelector('h2')?.textContent || 'Panel';
            notify.show(`${title} settings opened`, 'info');
        });
    });

    // Toggle switches
    const toggles = document.querySelectorAll('input[type="checkbox"]');
    toggles.forEach(toggle => {
        toggle.addEventListener('change', (e) => {
            const label = e.target.closest('label')?.textContent || 'Setting';
            const state = e.target.checked ? 'enabled' : 'disabled';
            notify.show(`${label.trim()} ${state}`, 'info', 2000);
        });
    });

    // Radio buttons
    const radios = document.querySelectorAll('input[type="radio"]');
    radios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            notify.show(`Output mode: ${e.target.value}`, 'info', 2000);
        });
    });
}

// ===================================
// Keyboard Shortcuts Enhancement
// ===================================
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            search.show();
        }

        // Ctrl/Cmd + T for terminal
        if ((e.ctrlKey || e.metaKey) && e.key === 't') {
            e.preventDefault();
            terminal.show();
        }

        // Ctrl/Cmd + , for settings
        if ((e.ctrlKey || e.metaKey) && e.key === ',') {
            e.preventDefault();
            settings.show();
        }

        // Ctrl/Cmd + O for open/load
        if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
            e.preventDefault();
            projectManager.showLoadDialog();
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal-overlay');
            modals.forEach(m => m.remove());
        }
    });
}

// ===================================
// Toggle CSS Styles
// ===================================
function addToggleStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .toggle-switch input:checked + span {
            background-color: #00d9ff !important;
        }

        .toggle-switch input:checked + span:before {
            transform: translateX(30px);
        }

        .toggle-switch span:before {
            position: absolute;
            content: "";
            height: 22px;
            width: 22px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }

        .search-result-item:hover {
            transform: translateX(5px);
        }

        button:active {
            transform: scale(0.95);
        }
    `;
    document.head.appendChild(style);
}

// ===================================
// Initialize Everything
// ===================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎮 Initializing Interactive Features...');

    setupInteractiveButtons();
    setupKeyboardShortcuts();
    addToggleStyles();

    // Welcome notification
    setTimeout(() => {
        notify.show('Welcome to AI Forge Studio! Press Ctrl+K for search', 'info', 5000);
    }, 1000);

    console.log('✨ Interactive Features Ready!');
});

// Make functions globally available
window.modal = modal;
window.notify = notify;
window.terminal = terminal;
window.search = search;
window.settings = settings;
window.projectManager = projectManager;
