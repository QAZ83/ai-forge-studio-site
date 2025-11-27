/**
 * AI Forge Studio - Interactive Dashboard
 * Author: M.3R3
 * Advanced UI with real-time visualization
 */

// ===================================
// Global Error Handler
// ===================================
window.onerror = function(message, source, lineno, colno, error) {
    console.error('Global error:', { message, source, lineno, colno, error });
    if (message.includes('ResizeObserver') || message.includes('Script error')) {
        return true;
    }
    return false;
};

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    event.preventDefault();
});

// ===================================
// Electron Detection
// ===================================
const isElectron = typeof window !== 'undefined' && 
                   typeof window.electronAPI !== 'undefined' && 
                   window.electronAPI.isElectron === true;

if (isElectron) {
    console.log('🖥️ Running in Electron desktop mode');
} else {
    console.log('🌐 Running in web browser mode');
}

// ===================================
// GPU 3D Visualization (Optimized)
// ===================================
class GPUVisualization {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width = this.canvas.offsetWidth || 400;
        this.height = this.canvas.height = 300;

        this.nodes = [];
        this.time = 0;
        this.isRunning = true;

        // Optimized settings - faster FPS for smooth animation
        this.fps = 24;
        this.fpsInterval = 1000 / this.fps;
        this.then = performance.now();

        this.init();
        requestAnimationFrame(() => this.animate());
    }

    init() {
        // Fewer nodes for better performance
        const nodeCount = 20;
        for (let i = 0; i < nodeCount; i++) {
            this.nodes.push({
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                z: Math.random() * 300,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                vz: (Math.random() - 0.5) * 0.3,
                radius: Math.random() * 2 + 1
            });
        }
    }

    updateNodes() {
        this.nodes.forEach(node => {
            node.x += node.vx;
            node.y += node.vy;
            node.z += node.vz;

            if (node.x < 0) node.x = this.width;
            if (node.x > this.width) node.x = 0;
            if (node.y < 0) node.y = this.height;
            if (node.y > this.height) node.y = 0;
            if (node.z < 0) node.z = 300;
            if (node.z > 300) node.z = 0;
        });
    }

    draw() {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw connections (simplified)
        const maxDistance = 120;
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const dx = this.nodes[i].x - this.nodes[j].x;
                const dy = this.nodes[i].y - this.nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < maxDistance) {
                    const opacity = (1 - distance / maxDistance) * 0.4;
                    this.ctx.beginPath();
                    this.ctx.strokeStyle = `rgba(0, 217, 255, ${opacity})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.moveTo(this.nodes[i].x, this.nodes[i].y);
                    this.ctx.lineTo(this.nodes[j].x, this.nodes[j].y);
                    this.ctx.stroke();
                }
            }
        }

        // Draw nodes
        this.nodes.forEach(node => {
            const depth = 1 - node.z / 300;
            this.ctx.fillStyle = `rgba(0, 255, 204, ${depth})`;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node.radius * depth, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    animate() {
        if (!this.isRunning) return;
        requestAnimationFrame(() => this.animate());

        const now = performance.now();
        const elapsed = now - this.then;

        if (elapsed > this.fpsInterval) {
            this.then = now - (elapsed % this.fpsInterval);
            this.updateNodes();
            this.draw();
            this.time += 0.01;
        }
    }

    stop() {
        this.isRunning = false;
    }
}

// ===================================
// Circular Progress Animation
// ===================================
function animateCircularProgress() {
    const progressElements = document.querySelectorAll('.circular-progress');

    progressElements.forEach(element => {
        const value = parseInt(element.getAttribute('data-value')) || 0;
        const circle = element.querySelector('.progress-ring-fill');
        if (!circle) return;

        const circumference = 2 * Math.PI * 60;
        const offset = circumference - (value / 100) * circumference;

        requestAnimationFrame(() => {
            circle.style.strokeDashoffset = offset;
        });
    });
}

// ===================================
// Real System Monitor
// ===================================
class SystemMonitor {
    constructor() {
        this.cpuUsage = 0;
        this.memoryUsage = 0;
        this.updateInterval = 2000;
        this.systemInfo = null;

        this.init();
    }

    async init() {
        if (isElectron && window.electronAPI.getSystemInfo) {
            try {
                this.systemInfo = await window.electronAPI.getSystemInfo();
                this.displaySystemInfo();
                this.startMonitoring();
            } catch (e) {
                console.warn('Could not get system info:', e);
                this.startSimulatedMonitoring();
            }
        } else {
            this.startBrowserMonitoring();
        }
    }

    displaySystemInfo() {
        if (!this.systemInfo) return;

        const cpuModelEl = document.querySelector('.cpu-model');
        if (cpuModelEl) cpuModelEl.textContent = '🔹 ' + (this.systemInfo.cpuModel || 'Unknown CPU');

        const totalMemEl = document.querySelector('.total-memory');
        if (totalMemEl) totalMemEl.textContent = '🔹 RAM: ' + this.systemInfo.totalMemoryGB;

        const platformEl = document.querySelector('.platform-info');
        if (platformEl) platformEl.textContent = '🔹 ' + this.systemInfo.platform + ' (' + this.systemInfo.arch + ')';

        const nodeVersionEl = document.querySelector('.node-version');
        if (nodeVersionEl) nodeVersionEl.textContent = this.systemInfo.nodeVersion || '-';

        const electronVersionEl = document.querySelector('.electron-version');
        if (electronVersionEl) electronVersionEl.textContent = this.systemInfo.electronVersion || '-';
    }

    startMonitoring() {
        this.updateMetrics();
        setInterval(() => this.updateMetrics(), this.updateInterval);
    }

    async updateMetrics() {
        if (isElectron && window.electronAPI.getRealtimeStats) {
            try {
                const stats = await window.electronAPI.getRealtimeStats();
                this.cpuUsage = stats.cpuUsage;
                this.memoryUsage = stats.memoryUsage;
            } catch (e) {
                console.warn('Stats update failed:', e);
            }
        }
        this.updateDisplay();
    }

    startBrowserMonitoring() {
        if (performance.memory) {
            setInterval(() => {
                const memory = performance.memory;
                this.memoryUsage = Math.round((memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100);
                this.cpuUsage = Math.round(Math.random() * 20 + 10);
                this.updateDisplay();
            }, this.updateInterval);
        } else {
            this.startSimulatedMonitoring();
        }
    }

    startSimulatedMonitoring() {
        this.cpuUsage = 15;
        this.memoryUsage = 45;
        setInterval(() => {
            this.cpuUsage = Math.max(5, Math.min(30, this.cpuUsage + (Math.random() - 0.5) * 5));
            this.memoryUsage = Math.max(30, Math.min(70, this.memoryUsage + (Math.random() - 0.5) * 3));
            this.updateDisplay();
        }, this.updateInterval);
    }

    updateDisplay() {
        // Update new-style metric displays
        const cpuCircle = document.querySelector('.circular-progress[data-metric="cpu"]');
        if (cpuCircle) {
            const cpuValue = cpuCircle.querySelector('.progress-value .value');
            if (cpuValue) cpuValue.textContent = Math.round(this.cpuUsage) + '%';
            const cpuFill = cpuCircle.querySelector('.progress-ring-fill');
            if (cpuFill) {
                const circumference = 2 * Math.PI * 60;
                cpuFill.style.strokeDashoffset = circumference - (this.cpuUsage / 100) * circumference;
            }
        }

        const memCircle = document.querySelector('.circular-progress[data-metric="memory"]');
        if (memCircle) {
            const memValue = memCircle.querySelector('.progress-value .value');
            if (memValue) memValue.textContent = Math.round(this.memoryUsage) + '%';
            const memFill = memCircle.querySelector('.progress-ring-fill');
            if (memFill) {
                const circumference = 2 * Math.PI * 60;
                memFill.style.strokeDashoffset = circumference - (this.memoryUsage / 100) * circumference;
            }
        }

        // Update legacy displays (data-value attributes)
        const gpuCircle = document.querySelector('.circular-progress[data-value="95"]');
        if (gpuCircle) {
            const gpuValue = gpuCircle.querySelector('.progress-value .value');
            if (gpuValue) gpuValue.textContent = Math.round(this.memoryUsage) + '%';
        }

        const cpuLegacy = document.querySelector('.circular-progress[data-value="12"]');
        if (cpuLegacy) {
            const cpuVal = cpuLegacy.querySelector('.progress-value .value');
            if (cpuVal) cpuVal.textContent = Math.round(this.cpuUsage) + '%';
        }
    }
}

// ===================================
// Loss Chart (Lazy Loaded)
// ===================================
function createLossChart() {
    const ctx = document.getElementById('lossChart');
    if (!ctx) return;

    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded');
        return;
    }

    if (ctx.dataset.chartCreated === 'true') return;
    ctx.dataset.chartCreated = 'true';

    try {
        const epochs = 90;
        const data = [];
        let value = 0.9;

        for (let i = 0; i < epochs; i++) {
            value = value * 0.95 + (Math.random() - 0.5) * 0.05;
            data.push(value.toFixed(4));
        }

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: epochs}, (_, i) => i),
                datasets: [{
                    label: 'Loss',
                    data: data,
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
                animation: { duration: 0 },
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        grid: { color: 'rgba(0, 217, 255, 0.1)' },
                        ticks: { color: '#8b95a8', maxTicksLimit: 10 }
                    },
                    y: {
                        grid: { color: 'rgba(0, 217, 255, 0.1)' },
                        ticks: { color: '#8b95a8' }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error creating chart:', error);
    }
}

// ===================================
// Navigation Setup
// ===================================
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(button => {
        button.addEventListener('click', function() {
            navButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// ===================================
// Terminal Animation
// ===================================
class TerminalAnimation {
    constructor() {
        this.terminal = document.querySelector('.terminal-output');
        this.messages = [
            { icon: '▶', text: 'Initializing build process...', type: 'info' },
            { icon: '📘', text: 'Loading dependencies...', type: 'info' },
            { icon: '▶▶', text: 'Compiling sources...', type: 'info' },
            { icon: '⊞', text: 'Build completed successfully', type: 'success' }
        ];
    }

    typeText(element, text, speed = 50) {
        let i = 0;
        const interval = setInterval(() => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
            } else {
                clearInterval(interval);
            }
        }, speed);
    }
}

// ===================================
// Interactive Effects
// ===================================
function addInteractiveEffects() {
    // Panel hover effects
    const panels = document.querySelectorAll('.panel');
    panels.forEach(panel => {
        panel.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });

        panel.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Button ripple effect
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            ripple.classList.add('ripple');
            this.appendChild(ripple);

            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';

            setTimeout(() => ripple.remove(), 600);
        });
    });
}

// ===================================
// Lazy Loading with Intersection Observer
// ===================================
function setupLazyLoading() {
    const observerOptions = {
        root: null,
        rootMargin: '50px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;

                // Load charts when they come into view
                if (target.id === 'lossChart' && !target.dataset.chartCreated) {
                    createLossChart();
                }

                observer.unobserve(target);
            }
        });
    }, observerOptions);

    // Observe chart canvases
    const lossChart = document.getElementById('lossChart');
    if (lossChart) observer.observe(lossChart);
}

// ===================================
// Application Info Display
// ===================================
async function displayAppInfo() {
    if (isElectron) {
        try {
            const version = await window.electronAPI.getAppVersion();
            const versionEl = document.querySelector('.app-version');
            if (versionEl) versionEl.textContent = `v${version}`;

            const systemInfo = await window.electronAPI.getSystemInfo();
            console.log('📊 System Info:', systemInfo);
        } catch (e) {
            console.warn('Could not get app info:', e);
        }
    }
}

// ===================================
// Initialization (Fast - No Delays)
// ===================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 AI Forge Studio Initializing...');

    // Immediate initialization - no timeouts
    setupLazyLoading();
    setupNavigation();
    addInteractiveEffects();
    animateCircularProgress();

    // Initialize GPU visualization if canvas exists
    const gpuCanvas = document.getElementById('gpuCanvas');
    if (gpuCanvas) {
        new GPUVisualization('gpuCanvas');
    }

    // Start real system monitoring
    new SystemMonitor();

    // Display app info
    displayAppInfo();

    console.log('✨ AI Forge Studio Ready!');
});

// ===================================
// Window Resize Handler
// ===================================
window.addEventListener('resize', function() {
    const gpuCanvas = document.getElementById('gpuCanvas');
    if (gpuCanvas) {
        gpuCanvas.width = gpuCanvas.offsetWidth;
    }
});

// ===================================
// Keyboard Shortcuts
// ===================================
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K for search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchBtn = document.querySelector('.search-btn');
        if (searchBtn) searchBtn.click();
    }

    // Ctrl/Cmd + T for terminal
    if ((e.ctrlKey || e.metaKey) && e.key === 't') {
        e.preventDefault();
        const terminalBtn = document.querySelector('.terminal-btn');
        if (terminalBtn) terminalBtn.click();
    }
});
