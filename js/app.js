/**
 * AI Forge Studio - Interactive Dashboard
 * Author: M.3R3
 * Advanced UI with real-time visualization
 */

// ===================================
// GPU 3D Visualization
// ===================================
class GPUVisualization {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width = this.canvas.offsetWidth;
        this.height = this.canvas.height = 300;

        this.nodes = [];
        this.connections = [];
        this.time = 0;

        this.init();
        this.animate();
    }

    init() {
        // Create 3D network nodes
        const nodeCount = 80;
        for (let i = 0; i < nodeCount; i++) {
            this.nodes.push({
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                z: Math.random() * 500,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                vz: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 2 + 1
            });
        }
    }

    updateNodes() {
        this.nodes.forEach(node => {
            node.x += node.vx;
            node.y += node.vy;
            node.z += node.vz;

            // Wrap around edges
            if (node.x < 0) node.x = this.width;
            if (node.x > this.width) node.x = 0;
            if (node.y < 0) node.y = this.height;
            if (node.y > this.height) node.y = 0;
            if (node.z < 0) node.z = 500;
            if (node.z > 500) node.z = 0;
        });
    }

    drawConnections() {
        const maxDistance = 150;

        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const dx = this.nodes[i].x - this.nodes[j].x;
                const dy = this.nodes[i].y - this.nodes[j].y;
                const dz = this.nodes[i].z - this.nodes[j].z;
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

                if (distance < maxDistance) {
                    const opacity = (1 - distance / maxDistance) * 0.5;
                    const depth1 = 1 - this.nodes[i].z / 500;
                    const depth2 = 1 - this.nodes[j].z / 500;
                    const avgDepth = (depth1 + depth2) / 2;

                    this.ctx.beginPath();
                    this.ctx.strokeStyle = `rgba(0, 217, 255, ${opacity * avgDepth})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.moveTo(this.nodes[i].x, this.nodes[i].y);
                    this.ctx.lineTo(this.nodes[j].x, this.nodes[j].y);
                    this.ctx.stroke();
                }
            }
        }
    }

    drawNodes() {
        this.nodes.forEach(node => {
            const depth = 1 - node.z / 500;
            const size = node.radius * depth;
            const brightness = depth;

            // Draw glow
            const gradient = this.ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, size * 3
            );
            gradient.addColorStop(0, `rgba(0, 217, 255, ${brightness * 0.8})`);
            gradient.addColorStop(1, 'rgba(0, 217, 255, 0)');

            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, size * 3, 0, Math.PI * 2);
            this.ctx.fill();

            // Draw node
            this.ctx.fillStyle = `rgba(0, 255, 204, ${brightness})`;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    animate() {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        this.ctx.fillRect(0, 0, this.width, this.height);

        this.updateNodes();
        this.drawConnections();
        this.drawNodes();

        this.time += 0.01;
        requestAnimationFrame(() => this.animate());
    }
}

// ===================================
// Circular Progress Animation
// ===================================
function animateCircularProgress() {
    const progressElements = document.querySelectorAll('.circular-progress');

    progressElements.forEach(element => {
        const value = parseInt(element.getAttribute('data-value'));
        const circle = element.querySelector('.progress-ring-fill');
        const circumference = 2 * Math.PI * 60; // radius = 60

        // Calculate offset
        const offset = circumference - (value / 100) * circumference;

        // Animate
        setTimeout(() => {
            circle.style.strokeDashoffset = offset;
        }, 100);
    });
}

// ===================================
// Loss Chart
// ===================================
function createLossChart() {
    const ctx = document.getElementById('lossChart');
    if (!ctx) return;

    // Generate realistic training data
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
                pointHoverRadius: 5,
                pointBackgroundColor: '#00ffcc',
                pointBorderColor: '#00d9ff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#00d9ff',
                    bodyColor: '#e0e6ed',
                    borderColor: '#00d9ff',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return 'Loss: ' + context.parsed.y;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 217, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8b95a8',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 217, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8b95a8'
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

// ===================================
// Navigation
// ===================================
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');

    navButtons.forEach(button => {
        button.addEventListener('click', function() {
            navButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            const view = this.getAttribute('data-view');
            console.log('Switching to view:', view);
            // Add your view switching logic here
        });
    });
}

// ===================================
// Real-time Updates Simulation
// ===================================
class SystemMonitor {
    constructor() {
        this.gpuUsage = 95;
        this.cpuUsage = 12;
        this.updateInterval = 2000;

        this.startMonitoring();
    }

    startMonitoring() {
        setInterval(() => {
            this.updateMetrics();
        }, this.updateInterval);
    }

    updateMetrics() {
        // Simulate metric fluctuations
        this.gpuUsage = Math.max(90, Math.min(99, this.gpuUsage + (Math.random() - 0.5) * 5));
        this.cpuUsage = Math.max(8, Math.min(20, this.cpuUsage + (Math.random() - 0.5) * 3));

        // Update display
        this.updateProgressCircles();
    }

    updateProgressCircles() {
        const gpuCircle = document.querySelector('.circular-progress[data-value="95"]');
        const cpuCircle = document.querySelector('.circular-progress[data-value="12"]');

        if (gpuCircle) {
            const gpuValue = gpuCircle.querySelector('.progress-value .value');
            if (gpuValue) {
                gpuValue.textContent = Math.round(this.gpuUsage) + '%';
            }

            const gpuFill = gpuCircle.querySelector('.progress-ring-fill');
            const circumference = 2 * Math.PI * 60;
            const offset = circumference - (this.gpuUsage / 100) * circumference;
            gpuFill.style.strokeDashoffset = offset;
        }

        if (cpuCircle) {
            const cpuValue = cpuCircle.querySelector('.progress-value .value');
            if (cpuValue) {
                cpuValue.textContent = Math.round(this.cpuUsage) + '%';
            }

            const cpuFill = cpuCircle.querySelector('.progress-ring-fill');
            const circumference = 2 * Math.PI * 60;
            const offset = circumference - (this.cpuUsage / 100) * circumference;
            cpuFill.style.strokeDashoffset = offset;
        }
    }
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
            this.style.transform = 'translateY(-4px)';
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
// Initialization
// ===================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 AI Forge Studio Initializing...');

    // Initialize GPU Visualization
    const gpuViz = new GPUVisualization('gpuCanvas');
    console.log('✅ GPU Visualization loaded');

    // Animate circular progress
    animateCircularProgress();
    console.log('✅ Progress indicators animated');

    // Create loss chart
    createLossChart();
    console.log('✅ Loss chart created');

    // Setup navigation
    setupNavigation();
    console.log('✅ Navigation setup complete');

    // Start system monitoring
    const monitor = new SystemMonitor();
    console.log('✅ System monitoring active');

    // Add interactive effects
    addInteractiveEffects();
    console.log('✅ Interactive effects enabled');

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
