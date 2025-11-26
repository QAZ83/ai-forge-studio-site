/**
 * AI Forge Studio - Multi-Dashboard Interface
 * Optimized for performance with multiple windows
 */

// ===================================
// Circular Progress Animation
// ===================================
function animateCircularProgress() {
    const progressElements = document.querySelectorAll('.circular-mini');

    progressElements.forEach(element => {
        const value = parseInt(element.getAttribute('data-value'));
        const circle = element.querySelector('.progress-ring-fill');

        if (!circle) return;

        const radius = circle.getAttribute('r');
        const circumference = 2 * Math.PI * radius;

        // Set initial dasharray
        circle.style.strokeDasharray = `${circumference} ${circumference}`;
        circle.style.strokeDashoffset = circumference;

        // Calculate offset
        const offset = circumference - (value / 100) * circumference;

        // Animate
        setTimeout(() => {
            circle.style.strokeDashoffset = offset;
        }, 300);
    });
}

// ===================================
// Create Charts with Performance Optimization
// ===================================
const chartConfig = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
        duration: 750
    },
    plugins: {
        legend: {
            display: false
        },
        tooltip: {
            enabled: false
        }
    },
    scales: {
        x: {
            display: false
        },
        y: {
            display: false
        }
    },
    elements: {
        point: {
            radius: 0
        }
    }
};

function createGPUChart() {
    const ctx = document.getElementById('gpuChart1');
    if (!ctx) return;

    const data = generateTimeSeriesData(30, 85, 95);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 30}, (_, i) => i),
            datasets: [{
                data: data,
                borderColor: '#00d9ff',
                backgroundColor: 'rgba(0, 217, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: chartConfig
    });
}

function createSceneChart() {
    const ctx = document.getElementById('sceneChart');
    if (!ctx) return;

    const data = generateTimeSeriesData(25, 60, 80);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 25}, (_, i) => i),
            datasets: [{
                data: data,
                borderColor: '#00ffcc',
                backgroundColor: 'rgba(0, 255, 204, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: chartConfig
    });
}

function createPerfChart() {
    const ctx = document.getElementById('perfChart');
    if (!ctx) return;

    const gpuData = generateTimeSeriesData(40, 85, 98);
    const cpuData = generateTimeSeriesData(40, 10, 20);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 40}, (_, i) => i),
            datasets: [
                {
                    label: 'GPU',
                    data: gpuData,
                    borderColor: '#00d9ff',
                    backgroundColor: 'rgba(0, 217, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'CPU',
                    data: cpuData,
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            ...chartConfig,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#8b95a8',
                        font: {
                            family: 'Rajdhani',
                            size: 11
                        },
                        padding: 10
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
                        maxTicksLimit: 8,
                        font: {
                            size: 9
                        }
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 217, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8b95a8',
                        maxTicksLimit: 5,
                        font: {
                            size: 9
                        }
                    }
                }
            }
        }
    });
}

// ===================================
// Helper Functions
// ===================================
function generateTimeSeriesData(length, min, max) {
    const data = [];
    let value = (min + max) / 2;

    for (let i = 0; i < length; i++) {
        value = value + (Math.random() - 0.5) * (max - min) * 0.1;
        value = Math.max(min, Math.min(max, value));
        data.push(parseFloat(value.toFixed(2)));
    }

    return data;
}

// ===================================
// Window Interactions
// ===================================
function setupWindowControls() {
    // Minimize buttons
    document.querySelectorAll('.win-btn.minimize').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const card = btn.closest('.window-card');
            card.classList.toggle('minimized');
        });
    });

    // Maximize buttons
    document.querySelectorAll('.win-btn.maximize').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const card = btn.closest('.window-card');
            card.classList.toggle('maximized');
        });
    });

    // Close buttons
    document.querySelectorAll('.win-btn.close').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const card = btn.closest('.window-card');
            card.style.opacity = '0';
            card.style.transform = 'scale(0.8)';
            setTimeout(() => {
                card.style.display = 'none';
            }, 300);
        });
    });
}

// ===================================
// Real-time Updates (optimized)
// ===================================
class MultiDashboardMonitor {
    constructor() {
        this.updateInterval = 3000; // Update every 3 seconds
        this.startMonitoring();
    }

    startMonitoring() {
        setInterval(() => {
            this.updateRandomMetrics();
        }, this.updateInterval);
    }

    updateRandomMetrics() {
        // Randomly update one or two metrics to simulate real-time changes
        const elements = document.querySelectorAll('.circular-mini');
        if (elements.length === 0) return;

        // Pick random element to update
        const randomIndex = Math.floor(Math.random() * elements.length);
        const element = elements[randomIndex];

        const currentValue = parseInt(element.getAttribute('data-value'));
        const newValue = Math.max(5, Math.min(99, currentValue + (Math.random() - 0.5) * 5));

        element.setAttribute('data-value', Math.round(newValue));

        const valueSpan = element.querySelector('.value');
        if (valueSpan) {
            valueSpan.textContent = Math.round(newValue) + '%';
        }

        const circle = element.querySelector('.progress-ring-fill');
        if (circle) {
            const radius = circle.getAttribute('r');
            const circumference = 2 * Math.PI * radius;
            const offset = circumference - (newValue / 100) * circumference;
            circle.style.strokeDashoffset = offset;
        }
    }
}

// ===================================
// Lazy Loading for Charts
// ===================================
function setupChartLazyLoading() {
    const observerOptions = {
        root: null,
        rootMargin: '100px',
        threshold: 0.1
    };

    const chartObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;

                if (target.id === 'gpuChart1' && !target.dataset.chartCreated) {
                    target.dataset.chartCreated = 'true';
                    setTimeout(() => createGPUChart(), 100);
                } else if (target.id === 'sceneChart' && !target.dataset.chartCreated) {
                    target.dataset.chartCreated = 'true';
                    setTimeout(() => createSceneChart(), 150);
                } else if (target.id === 'perfChart' && !target.dataset.chartCreated) {
                    target.dataset.chartCreated = 'true';
                    setTimeout(() => createPerfChart(), 200);
                }

                chartObserver.unobserve(target);
            }
        });
    }, observerOptions);

    // Observe all chart canvases
    const charts = ['gpuChart1', 'sceneChart', 'perfChart'];
    charts.forEach(id => {
        const chart = document.getElementById(id);
        if (chart) chartObserver.observe(chart);
    });
}

// ===================================
// Initialization
// ===================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Multi-Dashboard Initializing...');

    // Animate progress circles with slight delay
    setTimeout(() => {
        animateCircularProgress();
        console.log('✅ Progress circles animated');
    }, 200);

    // Setup lazy loading for charts
    setupChartLazyLoading();
    console.log('✅ Chart lazy loading configured');

    // Setup window controls
    setupWindowControls();
    console.log('✅ Window controls ready');

    // Start monitoring (with delay to reduce initial load)
    setTimeout(() => {
        const monitor = new MultiDashboardMonitor();
        console.log('✅ Real-time monitoring active');
    }, 500);

    console.log('✨ Multi-Dashboard Ready!');
});

// ===================================
// Window Resize Handler (throttled)
// ===================================
let resizeTimeout;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        // Refresh charts on resize if needed
        console.log('Window resized');
    }, 250);
});
