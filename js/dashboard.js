/**
 * Dashboard Specific JavaScript
 * Advanced charts and visualizations
 */

// ===================================
// GPU Tracking Chart
// ===================================
function createGPUTrackingChart() {
    const ctx = document.getElementById('gpuTrackingChart');
    if (!ctx) return;

    const labels = Array.from({length: 50}, (_, i) => i);
    const data1 = generateRealisticData(50, 85, 95);
    const data2 = generateRealisticData(50, 70, 85);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'GPU Usage',
                    data: data1,
                    borderColor: '#00d9ff',
                    backgroundColor: 'rgba(0, 217, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'VRAM Usage',
                    data: data2,
                    borderColor: '#00ffcc',
                    backgroundColor: 'rgba(0, 255, 204, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#8b95a8',
                        font: {
                            family: 'Rajdhani',
                            size: 12
                        },
                        usePointStyle: true,
                        padding: 15
                    }
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 217, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8b95a8',
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// ===================================
// Status Chart
// ===================================
function createStatusChart() {
    const ctx = document.getElementById('statusChart');
    if (!ctx) return;

    const data = generateRealisticData(30, 60, 95);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 30}, (_, i) => i),
            datasets: [{
                label: 'System Load',
                data: data,
                borderColor: '#00ffcc',
                backgroundColor: 'rgba(0, 255, 204, 0.2)',
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointRadius: 3,
                pointBackgroundColor: '#00ffcc',
                pointBorderColor: '#00d9ff',
                pointBorderWidth: 2,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: false
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
            }
        }
    });
}

// ===================================
// Training Metrics Chart
// ===================================
function createTrainingMetricsChart() {
    const ctx = document.getElementById('trainingMetricsChart');
    if (!ctx) return;

    const epochs = 100;
    const loss = generateDecreasingData(epochs, 0.9, 0.05);
    const accuracy = generateIncreasingData(epochs, 0.6, 0.95);
    const valLoss = generateDecreasingData(epochs, 0.95, 0.08);
    const valAccuracy = generateIncreasingData(epochs, 0.55, 0.92);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: epochs}, (_, i) => i),
            datasets: [
                {
                    label: 'Training Loss',
                    data: loss,
                    borderColor: '#ff3366',
                    backgroundColor: 'rgba(255, 51, 102, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Validation Loss',
                    data: valLoss,
                    borderColor: '#ffd500',
                    backgroundColor: 'rgba(255, 213, 0, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Training Accuracy',
                    data: accuracy,
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y1'
                },
                {
                    label: 'Validation Accuracy',
                    data: valAccuracy,
                    borderColor: '#00d9ff',
                    backgroundColor: 'rgba(0, 217, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
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
                        usePointStyle: true,
                        padding: 10
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#00d9ff',
                    bodyColor: '#e0e6ed',
                    borderColor: '#00d9ff',
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 217, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8b95a8',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: {
                        color: 'rgba(0, 217, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8b95a8'
                    },
                    title: {
                        display: true,
                        text: 'Loss',
                        color: '#8b95a8'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false,
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8b95a8',
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Accuracy',
                        color: '#8b95a8'
                    }
                }
            }
        }
    });
}

// ===================================
// Data Generation Helpers
// ===================================
function generateRealisticData(length, min, max) {
    const data = [];
    let value = (min + max) / 2;

    for (let i = 0; i < length; i++) {
        value = value + (Math.random() - 0.5) * (max - min) * 0.1;
        value = Math.max(min, Math.min(max, value));
        data.push(value.toFixed(2));
    }

    return data;
}

function generateDecreasingData(length, start, end) {
    const data = [];
    const decay = Math.pow(end / start, 1 / length);

    for (let i = 0; i < length; i++) {
        const value = start * Math.pow(decay, i);
        const noise = (Math.random() - 0.5) * 0.05;
        data.push((value + noise).toFixed(4));
    }

    return data;
}

function generateIncreasingData(length, start, end) {
    const data = [];
    const growth = Math.pow(end / start, 1 / length);

    for (let i = 0; i < length; i++) {
        const value = start * Math.pow(growth, i);
        const noise = (Math.random() - 0.5) * 0.02;
        data.push(Math.min(end, value + noise).toFixed(4));
    }

    return data;
}

// ===================================
// Real-time Updates
// ===================================
class DashboardMonitor {
    constructor() {
        this.updateInterval = 3000;
        this.startMonitoring();
    }

    startMonitoring() {
        setInterval(() => {
            this.updateMetrics();
        }, this.updateInterval);
    }

    updateMetrics() {
        // Update metric cards with animation
        this.animateMetricValue('.metric-card:nth-child(1) .metric-value', 93, 97);
        this.animateMetricValue('.metric-card:nth-child(2) .metric-value', 10, 15);
    }

    animateMetricValue(selector, min, max) {
        const element = document.querySelector(selector);
        if (!element) return;

        const currentValue = parseInt(element.textContent);
        const newValue = Math.floor(Math.random() * (max - min + 1)) + min;

        if (currentValue !== newValue) {
            element.style.transform = 'scale(1.1)';
            setTimeout(() => {
                element.textContent = newValue + '%';
                element.style.transform = 'scale(1)';
            }, 200);
        }
    }
}

// ===================================
// Initialize Dashboard
// ===================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('📊 Initializing Dashboard...');

    // Create all charts
    createGPUTrackingChart();
    createStatusChart();
    createTrainingMetricsChart();

    // Animate progress circles
    animateCircularProgress();

    // Start monitoring
    const monitor = new DashboardMonitor();

    console.log('✨ Dashboard Ready!');
});
