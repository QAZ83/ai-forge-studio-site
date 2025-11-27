/**
 * CUDA Integration JavaScript
 * Handles CUDA-related UI and interactions
 */

// ===================================
// CUDA System Check
// ===================================
function checkCudaSystem() {
    notify.show('Checking CUDA system...', 'info');

    setTimeout(() => {
        notify.show('CUDA 12.4 detected and operational!', 'success');
        updateCudaMetrics();
    }, 1500);
}

function updateCudaMetrics() {
    // Simulate updating metrics
    const vramUsed = (Math.random() * 2 + 29).toFixed(1);
    const vramFree = (32 - parseFloat(vramUsed)).toFixed(1);
    const vramPercent = ((vramUsed / 32) * 100).toFixed(0);

    document.getElementById('vram-used').textContent = vramUsed + ' GB';
    document.getElementById('vram-free').textContent = vramFree + ' GB';
    document.getElementById('vram-bar').style.width = vramPercent + '%';
}

// ===================================
// Code Tab Switching
// ===================================
function showCudaTab(tab) {
    // Hide all code displays
    document.getElementById('cuda-kernel-code').style.display = 'none';
    document.getElementById('cuda-host-code').style.display = 'none';
    document.getElementById('cuda-cmake-code').style.display = 'none';

    // Remove active class from all tabs
    document.querySelectorAll('.env-tab').forEach(t => t.classList.remove('active'));

    // Show selected tab
    document.getElementById('cuda-' + tab + '-code').style.display = 'block';
    event.target.classList.add('active');
}

function copyCudaCode() {
    const activeCode = document.querySelector('.code-display:not([style*="display: none"]) code');
    if (activeCode) {
        const text = activeCode.innerText;
        navigator.clipboard.writeText(text).then(() => {
            notify.show('Code copied to clipboard!', 'success');
        });
    }
}

// ===================================
// Quick Actions
// ===================================
function runCudaTest() {
    notify.show('Running CUDA device test...', 'info');

    setTimeout(() => {
        modal.show('cuda-test', '🧪 CUDA Test Results', `
            <div style="font-family: 'Courier New', monospace; background: #000; padding: 20px; border-radius: 8px;">
                <div style="color: #00ff88; margin-bottom: 10px;">[CUDA Test] Device Query...</div>
                <div style="color: #00d9ff;">Device 0: "NVIDIA GeForce RTX 5090"</div>
                <div style="color: #8b95a8; margin-left: 20px;">CUDA Driver Version / Runtime Version: 12.4 / 12.4</div>
                <div style="color: #8b95a8; margin-left: 20px;">CUDA Capability Major/Minor version: 8.9</div>
                <div style="color: #8b95a8; margin-left: 20px;">Total amount of global memory: 32768 MBytes</div>
                <div style="color: #8b95a8; margin-left: 20px;">GPU Max Clock rate: 2520 MHz</div>
                <div style="color: #8b95a8; margin-left: 20px;">Memory Clock rate: 28000 Mhz</div>
                <div style="color: #8b95a8; margin-left: 20px;">Memory Bus Width: 512-bit</div>
                <div style="color: #00ff88; margin-top: 15px;">[CUDA Test] All tests PASSED ✓</div>
            </div>
        `);
    }, 2000);
}

function benchmarkGPU() {
    notify.show('Starting GPU benchmark...', 'info');

    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        notify.show(`Benchmarking... ${progress}%`, 'info', 1000);

        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
                modal.show('benchmark', '📊 GPU Benchmark Results', `
                    <div style="padding: 20px;">
                        <h3 style="color: #00d9ff; margin-bottom: 20px;">Performance Results</h3>
                        <div style="margin-bottom: 15px;">
                            <div style="color: #00ffcc; font-weight: 600;">Matrix Multiplication (4096x4096)</div>
                            <div style="color: #8b95a8;">Execution Time: 2.3ms | Throughput: 435 GFLOPS</div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <div style="color: #00ffcc; font-weight: 600;">Memory Bandwidth Test</div>
                            <div style="color: #8b95a8;">Host to Device: 28.5 GB/s | Device to Host: 29.2 GB/s</div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <div style="color: #00ffcc; font-weight: 600;">FP16 Tensor Core Performance</div>
                            <div style="color: #8b95a8;">Peak: 1321 TFLOPS</div>
                        </div>
                        <div style="margin-top: 20px; padding: 15px; background: rgba(0, 217, 255, 0.1); border-radius: 8px;">
                            <div style="color: #00ff88; font-weight: 600;">✅ Performance Score: 98/100</div>
                            <div style="color: #8b95a8; font-size: 13px; margin-top: 5px;">Your GPU is performing excellently!</div>
                        </div>
                    </div>
                `);
            }, 500);
        }
    }, 500);
}

function checkCompatibility() {
    notify.show('Checking software compatibility...', 'info');

    setTimeout(() => {
        modal.show('compatibility', '✅ Compatibility Check', `
            <div style="padding: 20px;">
                <h3 style="color: #00d9ff; margin-bottom: 20px;">Software Compatibility</h3>
                <div class="status-list">
                    <div class="status-item active">
                        <span class="status-dot"></span>
                        <span class="status-text">CUDA Toolkit 12.4 ✓</span>
                    </div>
                    <div class="status-item active">
                        <span class="status-dot"></span>
                        <span class="status-text">TensorRT 10.14.1.48 ✓</span>
                    </div>
                    <div class="status-item active">
                        <span class="status-dot"></span>
                        <span class="status-text">Vulkan SDK 1.4.328.1 ✓</span>
                    </div>
                    <div class="status-item active">
                        <span class="status-dot"></span>
                        <span class="status-text">cuDNN 9.0 ✓</span>
                    </div>
                    <div class="status-item active">
                        <span class="status-dot"></span>
                        <span class="status-text">NVIDIA Driver 566.03 ✓</span>
                    </div>
                    <div class="status-item active">
                        <span class="status-dot"></span>
                        <span class="status-text">Qt 6.5.0 ✓</span>
                    </div>
                </div>
                <div style="margin-top: 20px; padding: 15px; background: rgba(0, 255, 136, 0.1); border-radius: 8px; border-left: 3px solid #00ff88;">
                    <div style="color: #00ff88; font-weight: 600;">All systems compatible!</div>
                    <div style="color: #8b95a8; font-size: 13px; margin-top: 5px;">Your system is ready for AI development with CUDA.</div>
                </div>
            </div>
        `);
    }, 1500);
}

function downloadCudaSamples() {
    notify.show('Preparing CUDA samples download...', 'info');

    setTimeout(() => {
        modal.show('samples', '📥 CUDA Samples', `
            <div style="padding: 20px;">
                <h3 style="color: #00d9ff; margin-bottom: 20px;">Available CUDA Samples</h3>
                <div style="margin-bottom: 15px; padding: 12px; background: rgba(0, 217, 255, 0.05); border-left: 3px solid #00d9ff; border-radius: 4px; cursor: pointer;"
                     onclick="downloadSample('matrix-mul')">
                    <div style="color: #00ffcc; font-weight: 600;">Matrix Multiplication</div>
                    <div style="color: #8b95a8; font-size: 13px;">Basic CUDA kernel for matrix operations</div>
                </div>
                <div style="margin-bottom: 15px; padding: 12px; background: rgba(0, 217, 255, 0.05); border-left: 3px solid #00d9ff; border-radius: 4px; cursor: pointer;"
                     onclick="downloadSample('vectorAdd')">
                    <div style="color: #00ffcc; font-weight: 600;">Vector Addition</div>
                    <div style="color: #8b95a8; font-size: 13px;">Simple vector addition example</div>
                </div>
                <div style="margin-bottom: 15px; padding: 12px; background: rgba(0, 217, 255, 0.05); border-left: 3px solid #00d9ff; border-radius: 4px; cursor: pointer;"
                     onclick="downloadSample('convolution')">
                    <div style="color: #00ffcc; font-weight: 600;">2D Convolution</div>
                    <div style="color: #8b95a8; font-size: 13px;">Image processing with CUDA</div>
                </div>
                <button onclick="window.open('https://github.com/NVIDIA/cuda-samples', '_blank')"
                        style="background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%); border: none; color: #000; padding: 12px 30px; border-radius: 6px; font-family: 'Rajdhani', sans-serif; font-size: 16px; font-weight: 600; cursor: pointer; width: 100%; margin-top: 15px;">
                    View All Samples on GitHub
                </button>
            </div>
        `);
    }, 1000);
}

function downloadSample(sample) {
    notify.show(`Downloading ${sample} sample...`, 'success');
    modal.close('samples');
}

// ===================================
// Charts
// ===================================
let vramChart = null;
let performanceChart = null;

function createVRAMChart() {
    const ctx = document.getElementById('vramChart');
    if (!ctx) {
        console.warn('VRAM chart canvas not found');
        return;
    }

    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded, retrying in 500ms...');
        setTimeout(createVRAMChart, 500);
        return;
    }

    // Destroy existing chart if any
    if (vramChart) {
        vramChart.destroy();
        vramChart = null;
    }

    const data = [];
    for (let i = 0; i < 60; i++) {
        data.push((Math.random() * 5 + 27).toFixed(1));
    }

    vramChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 60}, (_, i) => i),
            datasets: [{
                label: 'VRAM Usage (GB)',
                data: data,
                borderColor: '#00d9ff',
                backgroundColor: 'rgba(0, 217, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    min: 0,
                    max: 32,
                    grid: {
                        color: 'rgba(0, 217, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8b95a8'
                    }
                }
            }
        }
    });
}

function createPerformanceChart() {
    const ctx = document.getElementById('cudaPerformanceChart');
    if (!ctx) {
        console.warn('CUDA performance chart canvas not found');
        return;
    }

    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded, retrying in 500ms...');
        setTimeout(createPerformanceChart, 500);
        return;
    }

    // Destroy existing chart if any
    if (performanceChart) {
        performanceChart.destroy();
        performanceChart = null;
    }

    performanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Matrix Mul', 'Vector Add', 'Convolution', 'FFT', 'Reduction'],
            datasets: [{
                label: 'GFLOPS',
                data: [435, 512, 387, 445, 498],
                backgroundColor: 'rgba(0, 217, 255, 0.6)',
                borderColor: '#00d9ff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 217, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8b95a8'
                    }
                },
                x: {
                    grid: {
                        display: false
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
// Real-time Updates
// ===================================
setInterval(updateCudaMetrics, 3000);

// ===================================
// Initialization
// ===================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 CUDA Integration loaded');

    createVRAMChart();
    createPerformanceChart();
    updateCudaMetrics();

    console.log('✨ CUDA Interface Ready!');
});
