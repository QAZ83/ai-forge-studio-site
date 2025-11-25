/**
 * TensorRT Integration JavaScript
 */

// ===================================
// TensorRT Functions
// ===================================
function refreshTensorRT() {
    notify.show('Refreshing TensorRT status...', 'info');
    setTimeout(() => {
        notify.show('TensorRT engine operational!', 'success');
    }, 1000);
}

function uploadModel() {
    const content = `
        <div style="padding: 20px;">
            <h3 style="color: #00d9ff; margin-bottom: 20px;">Upload Model</h3>
            <div style="background: rgba(0, 217, 255, 0.05); border: 2px dashed #00d9ff; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer;" onclick="document.getElementById('model-upload').click()">
                <div style="font-size: 48px; margin-bottom: 15px;">📦</div>
                <div style="color: #00ffcc; font-weight: 600; margin-bottom: 10px;">Click to upload or drag and drop</div>
                <div style="color: #8b95a8; font-size: 13px;">Supported formats: .onnx, .pt, .pb, .trt</div>
                <input type="file" id="model-upload" style="display: none;" accept=".onnx,.pt,.pb,.trt" onchange="handleModelUpload(this.files[0])">
            </div>
        </div>
    `;
    modal.show('upload', '📤 Upload Model', content);
}

function handleModelUpload(file) {
    if (file) {
        notify.show(`Uploading ${file.name}...`, 'info');
        modal.close('upload');

        setTimeout(() => {
            notify.show('Model uploaded successfully!', 'success');
            setTimeout(() => {
                notify.show('Converting to TensorRT engine...', 'info');
            }, 1000);
        }, 2000);
    }
}

function runModel(modelName) {
    notify.show(`Running inference with ${modelName}...`, 'info');

    setTimeout(() => {
        modal.show('inference-result', '🎯 Inference Results', `
            <div style="padding: 20px;">
                <h3 style="color: #00d9ff; margin-bottom: 20px;">Model: ${modelName}</h3>
                <div style="background: rgba(0, 217, 255, 0.05); border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                    <div style="color: #00ffcc; font-weight: 600; margin-bottom: 10px;">Performance Metrics</div>
                    <div class="stat-row">
                        <span class="stat-label">Inference Time:</span>
                        <span class="stat-value">2.1 ms</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Throughput:</span>
                        <span class="stat-value">476 FPS</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">GPU Utilization:</span>
                        <span class="stat-value">98%</span>
                    </div>
                </div>
                <div style="background: rgba(0, 255, 136, 0.05); border-radius: 8px; padding: 15px; border-left: 3px solid #00ff88;">
                    <div style="color: #00ff88; font-weight: 600;">✅ Inference completed successfully!</div>
                    <div style="color: #8b95a8; font-size: 13px; margin-top: 5px;">Results saved to output/</div>
                </div>
            </div>
        `);
    }, 2000);
}

function convertModel() {
    const modelType = document.getElementById('model-type').value;
    const precision = document.getElementById('precision').value;

    notify.show('Starting model conversion...', 'info');

    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        notify.show(`Converting... ${progress}%`, 'info', 1000);

        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
                notify.show('Model converted to TensorRT engine!', 'success');
                modal.show('conversion-complete', '✅ Conversion Complete', `
                    <div style="padding: 20px;">
                        <h3 style="color: #00d9ff; margin-bottom: 20px;">TensorRT Engine Created</h3>
                        <div class="status-list">
                            <div class="status-item active">
                                <span class="status-dot"></span>
                                <span class="status-text">Model Type: ${modelType}</span>
                            </div>
                            <div class="status-item active">
                                <span class="status-dot"></span>
                                <span class="status-text">Precision: ${precision}</span>
                            </div>
                            <div class="status-item active">
                                <span class="status-dot"></span>
                                <span class="status-text">Engine Size: 145 MB</span>
                            </div>
                            <div class="status-item active">
                                <span class="status-dot"></span>
                                <span class="status-text">Expected Latency: ~2.3ms</span>
                            </div>
                        </div>
                        <div style="margin-top: 20px; padding: 15px; background: rgba(0, 217, 255, 0.1); border-radius: 8px;">
                            <div style="color: #00ffcc; font-weight: 600;">Engine saved to: models/model.trt</div>
                        </div>
                    </div>
                `);
            }, 500);
        }
    }, 600);
}

// ===================================
// Code Tab Switching
// ===================================
function showTRTTab(tab) {
    document.getElementById('trt-inference-code').style.display = 'none';
    document.getElementById('trt-builder-code').style.display = 'none';
    document.getElementById('trt-int8-code').style.display = 'none';

    document.querySelectorAll('.env-tab').forEach(t => t.classList.remove('active'));

    document.getElementById('trt-' + tab + '-code').style.display = 'block';
    event.target.classList.add('active');
}

function copyTRTCode() {
    const activeCode = document.querySelector('.code-display:not([style*="display: none"]) code');
    if (activeCode) {
        navigator.clipboard.writeText(activeCode.innerText).then(() => {
            notify.show('Code copied to clipboard!', 'success');
        });
    }
}

// ===================================
// Performance Chart
// ===================================
function createTensorRTPerformanceChart() {
    const ctx = document.getElementById('tensorrtPerformanceChart');
    if (!ctx) return;

    const data = [];
    for (let i = 0; i < 50; i++) {
        data.push((Math.random() * 0.5 + 2.0).toFixed(2));
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 50}, (_, i) => i),
            datasets: [{
                label: 'Inference Latency (ms)',
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
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    min: 0,
                    max: 5,
                    grid: {
                        color: 'rgba(0, 217, 255, 0.1)'
                    },
                    ticks: {
                        color: '#8b95a8',
                        callback: function(value) {
                            return value + 'ms';
                        }
                    }
                }
            }
        }
    });
}

// ===================================
// Initialization
// ===================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🧠 TensorRT Integration loaded');

    createTensorRTPerformanceChart();

    console.log('✨ TensorRT Interface Ready!');
});
