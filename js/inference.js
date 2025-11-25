/**
 * Model Inference JavaScript
 */

let inferenceCount = 0;
let currentPrecision = 'FP16';

function setPrecision(precision) {
    currentPrecision = precision;
    notify.show(`Precision set to ${precision}`, 'success');
}

function uploadFile() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*,.txt,.json';
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            notify.show(`File loaded: ${file.name}`, 'success');
            document.getElementById('input-data').value = `[File: ${file.name}]\nType: ${file.type}\nSize: ${(file.size / 1024).toFixed(2)} KB`;
        }
    };
    input.click();
}

function runInference() {
    const model = document.getElementById('model-select').value;
    notify.show('Running inference...', 'info');

    setTimeout(() => {
        inferenceCount++;
        document.getElementById('total-inferences').textContent = inferenceCount;

        const latency = (Math.random() * 1 + 1.5).toFixed(2);
        const throughput = (1000 / latency).toFixed(0);

        document.getElementById('latency-display').textContent = latency + ' ms';
        document.getElementById('throughput-display').textContent = throughput + ' FPS';
        document.getElementById('gpu-usage').textContent = (Math.random() * 5 + 95).toFixed(0) + '%';

        document.getElementById('output-results').innerHTML = `
            <div style="color: #00ff88; margin-bottom: 10px;">✓ Inference completed successfully</div>
            <div style="color: #8b95a8;">Model: ${model}</div>
            <div style="color: #8b95a8;">Precision: ${currentPrecision}</div>
            <div style="color: #8b95a8;">Latency: ${latency} ms</div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(0, 217, 255, 0.2);">
                <div style="color: #00ffcc;">Output Tensor Shape: [1, 1000]</div>
                <div style="color: #8b95a8; margin-top: 5px;">Top-5 predictions:</div>
                <div style="color: #00d9ff; margin-top: 5px;">1. Class 243: 0.95 confidence</div>
                <div style="color: #00d9ff;">2. Class 156: 0.03 confidence</div>
                <div style="color: #00d9ff;">3. Class 892: 0.01 confidence</div>
            </div>
        `;

        notify.show('Inference complete!', 'success');
    }, 1500);
}

function runBatchInference() {
    const batchSize = parseInt(document.getElementById('batch-size').value);
    const progressBar = document.getElementById('batch-bar');
    const progressDiv = document.getElementById('batch-progress');

    progressDiv.style.display = 'block';
    notify.show(`Starting batch inference (${batchSize} items)...`, 'info');

    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        progressBar.style.width = progress + '%';
        progressBar.textContent = progress + '%';

        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
                progressDiv.style.display = 'none';
                notify.show(`Batch inference complete! Processed ${batchSize} items`, 'success');
                inferenceCount += batchSize;
                document.getElementById('total-inferences').textContent = inferenceCount;
            }, 500);
        }
    }, 300);
}

// Create inference chart
document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('inferenceChart');
    if (!ctx) return;

    const data = Array.from({length: 50}, () => (Math.random() * 1 + 1.5).toFixed(2));

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 50}, (_, i) => i),
            datasets: [{
                label: 'Latency (ms)',
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
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: {
                    min: 0,
                    max: 5,
                    grid: { color: 'rgba(0, 217, 255, 0.1)' },
                    ticks: { color: '#8b95a8' }
                }
            }
        }
    });
});
