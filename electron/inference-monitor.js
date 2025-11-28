/**
 * AI Forge Studio - Inference Monitor Service
 * Runs trt_inference_json.exe and returns benchmark results
 * 
 * Modes:
 *   --cuda  : Real CUDA compute benchmark
 *   --mock  : Simulated TensorRT inference
 */

const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

class InferenceMonitor {
    constructor() {
        this.exePath = null;
        this.isRunning = false;
        this.lastResult = null;
        
        this.findExecutable();
    }
    
    /**
     * Find the trt_inference_json executable
     */
    findExecutable() {
        const possiblePaths = [
            // Development paths
            path.join(__dirname, '..', 'cpp-backend', 'build', 'bin', 'Release', 'trt_inference_json.exe'),
            path.join(__dirname, '..', 'cpp-backend', 'build', 'bin', 'Debug', 'trt_inference_json.exe'),
            // Production paths (packaged app)
            path.join(process.resourcesPath || '', 'bin', 'trt_inference_json.exe'),
            path.join(__dirname, '..', 'bin', 'trt_inference_json.exe'),
            // Current directory
            path.join(process.cwd(), 'cpp-backend', 'build', 'bin', 'Release', 'trt_inference_json.exe'),
        ];
        
        for (const exePath of possiblePaths) {
            if (fs.existsSync(exePath)) {
                this.exePath = exePath;
                console.log(`[InferenceMonitor] Found executable: ${exePath}`);
                return;
            }
        }
        
        console.warn('[InferenceMonitor] trt_inference_json.exe not found.');
        console.warn('[InferenceMonitor] Searched paths:', possiblePaths);
    }
    
    /**
     * Run inference benchmark
     * @param {string} mode - 'cuda', 'mock', or 'tensorrt'
     * @param {object} options - Additional options
     * @returns {Promise<object>} - Inference result JSON
     */
    async runBenchmark(mode = 'cuda', options = {}) {
        if (this.isRunning) {
            return {
                success: false,
                error: 'Benchmark already running',
                mode: mode
            };
        }
        
        if (!this.exePath) {
            return this.getFallbackResult(mode, 'Executable not found');
        }
        
        this.isRunning = true;
        
        const args = [];
        
        // Set mode
        if (mode === 'cuda') {
            args.push('--cuda');
        } else if (mode === 'mock') {
            args.push('--mock');
        } else if (mode === 'tensorrt' && options.enginePath) {
            args.push(options.enginePath);
        } else {
            args.push('--mock'); // Default to mock
        }
        
        // Optional: warmup and benchmark runs
        if (options.warmupRuns) {
            args.push('--warmup', options.warmupRuns.toString());
        }
        if (options.benchmarkRuns) {
            args.push('--runs', options.benchmarkRuns.toString());
        }
        
        return new Promise((resolve) => {
            execFile(this.exePath, args, { timeout: 60000 }, (error, stdout, stderr) => {
                this.isRunning = false;
                
                if (error) {
                    console.error('[InferenceMonitor] Execution error:', error.message);
                    resolve(this.getFallbackResult(mode, error.message));
                    return;
                }
                
                try {
                    const result = JSON.parse(stdout.trim());
                    result.timestamp = new Date().toISOString();
                    this.lastResult = result;
                    resolve(result);
                } catch (parseError) {
                    console.error('[InferenceMonitor] JSON parse error:', parseError.message);
                    console.error('[InferenceMonitor] Raw output:', stdout);
                    resolve(this.getFallbackResult(mode, 'Failed to parse output'));
                }
            });
        });
    }
    
    /**
     * Get fallback result when executable is not available
     */
    getFallbackResult(mode, error) {
        return {
            success: false,
            error: error,
            mode: mode,
            model: 'N/A',
            precision: 'N/A',
            batchSize: 0,
            inputShape: [0, 0, 0, 0],
            latencyMs: 0,
            throughputFPS: 0,
            inferenceMemoryMB: 0,
            device: 'Unknown',
            driver: 'N/A',
            cudaVersion: 'N/A',
            timestamp: new Date().toISOString()
        };
    }
    
    /**
     * Get last cached result
     */
    getLastResult() {
        return this.lastResult;
    }
    
    /**
     * Check if executable is available
     */
    isAvailable() {
        return this.exePath !== null;
    }
    
    /**
     * Get supported modes
     */
    getSupportedModes() {
        return [
            { id: 'cuda', name: 'CUDA Compute Benchmark', description: 'Real GPU matrix multiplication' },
            { id: 'mock', name: 'TensorRT Simulation', description: 'Simulated ResNet-50 FP16 inference' },
            { id: 'tensorrt', name: 'TensorRT Engine', description: 'Load and run .engine file' }
        ];
    }
}

// Export singleton instance
const inferenceMonitor = new InferenceMonitor();

module.exports = { InferenceMonitor, inferenceMonitor };
