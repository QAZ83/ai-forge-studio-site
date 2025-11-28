/**
 * AI Forge Studio - Inference Monitor Service
 * Runs trt_inference_json.exe and returns benchmark results
 * 
 * Modes:
 *   --cuda      : Real CUDA compute benchmark (matrix multiplication)
 *   --mock      : Simulated TensorRT inference
 *   --tensorrt  : Load and run TensorRT .engine file
 *   --onnx      : Convert ONNX model to TensorRT engine and run
 */

const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

class InferenceMonitor {
    constructor() {
        this.exePath = null;
        this.tensorrtPath = null;
        this.isRunning = false;
        this.lastResult = null;
        
        this.findTensorRTPath();
        this.findExecutable();
    }
    
    /**
     * Find TensorRT installation path for DLLs
     */
    findTensorRTPath() {
        const possiblePaths = [
            // Project distribution path
            path.join(__dirname, '..', 'dist', 'TensorRT-10.14.1.48', 'bin'),
            // Standard installation paths
            'C:\\TensorRT\\bin',
            'C:\\Program Files\\NVIDIA\\TensorRT\\bin',
            // Environment variable
            process.env.TENSORRT_PATH ? path.join(process.env.TENSORRT_PATH, 'bin') : null,
        ].filter(Boolean);
        
        for (const trtPath of possiblePaths) {
            if (fs.existsSync(trtPath) && fs.existsSync(path.join(trtPath, 'nvinfer_10.dll'))) {
                this.tensorrtPath = trtPath;
                console.log(`[InferenceMonitor] Found TensorRT: ${trtPath}`);
                return;
            }
        }
        
        console.warn('[InferenceMonitor] TensorRT DLLs not found.');
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
     * @param {string} mode - 'cuda', 'mock', 'tensorrt', or 'onnx'
     * @param {object} options - Additional options
     *   - enginePath: Path to .engine file (for tensorrt mode)
     *   - modelPath: Path to .onnx file (for onnx mode)
     *   - warmupRuns: Number of warmup iterations
     *   - benchmarkRuns: Number of benchmark iterations
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
        
        // Set mode based on type
        switch (mode) {
            case 'cuda':
                args.push('--cuda');
                break;
            case 'mock':
                args.push('--mock');
                break;
            case 'tensorrt':
                if (options.enginePath && fs.existsSync(options.enginePath)) {
                    args.push('--tensorrt', options.enginePath);
                } else {
                    this.isRunning = false;
                    return this.getFallbackResult(mode, 'Engine file not found: ' + options.enginePath);
                }
                break;
            case 'onnx':
                if (options.modelPath && fs.existsSync(options.modelPath)) {
                    args.push('--onnx', options.modelPath);
                } else {
                    this.isRunning = false;
                    return this.getFallbackResult(mode, 'ONNX file not found: ' + options.modelPath);
                }
                break;
            default:
                args.push('--mock'); // Default to mock
        }
        
        // Optional: warmup and benchmark runs
        if (options.warmupRuns) {
            args.push('--warmup', options.warmupRuns.toString());
        }
        if (options.benchmarkRuns) {
            args.push('--runs', options.benchmarkRuns.toString());
        }
        
        // Prepare environment with TensorRT DLL path
        const env = { ...process.env };
        if (this.tensorrtPath) {
            env.PATH = this.tensorrtPath + path.delimiter + (env.PATH || '');
        }
        
        return new Promise((resolve) => {
            execFile(this.exePath, args, { timeout: 120000, env }, (error, stdout, stderr) => {
                this.isRunning = false;
                
                if (error) {
                    console.error('[InferenceMonitor] Execution error:', error.message);
                    resolve(this.getFallbackResult(mode, error.message));
                    return;
                }
                
                // Extract JSON from output (may have TensorRT warnings before it)
                const jsonMatch = stdout.match(/\{[\s\S]*\}/);
                if (!jsonMatch) {
                    console.error('[InferenceMonitor] No JSON found in output:', stdout);
                    resolve(this.getFallbackResult(mode, 'No JSON in output'));
                    return;
                }
                
                try {
                    const result = JSON.parse(jsonMatch[0]);
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
            { id: 'cuda', name: 'CUDA Compute Benchmark', description: 'Real GPU matrix multiplication', requiresFile: false },
            { id: 'mock', name: 'TensorRT Simulation', description: 'Simulated ResNet-50 FP16 inference', requiresFile: false },
            { id: 'tensorrt', name: 'TensorRT Engine', description: 'Load and run optimized .engine file', requiresFile: true, fileType: '.engine' },
            { id: 'onnx', name: 'ONNX Model', description: 'Convert ONNX to TensorRT and benchmark', requiresFile: true, fileType: '.onnx' }
        ];
    }
    
    /**
     * Get TensorRT availability status
     */
    getStatus() {
        return {
            available: this.exePath !== null,
            tensorrtPath: this.tensorrtPath,
            executablePath: this.exePath,
            modes: this.getSupportedModes()
        };
    }
    
    /**
     * List available models in the models directory
     */
    listModels() {
        const modelsDir = path.join(__dirname, '..', 'models');
        const result = { engines: [], onnx: [] };
        
        if (!fs.existsSync(modelsDir)) {
            return result;
        }
        
        const files = fs.readdirSync(modelsDir);
        for (const file of files) {
            const fullPath = path.join(modelsDir, file);
            if (file.endsWith('.engine')) {
                result.engines.push({ name: file, path: fullPath });
            } else if (file.endsWith('.onnx')) {
                result.onnx.push({ name: file, path: fullPath });
            }
        }
        
        return result;
    }
}

// Export singleton instance
const inferenceMonitor = new InferenceMonitor();

module.exports = { InferenceMonitor, inferenceMonitor };
