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

// ============================================================================
// Benchmark History Management
// ============================================================================
const MODELS_DIR = path.join(process.cwd(), 'models');
const HISTORY_FILE = path.join(MODELS_DIR, 'history.json');
const MAX_HISTORY_ENTRIES = 100;

/**
 * Ensure history file exists
 */
function ensureHistoryFile() {
    try {
        // Ensure models directory exists
        if (!fs.existsSync(MODELS_DIR)) {
            fs.mkdirSync(MODELS_DIR, { recursive: true });
        }
        
        // Ensure history file exists
        if (!fs.existsSync(HISTORY_FILE)) {
            fs.writeFileSync(HISTORY_FILE, JSON.stringify({ runs: [] }, null, 2), 'utf-8');
        }
    } catch (error) {
        console.error('[InferenceMonitor] Failed to ensure history file:', error.message);
    }
}

/**
 * Load history synchronously
 * @returns {Array} Array of history entries
 */
function loadHistorySync() {
    try {
        ensureHistoryFile();
        const data = fs.readFileSync(HISTORY_FILE, 'utf-8');
        const parsed = JSON.parse(data);
        return Array.isArray(parsed.runs) ? parsed.runs : [];
    } catch (error) {
        console.error('[InferenceMonitor] Failed to load history:', error.message);
        return [];
    }
}

/**
 * Save history synchronously
 * @param {Array} runs - Array of history entries
 */
function saveHistorySync(runs) {
    try {
        ensureHistoryFile();
        fs.writeFileSync(HISTORY_FILE, JSON.stringify({ runs }, null, 2), 'utf-8');
    } catch (error) {
        console.error('[InferenceMonitor] Failed to save history:', error.message);
    }
}

/**
 * Append a new entry to history
 * @param {object} entry - Benchmark result entry
 */
function appendHistoryEntry(entry) {
    try {
        const runs = loadHistorySync();
        runs.unshift(entry); // Add to beginning
        
        // Keep only MAX_HISTORY_ENTRIES
        if (runs.length > MAX_HISTORY_ENTRIES) {
            runs.length = MAX_HISTORY_ENTRIES;
        }
        
        saveHistorySync(runs);
        console.log('[InferenceMonitor] History entry saved');
    } catch (error) {
        console.error('[InferenceMonitor] Failed to append history:', error.message);
    }
}

/**
 * Get all history entries
 * @returns {Array} Array of history entries
 */
function getHistory() {
    return loadHistorySync();
}

class InferenceMonitor {
    constructor() {
        this.exePath = null;
        this.tensorrtPath = null;
        this.isRunning = false;
        this.lastResult = null;
        
        // Model Manager state
        this.modelsConfig = null;
        this.selectedModelId = null;
        this.selectedMode = 'tensorrt';
        
        this.findTensorRTPath();
        this.findExecutable();
        this.loadModelsConfig();
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
     * Load models configuration from JSON file
     */
    loadModelsConfig() {
        const configPath = path.join(__dirname, '..', 'models', 'models.json');
        const modelsDir = path.join(__dirname, '..', 'models');
        
        try {
            if (fs.existsSync(configPath)) {
                const data = fs.readFileSync(configPath, 'utf-8');
                this.modelsConfig = JSON.parse(data);
                
                // Resolve relative paths to absolute
                this.modelsConfig.models = this.modelsConfig.models.map(model => ({
                    ...model,
                    absolutePath: path.isAbsolute(model.path) 
                        ? model.path 
                        : path.join(__dirname, '..', model.path),
                    exists: fs.existsSync(
                        path.isAbsolute(model.path) 
                            ? model.path 
                            : path.join(__dirname, '..', model.path)
                    )
                }));
                
                this.selectedModelId = this.modelsConfig.selectedModelId || null;
                this.selectedMode = this.modelsConfig.defaultMode || 'tensorrt';
                
                console.log(`[InferenceMonitor] Loaded ${this.modelsConfig.models.length} models from config`);
            } else {
                // Create default config by scanning models directory
                this.modelsConfig = { models: [], selectedModelId: null, defaultMode: 'tensorrt' };
                this.scanModelsDirectory(modelsDir);
            }
        } catch (error) {
            console.error('[InferenceMonitor] Failed to load models config:', error.message);
            this.modelsConfig = { models: [], selectedModelId: null, defaultMode: 'tensorrt' };
        }
    }
    
    /**
     * Scan models directory and add found models
     */
    scanModelsDirectory(modelsDir) {
        if (!fs.existsSync(modelsDir)) {
            return;
        }
        
        const files = fs.readdirSync(modelsDir);
        for (const file of files) {
            if (file === 'models.json') continue;
            
            const fullPath = path.join(modelsDir, file);
            const ext = path.extname(file).toLowerCase();
            
            if (ext === '.engine') {
                this.modelsConfig.models.push({
                    id: file.replace('.engine', '-engine'),
                    name: file.replace('.engine', ' TensorRT Engine'),
                    description: 'TensorRT optimized engine',
                    type: 'engine',
                    path: `models/${file}`,
                    absolutePath: fullPath,
                    exists: true
                });
            } else if (ext === '.onnx') {
                this.modelsConfig.models.push({
                    id: file.replace('.onnx', '-onnx'),
                    name: file.replace('.onnx', ' ONNX Model'),
                    description: 'ONNX model file',
                    type: 'onnx',
                    path: `models/${file}`,
                    absolutePath: fullPath,
                    exists: true
                });
            }
        }
        
        // Set first model as selected if none set
        if (this.modelsConfig.models.length > 0 && !this.selectedModelId) {
            this.selectedModelId = this.modelsConfig.models[0].id;
        }
    }
    
    /**
     * Get all models with their status
     * @returns {object} - { models: [...], selectedModelId, selectedMode }
     */
    getModels() {
        return {
            models: this.modelsConfig.models || [],
            selectedModelId: this.selectedModelId,
            selectedMode: this.selectedMode
        };
    }
    
    /**
     * Select a model by ID
     * @param {string} modelId - The model ID to select
     * @returns {object} - Updated models state
     */
    selectModel(modelId) {
        const model = this.modelsConfig.models.find(m => m.id === modelId);
        if (model) {
            this.selectedModelId = modelId;
            console.log(`[InferenceMonitor] Selected model: ${model.name}`);
            return { success: true, selectedModelId: modelId, model };
        }
        return { success: false, error: `Model not found: ${modelId}` };
    }
    
    /**
     * Set the benchmark mode
     * @param {string} mode - 'cuda', 'mock', 'tensorrt', or 'onnx'
     */
    setMode(mode) {
        const validModes = ['cuda', 'mock', 'tensorrt', 'onnx'];
        if (validModes.includes(mode)) {
            this.selectedMode = mode;
            return { success: true, mode };
        }
        return { success: false, error: `Invalid mode: ${mode}` };
    }
    
    /**
     * Get the currently selected model object
     */
    getSelectedModel() {
        if (!this.selectedModelId) return null;
        return this.modelsConfig.models.find(m => m.id === this.selectedModelId) || null;
    }
    
    /**
     * Run inference benchmark
     * Uses the selected model and mode, or accepts overrides via options
     * 
     * @param {string} mode - 'cuda', 'mock', 'tensorrt', or 'onnx' (optional, uses selectedMode if not provided)
     * @param {object} options - Additional options
     *   - modelId: Override selected model by ID
     *   - enginePath: Direct path to .engine file (for tensorrt mode)
     *   - modelPath: Direct path to .onnx file (for onnx mode)
     *   - warmupRuns: Number of warmup iterations
     *   - benchmarkRuns: Number of benchmark iterations
     * @returns {Promise<object>} - Inference result JSON
     */
    async runBenchmark(mode = null, options = {}) {
        if (this.isRunning) {
            return {
                success: false,
                error: 'Benchmark already running',
                mode: mode || this.selectedMode
            };
        }
        
        if (!this.exePath) {
            return this.getFallbackResult(mode || this.selectedMode, 'Executable not found');
        }
        
        // Determine mode and model
        const effectiveMode = mode || this.selectedMode || 'cuda';
        let effectiveModel = null;
        
        // Get model from options or use selected
        if (options.modelId) {
            effectiveModel = this.modelsConfig.models.find(m => m.id === options.modelId);
        } else if (effectiveMode === 'tensorrt' || effectiveMode === 'onnx') {
            effectiveModel = this.getSelectedModel();
        }
        
        this.isRunning = true;
        
        const args = [];
        
        // Set mode based on type
        switch (effectiveMode) {
            case 'cuda':
                args.push('--cuda');
                break;
            case 'mock':
                args.push('--mock');
                break;
            case 'tensorrt':
                // Use direct path or model's absolutePath
                const enginePath = options.enginePath || (effectiveModel?.absolutePath);
                if (enginePath && fs.existsSync(enginePath)) {
                    args.push('--tensorrt', enginePath);
                } else {
                    this.isRunning = false;
                    return this.getFallbackResult(effectiveMode, 'Engine file not found: ' + (enginePath || 'No model selected'));
                }
                break;
            case 'onnx':
                // Use direct path or model's absolutePath
                const onnxPath = options.modelPath || (effectiveModel?.absolutePath);
                if (onnxPath && fs.existsSync(onnxPath)) {
                    args.push('--onnx', onnxPath);
                } else {
                    this.isRunning = false;
                    return this.getFallbackResult(effectiveMode, 'ONNX file not found: ' + (onnxPath || 'No model selected'));
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
        
        console.log(`[InferenceMonitor] Running: ${this.exePath} ${args.join(' ')}`);
        
        return new Promise((resolve) => {
            execFile(this.exePath, args, { timeout: 120000, env }, (error, stdout, stderr) => {
                this.isRunning = false;
                
                if (error) {
                    console.error('[InferenceMonitor] Execution error:', error.message);
                    resolve(this.getFallbackResult(effectiveMode, error.message));
                    return;
                }
                
                // Extract JSON from output (may have TensorRT warnings before it)
                const jsonMatch = stdout.match(/\{[\s\S]*\}/);
                if (!jsonMatch) {
                    console.error('[InferenceMonitor] No JSON found in output:', stdout);
                    resolve(this.getFallbackResult(effectiveMode, 'No JSON in output'));
                    return;
                }
                
                try {
                    const result = JSON.parse(jsonMatch[0]);
                    result.timestamp = new Date().toISOString();
                    result.selectedModelId = effectiveModel?.id || null;
                    result.selectedModelName = effectiveModel?.name || null;
                    this.lastResult = result;
                    
                    // Save to persistent history
                    if (result.success) {
                        appendHistoryEntry({
                            timestamp: result.timestamp,
                            mode: result.mode,
                            model: result.model || result.selectedModelName || 'Unknown',
                            throughputFPS: result.throughputFPS,
                            latencyMs: result.latencyMs,
                            minLatencyMs: result.minLatencyMs,
                            maxLatencyMs: result.maxLatencyMs,
                            p95LatencyMs: result.p95LatencyMs,
                            precision: result.precision,
                            device: result.device || result.gpu || 'Unknown GPU'
                        });
                    }
                    
                    resolve(result);
                } catch (parseError) {
                    console.error('[InferenceMonitor] JSON parse error:', parseError.message);
                    console.error('[InferenceMonitor] Raw output:', stdout);
                    resolve(this.getFallbackResult(effectiveMode, 'Failed to parse output'));
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
        const selectedModel = this.getSelectedModel();
        return {
            available: this.exePath !== null,
            tensorrtPath: this.tensorrtPath,
            executablePath: this.exePath,
            modes: this.getSupportedModes(),
            selectedModelId: this.selectedModelId,
            selectedModelName: selectedModel?.name || null,
            selectedMode: this.selectedMode,
            modelsCount: this.modelsConfig.models.length
        };
    }
    
    /**
     * List available models - DEPRECATED, use getModels() instead
     */
    listModels() {
        return this.getModels();
    }
    
    /**
     * Add a new model to the configuration
     * @param {object} model - Model definition
     */
    addModel(model) {
        if (!model.id || !model.path || !model.type) {
            return { success: false, error: 'Model requires id, path, and type' };
        }
        
        const absolutePath = path.isAbsolute(model.path) 
            ? model.path 
            : path.join(__dirname, '..', model.path);
        
        const newModel = {
            id: model.id,
            name: model.name || model.id,
            description: model.description || '',
            type: model.type,
            path: model.path,
            absolutePath,
            exists: fs.existsSync(absolutePath),
            inputShape: model.inputShape || [1, 3, 224, 224],
            precision: model.precision || 'FP32'
        };
        
        // Check if model already exists
        const existingIndex = this.modelsConfig.models.findIndex(m => m.id === model.id);
        if (existingIndex >= 0) {
            this.modelsConfig.models[existingIndex] = newModel;
        } else {
            this.modelsConfig.models.push(newModel);
        }
        
        this.saveModelsConfig();
        return { success: true, model: newModel };
    }
    
    /**
     * Save models configuration to JSON file
     */
    saveModelsConfig() {
        const configPath = path.join(__dirname, '..', 'models', 'models.json');
        
        // Prepare config for saving (remove absolutePath and exists)
        const saveConfig = {
            models: this.modelsConfig.models.map(m => ({
                id: m.id,
                name: m.name,
                description: m.description,
                type: m.type,
                path: m.path,
                inputShape: m.inputShape,
                precision: m.precision
            })),
            selectedModelId: this.selectedModelId,
            defaultMode: this.selectedMode
        };
        
        try {
            fs.writeFileSync(configPath, JSON.stringify(saveConfig, null, 2), 'utf-8');
            console.log('[InferenceMonitor] Saved models config');
            return { success: true };
        } catch (error) {
            console.error('[InferenceMonitor] Failed to save config:', error.message);
            return { success: false, error: error.message };
        }
    }
}

// Export singleton instance
const inferenceMonitor = new InferenceMonitor();

module.exports = { InferenceMonitor, inferenceMonitor, getHistory };
