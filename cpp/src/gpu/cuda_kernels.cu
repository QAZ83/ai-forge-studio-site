/**
 * AI Forge Studio - CUDA Kernels
 * Author: M.3R3
 * 
 * GPU compute kernels for various operations.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Block size for kernel launches
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 16;

/**
 * Vector Addition Kernel
 * C = A + B
 */
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * Matrix Multiplication Kernel (Tiled)
 * C = A * B (all matrices are NxN)
 */
__global__ void matrixMultiplyKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;
        
        if (row < n && tiledCol < n) {
            tileA[threadIdx.y][threadIdx.x] = a[row * n + tiledCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tiledRow < n && col < n) {
            tileB[threadIdx.y][threadIdx.x] = b[tiledRow * n + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

/**
 * Neural Network Dense Layer Kernel
 * output = ReLU(input @ weights + bias)
 */
__global__ void neuralLayerKernel(
    const float* input,      // [batchSize, inputSize]
    const float* weights,    // [inputSize, outputSize]
    const float* bias,       // [outputSize]
    float* output,           // [batchSize, outputSize]
    int inputSize,
    int outputSize,
    int batchSize
) {
    int batch = blockIdx.y;
    int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batchSize && outIdx < outputSize) {
        float sum = bias[outIdx];
        
        // Compute dot product for this output neuron
        for (int i = 0; i < inputSize; ++i) {
            sum += input[batch * inputSize + i] * weights[i * outputSize + outIdx];
        }
        
        // ReLU activation
        output[batch * outputSize + outIdx] = fmaxf(0.0f, sum);
    }
}

/**
 * GELU Activation Kernel (used in transformers)
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
__global__ void geluKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = data[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        data[idx] = x * cdf;
    }
}

/**
 * Softmax Kernel (for single row)
 */
__global__ void softmaxKernel(float* data, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and find max
    float val = (idx < n) ? data[idx] : -INFINITY;
    sdata[tid] = val;
    __syncthreads();
    
    // Reduction to find max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float maxVal = sdata[0];
    __syncthreads();
    
    // Compute exp(x - max)
    val = (idx < n) ? expf(data[idx] - maxVal) : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    
    // Reduction to find sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum = sdata[0];
    
    // Normalize
    if (idx < n) {
        data[idx] = val / sum;
    }
}

/**
 * Layer Normalization Kernel
 */
__global__ void layerNormKernel(
    float* data,
    const float* gamma,
    const float* beta,
    int n,
    float epsilon
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    float val = (idx < n) ? data[idx] : 0.0f;
    sdata[tid] = val;
    sdata[tid + blockDim.x] = val * val;
    __syncthreads();
    
    // Reduction for mean and variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }
    
    float mean = sdata[0] / n;
    float variance = sdata[blockDim.x] / n - mean * mean;
    float invStd = rsqrtf(variance + epsilon);
    
    // Normalize and apply gamma/beta
    if (idx < n) {
        float normalized = (val - mean) * invStd;
        data[idx] = gamma[idx] * normalized + beta[idx];
    }
}

// ==============================================================================
// Kernel Launch Wrappers (C-linkage for external calls)
// ==============================================================================

extern "C" {

void launchVectorAdd(float* a, float* b, float* c, int n, cudaStream_t stream) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAddKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(a, b, c, n);
}

void launchMatrixMultiply(float* a, float* b, float* c, int n, cudaStream_t stream) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    matrixMultiplyKernel<<<gridDim, blockDim, 0, stream>>>(a, b, c, n);
}

void launchNeuralLayer(
    float* input, float* weights, float* bias, float* output,
    int inputSize, int outputSize, int batchSize, cudaStream_t stream
) {
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE, batchSize);
    neuralLayerKernel<<<gridDim, blockDim, 0, stream>>>(
        input, weights, bias, output, inputSize, outputSize, batchSize
    );
}

void launchGelu(float* data, int n, cudaStream_t stream) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    geluKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(data, n);
}

void launchSoftmax(float* data, int n, cudaStream_t stream) {
    int sharedMemSize = BLOCK_SIZE * sizeof(float);
    softmaxKernel<<<1, BLOCK_SIZE, sharedMemSize, stream>>>(data, n);
}

void launchLayerNorm(
    float* data, const float* gamma, const float* beta,
    int n, float epsilon, cudaStream_t stream
) {
    int sharedMemSize = 2 * BLOCK_SIZE * sizeof(float);
    layerNormKernel<<<1, BLOCK_SIZE, sharedMemSize, stream>>>(data, gamma, beta, n, epsilon);
}

} // extern "C"
