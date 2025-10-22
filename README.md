# PCA-EXP-3-PARALLEL-REDUCTION-USING-UNROLLING-TECHNIQUES AY 23-24
<h3>NAME : Sriram Gopalan G</h3>
<h3>REGISTER NO: 212222230149</h3>
<h3>EX. NO : 3</h3>
<h3>DATE:          </h3>

<h1> <align=center> PARALLEL REDUCTION USING UNROLLING TECHNIQUES </h3>
  Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.</h3>

## AIM:
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Initialization and Memory Allocation
2.	Define the input size n.
3.	Allocate host memory (h_idata and h_odata) for input and output data.
Input Data Initialization
4.	Initialize the input data on the host (h_idata) by assigning a value of 1 to each element.
Device Memory Allocation
5.	Allocate device memory (d_idata and d_odata) for input and output data on the GPU.
Data Transfer: Host to Device
6.	Copy the input data from the host (h_idata) to the device (d_idata) using cudaMemcpy.
Grid and Block Configuration
7.	Define the grid and block dimensions for the kernel launch:
8.	Each block consists of 256 threads.
9.	Calculate the grid size based on the input size n and block size.
10.	Start CPU Timer
11.	Initialize a CPU timer to measure the CPU execution time.
12.	Compute CPU Sum
13.	Calculate the sum of the input data on the CPU using a for loop and store the result in sum_cpu.
14.	Stop CPU Timer
15.	Record the elapsed CPU time.
16.	Start GPU Timer
17.	Initialize a GPU timer to measure the GPU execution time.
Kernel Execution
18.	Launch the reduceUnrolling16 kernel on the GPU with the specified grid and block dimensions.
Data Transfer: Device to Host
19.	Copy the result data from the device (d_odata) to the host (h_odata) using cudaMemcpy.
20.	Compute GPU Sum
21.	Calculate the final sum on the GPU by summing the elements in h_odata and store the result in sum_gpu.
22.	Stop GPU Timer
23.	Record the elapsed GPU time.
24.	Print Results
25.	Display the computed CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.
Memory Deallocation
26.	Free the allocated host and device memory using free and cudaFree.
27.	Exit
28.	Return from the main function.

## PROGRAM:
### Unrolling8
```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```
```
%%writefile reduction_unroll8.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) {                                                    \
    const cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                          \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                    \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                         \
    }                                                                    \
}

#define BDIM 512

// GPU reduction kernel with unrolling-8
__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n) {
    __shared__ int smem[BDIM];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int sum = 0;
    if (idx + 7 * blockDim.x < n) {
        int *ptr = g_idata + idx;
        sum += ptr[0] + ptr[blockDim.x] + ptr[2 * blockDim.x] + ptr[3 * blockDim.x] +
               ptr[4 * blockDim.x] + ptr[5 * blockDim.x] + ptr[6 * blockDim.x] + ptr[7 * blockDim.x];
    }

    smem[tid] = sum;
    __syncthreads();

    // in-block reduction
    if (BDIM >= 1024 && tid < 512) smem[tid] += smem[tid + 512]; __syncthreads();
    if (BDIM >= 512  && tid < 256) smem[tid] += smem[tid + 256]; __syncthreads();
    if (BDIM >= 256  && tid < 128) smem[tid] += smem[tid + 128]; __syncthreads();
    if (BDIM >= 128  && tid < 64)  smem[tid] += smem[tid + 64];  __syncthreads();

    if (tid < 32) {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

// CPU reference reduction
int cpuReduce(int *h_idata, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) sum += h_idata[i];
    return sum;
}

int main(int argc, char **argv) {
    int size = 1 << 24;  // 16M elements
    int bytes = size * sizeof(int);

    int *h_idata = (int *)malloc(bytes);
    for (int i = 0; i < size; i++) h_idata[i] = (int)(rand() & 0xFF);

    int *d_idata, *d_odata;
    CHECK(cudaMalloc((void **)&d_idata, bytes));
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    int grid = (size + BDIM * 8 - 1) / (BDIM * 8);
    CHECK(cudaMalloc((void **)&d_odata, grid * sizeof(int)));

    // Launch kernel
    reduceUnrolling8<<<grid, BDIM>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());

    // Copy partial results back
    int *h_odata = (int *)malloc(grid * sizeof(int));
    CHECK(cudaMemcpy(h_odata, d_odata, grid * sizeof(int), cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    int gpu_sum = 0;
    for (int i = 0; i < grid; i++) gpu_sum += h_odata[i];

    int cpu_sum = cpuReduce(h_idata, size);

    printf("CPU sum: %d\n", cpu_sum);
    printf("GPU sum: %d\n", gpu_sum);

    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}
```
```
!nvcc -arch=sm_75 reduction_unroll8.cu -o reduction
!./reduction
```

### Unrolling16

```
%%writefile reduction_unroll8_16_simple.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) {                                                    \
    const cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                          \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                    \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                         \
    }                                                                    \
}

#define BDIM 512

// ----------- Unroll-8 kernel -----------
__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// ----------- Unroll-16 kernel -----------

__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    // convert global data pointer to local block pointer
    int *idata = g_idata + blockIdx.x * blockDim.x * 16;

    // unrolling 16
    if (idx + 15 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        int c1 = g_idata[idx + 8 * blockDim.x];
        int c2 = g_idata[idx + 9 * blockDim.x];
        int c3 = g_idata[idx + 10 * blockDim.x];
        int c4 = g_idata[idx + 11 * blockDim.x];
        int d1 = g_idata[idx + 12 * blockDim.x];
        int d2 = g_idata[idx + 13 * blockDim.x];
        int d3 = g_idata[idx + 14 * blockDim.x];
        int d4 = g_idata[idx + 15 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 +
                       b1 + b2 + b3 + b4 +
                       c1 + c2 + c3 + c4 +
                       d1 + d2 + d3 + d4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result of this block
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


// ----------- CPU reference reduction -----------
int cpuReduce(int *h_idata, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) sum += h_idata[i];
    return sum;
}

int main() {
    int size = 1 << 24;  // 16M elements
    int bytes = size * sizeof(int);

    // Allocate and initialize host array
    int *h_idata = (int *)malloc(bytes);
    for (int i = 0; i < size; i++)
        h_idata[i] = rand() & 0xFF;

    int *d_idata, *d_odata;
    CHECK(cudaMalloc((void **)&d_idata, bytes));

    // Grid sizes for unroll-8 and unroll-16
    int grid8  = (size + BDIM * 8 - 1) / (BDIM * 8);
    int grid16 = (size + BDIM * 16 - 1) / (BDIM * 16);
    CHECK(cudaMalloc((void **)&d_odata, grid8 * sizeof(int))); // max needed

    // ---------------- CPU reduction ----------------
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    int cpu_sum = cpuReduce(h_idata, size);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float cpuTime;
    CHECK(cudaEventElapsedTime(&cpuTime, start, stop));

    // ---------------- GPU Unroll-8 ----------------
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    reduceUnrolling8<<<grid8, BDIM>>>(d_idata, d_odata, size);
    cudaError_t err8 = cudaGetLastError();
    if (err8 != cudaSuccess) {
        printf("Kernel launch error (Unroll-8): %s\n", cudaGetErrorString(err8));
        return -1;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float gpuTime8;
    CHECK(cudaEventElapsedTime(&gpuTime8, start, stop));

    int *h_odata = (int *)malloc(grid8 * sizeof(int));
    CHECK(cudaMemcpy(h_odata, d_odata, grid8 * sizeof(int), cudaMemcpyDeviceToHost));
    int gpu_sum8 = 0;
    for (int i = 0; i < grid8; i++) gpu_sum8 += h_odata[i];

    // ---------------- GPU Unroll-16 ----------------
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    reduceUnrolling16<<<grid16, BDIM>>>(d_idata, d_odata, size);
    cudaError_t err16 = cudaGetLastError();
    if (err16 != cudaSuccess) {
        printf("Kernel launch error (Unroll-16): %s\n", cudaGetErrorString(err16));
        return -1;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float gpuTime16;
    CHECK(cudaEventElapsedTime(&gpuTime16, start, stop));

    CHECK(cudaMemcpy(h_odata, d_odata, grid16 * sizeof(int), cudaMemcpyDeviceToHost));
    int gpu_sum16 = 0;
    for (int i = 0; i < grid16; i++) gpu_sum16 += h_odata[i];

    // ---------------- Print results ----------------
    printf("CPU sum   : %d, time %.5f sec\n", cpu_sum, cpuTime / 1000.0f);
    printf("GPU sum-8 : %d, time %.5f ms\n", gpu_sum8, gpuTime8);
    printf("GPU sum-16: %d, time %.5f ms\n", gpu_sum16, gpuTime16);

    // ---------------- Cleanup ----------------
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

```
```
!nvcc -arch=sm_75 reduction_unroll8_16_simple.cu -o reduction
!./reduction
```

## OUTPUT:
### Unrolling8

![17611272030544486879597057801210](https://github.com/user-attachments/assets/04555c55-88cb-4740-b2a0-e4c97a93000f)

### Unrolling16



## RESULT:
Thus the program has been executed by unrolling by 8 and unrolling by 16. It is observed that `Unrolling16` has executed with less elapsed time than `Unrolling8` with blocks `2139353471`,`2139353471`.
