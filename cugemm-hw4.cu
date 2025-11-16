/*
HW4
- I used
    - NVIDIA GeForce GTX 1060 6GB,
    - NVIDIA-SMI 550.144.03
    - Driver Version: 550.144.03
    - CUDA Version: 12.4

- Implemented CUDA streams to overlap memory copies with kernel computation. Building on HW3
    - Copy entire B matrix to device once (all outputs depend on all of B)
    - Split A and C into row slices across NUM_STREAMS
    - Each stream asynchronously: transfers A slice, transfers C slice, launches kernel,
      copies result back
    - This overlaps data transfers with kernel execution for performance improvement

    | Algorithm        | Size | Reps | Streams | Time (s) | GFLOPS  | Speedup  |
    |------------------|------|------|---------|----------|---------|----------|
    | 1stream baseline | 1024 | 25   | 1       | 0.003324 |  646.11 | -----    |
    | multistream      | 1024 | 25   | 1       | 0.003394 |  632.82 | 0.98x    |
    | multistream      | 1024 | 25   | 2       | 0.002654 |  809.04 | 1.25x    |
    | multistream      | 1024 | 25   | 4       | 0.002354 |  912.44 | 1.41x    |
    | multistream      | 1024 | 25   | 8       | 0.002254 |  952.82 | 1.47x    |
    | 1stream baseline | 2048 | 25   | 1       | 0.019824 |  866.62 | -----    |
    | multistream      | 2048 | 25   | 1       | 0.019499 |  881.08 | 1.02x    |
    | multistream      | 2048 | 25   | 2       | 0.017865 |  961.65 | 1.11x    |
    | multistream      | 2048 | 25   | 4       | 0.016660 | 1031.20 | 1.19x    |
    | multistream      | 2048 | 25   | 8       | 0.016294 | 1054.35 | 1.22x    |
    | 1stream baseline | 4096 | 25   | 1       | 0.137596 |  998.86 | -----    |
    | multistream      | 4096 | 25   | 1       | 0.138877 |  989.64 | 0.99x    |
    | multistream      | 4096 | 25   | 2       | 0.131524 | 1044.97 | 1.05x    |
    | multistream      | 4096 | 25   | 4       | 0.127531 | 1077.69 | 1.08x    |
    | multistream      | 4096 | 25   | 8       | 0.125857 | 1092.03 | 1.09x    |
*/

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

// from https://github.com/jarro2783/cxxopts
#include "cxxopts.hpp"

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))
#define ROUND_UP_TO_NEAREST(M, N) (((M) + (N) - 1) / (N))

enum Algo {
    cublas = 0,
    basic,
    gmem_coalesced,
    smem,
    smem_multioutput,
    smem_multioutput_1stream,
    smem_multioutput_multistream,
    numAlgos
};

const char* algo2str(Algo a) {
    switch (a) {
        case cublas:
            return "cublas";
        case basic:
            return "basic";
        case gmem_coalesced:
            return "gmem_coalesced";
        case smem:
            return "sharedmem";
        case smem_multioutput:
            return "sharedmem_multioutput";
        case smem_multioutput_1stream:
            return "sharedmem_multioutput_1stream";
        case smem_multioutput_multistream:
            return "sharedmem_multioutput_multistream";
        default:
            return "INVALID";
    }
}

void cudaErrorCheck(cudaError_t error, const char* file, int line);
void cublasErrorCheck(cublasStatus_t status, const char* file, int line);
void randomize_matrix(float* mat, int N);
void const_init_matrix(float* mat, int N, float F);
bool verify_matrix(float* expected, float* actual, int M, int N);
void print_matrix(const float* A, int M, int N, std::ostream& outs);
void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha, float* A, float* B,
             float beta, float* C, uint NUM_STREAMS, float* hA, float* hB, float* hC);
void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha, float* A, float* B,
               float beta, float* C);

const std::string errLogFile = "gemmValidationFailure.txt";

// NB: must use a single generator to avoid duplicates
std::default_random_engine generator(2);
std::uniform_real_distribution<float> distribution(0, 1);

int main(int argc, char** argv) {
    // command-line flags
    cxxopts::Options options("gemm.cu", "CUDA GEMM kernels");
    options.add_options()("size", "matrix size (N x N)",
                          cxxopts::value<uint16_t>()->default_value("128"))                      //
        ("reps", "repeat GEMM this many times", cxxopts::value<uint16_t>()->default_value("1"))  //
        ("algo", "GEMM algorithm to use, a number in [0,6], 0 is cuBLAS",
         cxxopts::value<uint16_t>()->default_value("0"))  //
        ("validate", "Validate output against cuBLAS",
         cxxopts::value<bool>()->default_value("true"))                                           //
        ("rngseed", "PRNG seed", cxxopts::value<uint>()->default_value("2"))                      //
        ("streams", "number of CUDA streams to use", cxxopts::value<uint>()->default_value("1"))  //
        ("h,help", "Print usage");

    auto clFlags = options.parse(argc, argv);
    if (clFlags.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    const uint16_t SIZE = clFlags["size"].as<uint16_t>();
    if (SIZE % 32 != 0) {
        std::cout << "--size must be a multiple of 32" << std::endl;
        exit(EXIT_FAILURE);
    }
    const uint16_t REPS = clFlags["reps"].as<uint16_t>();
    const Algo ALGO = static_cast<Algo>(clFlags["algo"].as<uint16_t>());
    if (ALGO >= numAlgos) {
        printf("Invalid algorithm: %d\n", ALGO);
        exit(EXIT_FAILURE);
    }
    const uint NUM_STREAMS = clFlags["streams"].as<uint>();

    const bool VALIDATE = clFlags["validate"].as<bool>();
    const uint SEED = clFlags["rngseed"].as<uint>();
    generator.seed(SEED);
    printf("Multiplying two %u x %u matrices with %u trials using %s algorithm\n", SIZE, SIZE, REPS,
           algo2str(ALGO));

    cudaCheck(cudaSetDevice(0));

    // Setup cublas
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    uint16_t m = SIZE, n = SIZE, k = SIZE;

    // GEMM computes C = α*AB+β*C

    // just do pure A*B (for simpler debugging)
    float alpha = 1.0, beta = 1.0, initC = 1.0;

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;      // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr;  // device matrices

    cudaMallocHost(&A, sizeof(float) * SIZE * SIZE);
    cudaMallocHost(&B, sizeof(float) * SIZE * SIZE);
    cudaMallocHost(&C, sizeof(float) * SIZE * SIZE);
    cudaMallocHost(&C_ref, sizeof(float) * SIZE * SIZE);

    randomize_matrix(A, SIZE * SIZE);
    randomize_matrix(B, SIZE * SIZE);
    randomize_matrix(C, SIZE * SIZE);

    const_init_matrix(C, SIZE * SIZE, initC);
    // print_matrix(A, SIZE, SIZE, std::cout);
    // print_matrix(B, SIZE, SIZE, std::cout);
    // print_matrix(C, SIZE, SIZE, std::cout);

    cudaCheck(cudaMalloc((void**)&dA, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void**)&dB, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void**)&dC, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void**)&dC_ref, sizeof(float) * SIZE * SIZE));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));

    printf("dimensions(m=n=k) %u, alpha: %f, beta: %f\n", m, alpha, beta);

    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (!VALIDATE) {
        printf("disabled validation\n");
    } else {
        // run cublas to get correct answer in dC_ref
        runCublas(handle, m, n, k, alpha, dA, dB, beta, dC_ref);

        // run user's algorithm, filling in dC
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC, NUM_STREAMS, A, B, C);

        cudaCheck(cudaDeviceSynchronize());

        // copy both results back to host
        cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        if (verify_matrix(C_ref, C, n, m)) {
            printf("Validated successfully!\n");
        } else {
            printf("Failed validation against NVIDIA cuBLAS.\n");
            std::cout << " Logging faulty output into " << errLogFile << "\n";
            std::ofstream fs;
            fs.open(errLogFile, std::ios::out | std::ios::trunc);
            fs << "α=" << alpha << " β=" << beta << std::endl;
            fs << "C matrix initialized to " << initC << std::endl << std::endl;
            fs << "A:" << std::endl;
            print_matrix(A, m, n, fs);
            fs << "B:" << std::endl;
            print_matrix(B, m, n, fs);
            fs << "C:" << std::endl;
            print_matrix(C, m, n, fs);
            fs << "Expected:" << std::endl;
            print_matrix(C_ref, m, n, fs);
            fs.close();
            exit(EXIT_FAILURE);
        }
    }

    // timing run(s)
    cudaEventRecord(beg);
    for (int j = 0; j < REPS; j++) {
        // We don't reset dC between runs to save time
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC, NUM_STREAMS, A, B, C);
        cudaCheck(cudaDeviceSynchronize());
    }

    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.;  // Convert to seconds

    double flops = (double)2 * m * n * k;
    printf("Average elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: (%u).\n",
           elapsed_time / REPS, (REPS * flops * 1e-9) / elapsed_time, m);

    // free CPU and GPU memory
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(C_ref);
    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dB));
    cudaCheck(cudaFree(dC));
    cudaCheck(cudaFree(dC_ref));
    cublasCheck(cublasDestroy(handle));

    return 0;
}

/** Function to check for errors in CUDA API calls */
void cudaErrorCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line, cudaGetErrorName(error),
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void cublasErrorCheck(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[CUDA ERROR] at file %s:%d:\n %s: %s\n", file, line, cublasGetStatusName(status),
               cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

/** Initialize the given matrix `mat` which has `N` contiguous values. Contents of `mat` are set to
 * random values. */
void randomize_matrix(float* mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = distribution(generator);
    }
}

void const_init_matrix(float* mat, int N, float F) {
    for (int i = 0; i < N; i++) {
        mat[i] = F;
    }
}

/** Print the given MxN matrix `mat` to the provided output stream. */
void print_matrix(const float* A, int M, int N, std::ostream& outs) {
    outs << "[";
    for (int i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0) {
            outs << std::fixed << std::setprecision(3) << A[i];
        } else {
            outs << std::fixed << std::setprecision(3) << A[i] << ", ";
        }
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N) outs << ";" << std::endl;
        }
    }
    outs << "]" << std::endl << std::endl;
}

bool verify_matrix(float* expected, float* actual, int M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float fexp = (expected[(i * N) + j]);
            float fact = (actual[(i * N) + j]);
            double diff = std::fabs(fexp - fact);
            if (diff > 0.002) {
                printf("Divergence! Should be %5.3f, is %5.3f (diff %5.3f) at [%d,%d]\n", fexp,
                       fact, diff, i, j);
                return false;
            }
        }
    }
    return true;
}

void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha, float* A, float* B,
               float beta, float* C) {
    // cuBLAS uses *column-major* order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // cublasStatus_t ok = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
    // CUDA_R_16F,
    //                                  N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N,
    //                                  /*CUBLAS_COMPUTE_16F*/ CUBLAS_COMPUTE_16F_PEDANTIC,
    //                                  CUBLAS_GEMM_DEFAULT);
    cublasStatus_t ok =
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    cublasCheck(ok);
}

__global__ void runBasic(int M, int N, int K, float alpha, float* A, float* B, float beta,
                         float* C) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        // C = α*(AxB)+β*C
        for (int i = 0; i < K; ++i) {
            // tmp += __A__[x][i] * __B__[i][y]
            tmp += A[(x * K) + i] * B[(i * N) + y];
        }
        // __C__[x][y]
        C[(x * N) + y] = (alpha * tmp) + (beta * C[x * N + y]);
    }
}

__global__ void runGmemCoalesced(int M, int N, int K, float alpha, float* A, float* B, float beta,
                                 float* C) {
    // HW1 TODO: copy runBasic() code here and update to avoid uncoalesced accesses to global
    // memory. Note, you are also free to change the grid dimensions in the kernel launch below.
}

const uint F = 32;

__global__ void runSharedMem(int M, int N, int K, float alpha, float* A, float* B, float beta,
                             float* C) {
    // HW2 TODO: Use shared memory to cache square FxF tiles of the A and B matrices in shared
    // memory (SA and SB, respectively, provided below). Each thread should compute the result for
    // one cell of the output matrix C.

    // Note, you will also need to change the grid dimensions in the kernel launch below to take
    // into account the value of F (which is a constant, defined above). You should experiment with
    // different values of F to see how it affects performance.

    __shared__ float SA[F][F];
    __shared__ float SB[F][F];
}

const uint G = 4;

__global__ void runSharedMemMultiOutput(int M, int N, int K, float alpha, float* A, float* B,
                                        float beta, float* C) {
    // HW3 TODO: Copy your runSharedMem() code here and update it so that each thread computes the
    // result for GxG cells of the output matrix C. Each thread should accumulate temporary results
    // in the local LC matrix, provided below, before writing them to C in global memory.

    // Note, you will also need to change the grid dimensions in the kernel launch below. You should
    // experiment with different values of F and G to see how they affect performance.

    __shared__ float SA[F][F];
    __shared__ float SB[F][F];

    float LC[G][G] = {0.0};

    // Each thread computes G×G output elements
    // Thread block size is (F/G, F/G), each block computes F×F outputs
    const int baseRow = blockIdx.y * F + threadIdx.y * G;
    const int baseCol = blockIdx.x * F + threadIdx.x * G;

    for (int t = 0; t < (K + F - 1) / F; ++t) {
        // Load F×F tile of A and B into shared memory
        // Each thread loads G×G elements to fully populate the shared memory tile
        for (int gy = 0; gy < G; ++gy) {
            for (int gx = 0; gx < G; ++gx) {
                int row = baseRow + gy;
                int col = baseCol + gx;
                int sRow = threadIdx.y * G + gy;
                int sCol = threadIdx.x * G + gx;

                // Load tile of A
                int aCol = t * F + sCol;
                if (row < M && aCol < K) {
                    SA[sRow][sCol] = A[row * K + aCol];
                } else {
                    SA[sRow][sCol] = 0.0;
                }

                // Load tile of B
                int bRow = t * F + sRow;
                if (bRow < K && col < N) {
                    SB[sRow][sCol] = B[bRow * N + col];
                } else {
                    SB[sRow][sCol] = 0.0;
                }
            }
        }

        __syncthreads();

        // Compute partial results
        for (int k = 0; k < F; ++k) {
            for (int gy = 0; gy < G; ++gy) {
                for (int gx = 0; gx < G; ++gx) {
                    LC[gy][gx] += SA[threadIdx.y * G + gy][k] * SB[k][threadIdx.x * G + gx];
                }
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    for (int gy = 0; gy < G; ++gy) {
        for (int gx = 0; gx < G; ++gx) {
            int row = baseRow + gy;
            int col = baseCol + gx;
            if (row < M && col < N) {
                const int idx = row * N + col;
                C[idx] = alpha * LC[gy][gx] + beta * C[idx];
            }
        }
    }
}

void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha, float* A, float* B,
             float beta, float* C, uint NUM_STREAMS, float* hA, float* hB, float* hC) {
    switch (algo) {
        case cublas:
            runCublas(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case basic: {
            dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
            dim3 blockDim(32, 32);
            runBasic<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case gmem_coalesced: {
            dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
            dim3 blockDim(32, 32);
            runGmemCoalesced<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case smem: {
            assert(0 == M % F);
            assert(0 == N % F);
            assert(0 == K % F);
            // TODO: update your grid here
            dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
            dim3 blockDim(32, 32);
            runSharedMem<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case smem_multioutput: {
            assert(0 == M % F);
            assert(0 == N % F);
            assert(0 == K % F);
            assert(0 == F % G);
            assert((F * F) / (G * G) >= F);
            // TODO: update your grid here
            dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
            dim3 blockDim(32, 32);
            runSharedMemMultiOutput<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }
        case smem_multioutput_1stream: {
            assert(0 == M % F);
            assert(0 == N % F);
            assert(0 == K % F);
            assert(0 == F % G);
            assert((F * F) / (G * G) >= F);

            // Synchronous memory copies
            cudaCheck(cudaMemcpy(A, hA, sizeof(float) * M * K, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(B, hB, sizeof(float) * K * N, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(C, hC, sizeof(float) * M * N, cudaMemcpyHostToDevice));

            dim3 gridDim(ROUND_UP_TO_NEAREST(M, F), ROUND_UP_TO_NEAREST(N, F));
            dim3 blockDim(F / G, F / G);
            runSharedMemMultiOutput<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            cudaCheck(cudaDeviceSynchronize());

            // TODO: HW4: use same grid & kernel launch as HW3
            cudaCheck(cudaMemcpy(hC, C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
            break;
        }
        case smem_multioutput_multistream: {
            assert(0 == M % F);
            assert(0 == N % F);
            assert(0 == K % F);
            assert(0 == F % G);
            assert((F * F) / (G * G) >= F);
            assert(0 == (N / F) % NUM_STREAMS);

            cudaStream_t streams[NUM_STREAMS];
            for (int i = 0; i < NUM_STREAMS; ++i) {
                cudaCheck(cudaStreamCreate(&streams[i]));
            }

            // TODO: HW4: use streams to overlap memory copies with kernel computation

            cudaCheck(cudaMemcpy(B, hB, sizeof(float) * K * N, cudaMemcpyHostToDevice));
            int rows_per_stream = M / NUM_STREAMS;

            for (int stream_id = 0; stream_id < NUM_STREAMS; ++stream_id) {
                int row_start = stream_id * rows_per_stream;
                int rows_in_slice = rows_per_stream;

                // Calculate slice sizes
                size_t A_slice_size = sizeof(float) * rows_in_slice * K;
                size_t C_slice_size = sizeof(float) * rows_in_slice * N;

                // Calculate device and host pointers for this slice
                float* dA_slice = A + row_start * K;
                float* dC_slice = C + row_start * N;
                float* hA_slice = hA + row_start * K;
                float* hC_slice = hC + row_start * N;

                // Async copy A slice to device
                cudaCheck(cudaMemcpyAsync(dA_slice, hA_slice, A_slice_size, cudaMemcpyHostToDevice,
                                          streams[stream_id]));

                // Async copy C slice to device
                cudaCheck(cudaMemcpyAsync(dC_slice, hC_slice, C_slice_size, cudaMemcpyHostToDevice,
                                          streams[stream_id]));

                // Launch kernel for this slice
                dim3 gridDim(ROUND_UP_TO_NEAREST(N, F), ROUND_UP_TO_NEAREST(rows_in_slice, F));
                dim3 blockDim(F / G, F / G);
                runSharedMemMultiOutput<<<gridDim, blockDim, 0, streams[stream_id]>>>(
                    rows_in_slice, N, K, alpha, dA_slice, B, beta, dC_slice);

                // Async copy result back to host
                cudaCheck(cudaMemcpyAsync(hC_slice, dC_slice, C_slice_size, cudaMemcpyDeviceToHost,
                                          streams[stream_id]));
            }

            for (int i = 0; i < NUM_STREAMS; ++i) {
                cudaCheck(cudaStreamSynchronize(streams[i]));
                cudaCheck(cudaStreamDestroy(streams[i]));
            }

            break;
        }
        default:
            printf("Invalid algorithm: %d\n", algo);
            exit(EXIT_FAILURE);
    }
    cudaCheck(cudaDeviceSynchronize());  // wait for kernel to finish
    cudaCheck(cudaGetLastError());       // check for errors from kernel run
}
