
#include <cuComplex.h>


#include <stdio.h>
#include <stdlib.h>


#include "qft_gpu_launch.h"
#include "qft_gpu_v1.h"
#include "qft_gpu_v2.h"
#include "qft_gpu_v3.h"
#include "qft_gpu_v4.h"
#include "qft_gpu_v5.h"
#include "qft_gpu_v6.h"
#include "qft_host.h"

#include "Stopwatch.h"
#include "quantum_utils.h"
#include "cutil.h"


/*
 * Copies 'h_v' into device memory, runs qft_gpu_bygate_v1_helper() on it,
 * copies the result back, and returns the CUDA time used in seconds.
 */
double qft_gpu(int width, cuDoubleComplex *h_v, int threadsPerBlock, QFT_versions vers) {
    int deviceCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "There is no device supporting CUDA\n");
        exit(1);
    }
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));

    cudaSetDevice(0);

    threadsPerBlock = min(threadsPerBlock, deviceProp.maxThreadsPerBlock);

    unsigned long long N = (1ull << width);

    cuDoubleComplex *d_v = NULL;

    if (vers != QFT_v0_HOST) {
        if (cudaMalloc((void **) &d_v, sizeof(cuDoubleComplex) * N) != cudaSuccess) {
            fprintf(stderr, "Error allocating %llu states in device memory.\n", N);
            exit(1);
        }
    }

    //unsigned int htimer;//obsolete
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cutCreateTimer(&htimer);//obsolete

    if (vers != QFT_v0_HOST) {
        cudaMemcpy((void *) d_v, (void *) h_v, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }


    // Run the QFT.
    printf("Running GPU-QFT v%d with %d qubits\n", (int) vers, width);
    fflush(stdout);

    Stopwatch s(false); // CPU time
    //cutResetTimer(htimer); //obsolete
    //cutStartTimer(htimer); //obsolete
    cudaEventRecord(start, 0);
    switch (vers) {
        case QFT_v0_HOST:
            //TODO: count CPU time
            s.restart();
            qft_host(width, h_v);
            s.stop();
            break;

        case QFT_v1_PLAIN:
            qft_gpu_v1_helper(width, d_v, threadsPerBlock);
            break;

        case QFT_v2_PHASE:
            qft_gpu_v2_helper(width, d_v, threadsPerBlock);
            break;

        case QFT_v3_TRIG:
            qft_gpu_v3_helper(width, d_v, threadsPerBlock);
            break;

        case QFT_v4_SHARED:
            qft_gpu_v4_helper(width, d_v, threadsPerBlock);
            break;

        case QFT_v5_0TO8:
            qft_gpu_v5_helper(width, d_v, threadsPerBlock);
            break;

        case QFT_v6_GROUPED:
            qft_gpu_v6_helper(width, d_v, threadsPerBlock);
            break;

        default:
            fprintf(stderr, "An unknown GPU QFT implementation version, %d, was requested.\n", (int) vers);
            exit(1);
    }

    //cudaDeviceSynchronize(); //obsolete
    //cutStopTimer(htimer); //obsolete
    cudaEventRecord(stop, 0);

    if (vers != QFT_v0_HOST) {
        cudaMemcpy((void *) h_v, (void *) d_v, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        cudaFree(d_v);
    }

    float runtime_ms;
    if (vers == QFT_v0_HOST) {
        runtime_ms = s.getElapsed() * 1000.0;
    } else {
        //runtime_ms = cutGetTimerValue(htimer); //obsolete
        cudaEventElapsedTime(&runtime_ms, start, stop);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return double(runtime_ms / 1000.0);
}


double qft_gpu_qureg(quantum_reg &qr, int threadsPerBlock, QFT_versions vers) {
    // Ekspand qr.
    cuDoubleComplex *v = NULL;
    qutil_expand_qr(qr, &v);

    double runtime_ms = qft_gpu(qr.width, v, threadsPerBlock, vers);

    // Delete the existing qr and replace it by expanding the calcualted vector.
    quantum_delete_qureg(&qr);
    qr = qutil_collapse_qr(qr.width, v);

    // Clean up.
    qutil_destroy_qvec(&v);
    return runtime_ms;
}
