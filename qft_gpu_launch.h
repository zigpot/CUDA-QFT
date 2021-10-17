#ifndef QFT_GPU_LAUNCH_H_INCLUDED
#define QFT_GPU_LAUNCH_H_INCLUDED

#include <cuComplex.h>

extern "C" {
    #include <quantum.h>
}

/*
 * Code in this header file must not use and CUDA structures.
 * However, code in the associated .cpp file *may* make CUDA calls.
 *
 * In particular, the functions here should take "normal" C data structures
 * and pass them off to CUDA kernels.
 */

typedef enum {
    QFT_v0_HOST = 0,
    QFT_v1_PLAIN,
    QFT_v2_PHASE,
    QFT_v3_TRIG,
    QFT_v4_SHARED,
    QFT_v5_0TO8,
    QFT_v6_GROUPED,
    QFT_BEST = QFT_v5_0TO8,
} QFT_versions;


double qft_gpu(int width, cuDoubleComplex *h_v, int threadsPerBlock, QFT_versions vers);
//double qft_gpu_qureg(quantum_reg& qr, int threadsPerBlock, QFT_versions vers);


#endif // QFT_HOST_SERIAL_H_INCLUDED
