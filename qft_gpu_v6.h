#ifndef QFT_GPU_V6_H_INCLUDED
#define QFT_GPU_V6_H_INCLUDED

#include <cuComplex.h>

void qft_gpu_v6_helper(int width, cuDoubleComplex *d_v, int threadsPerBlock);

#endif // QFT_HOST_V6_H_INCLUDED
