#ifndef QFT_GPU_V2_H_INCLUDED
#define QFT_GPU_V2_H_INCLUDED

#include <cuComplex.h>

void qft_gpu_v2_helper(int width, cuDoubleComplex *d_v, int threadsPerBlock);

#endif // QFT_HOST_V2_H_INCLUDED
