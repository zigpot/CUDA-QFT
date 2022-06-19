#include <cuComplex.h>
#include "cutil.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // M_PI
#include "qft_gpu_v3.h"

#define BLOCKSIZE 65535


// Instead of calculating both sinf and cosf, calculates sin from +/- sqrt(1-cos^2).


// process bits i and i^(1ul << tgt)
// Note: This should ONLY be called with ((i & tgt_bit) == 1)
__device__ static void hadamard_gpu(unsigned long tgt, unsigned long i, int width, cuDoubleComplex *v){
	unsigned long tgt_bit = (1ul << tgt);
	if ((i & tgt_bit) == 0) {
        // This function should not have been called in this case.
        return;
    }

    unsigned long i_other = i^tgt_bit;
    cuDoubleComplex ai, aother;    // coefficients i and (i^tgt_bit)
	cuDoubleComplex v_i = v[i];
	cuDoubleComplex v_iother = v[i_other];

	cuDoubleComplex cuM_SQRT1_2 = make_cuDoubleComplex(M_SQRT1_2, 0);
	ai = cuCmul(cuM_SQRT1_2, cuCsub(v_iother, v_i));
	aother = cuCmul(cuM_SQRT1_2, cuCadd(v_iother, v_i));

	v[i] = ai;
	v[i_other] = aother;
}



// This applies the phase shifts and Hadamard transform for qubit 'tgt' and state 'i'.
// Note: This should ONLY be called with ((i & tgt_bit) == 1)
__device__ static void qft_gpu_v3_single_state(int tgt, unsigned long i, int width, cuDoubleComplex *v){
	unsigned long phase_coef = 1ul;
	unsigned long tgt_bit = (1ul << tgt);
	unsigned long normal = (1ul << (width - tgt - 1));

	if ((i & tgt_bit) == 0) {
		return;
	}

	unsigned long ctl_bit = (1ul << tgt);
	for (int ctl = tgt + 1; ctl < width; ctl++) {
		ctl_bit = (ctl_bit << 1);
		phase_coef = (phase_coef <<1);
		if ((i & ctl_bit) != 0) {
			phase_coef = phase_coef ^ 1ul;
		}
	}
	phase_coef = phase_coef ^ (normal);
	float phi = float(phase_coef) * float(M_PI) / float(normal);

	float c = cosf(phi);
	float s = sqrtf(1.0f - c*c);
	if (phi > float(M_PI)) {
		s *= -1.0f;
	}
	cuDoubleComplex phase = {c, s};

	v[i] = cuCmul(v[i], phase);

	hadamard_gpu(tgt, i, width, v);
}


// This kernel performs a single stage of the QFT.
__global__ static void K_qft_gpu_v3_stage(int width, cuDoubleComplex *v, int tgt){
	unsigned long N = (1ul << width);

    // Split threads over states.
	unsigned long long bidx = blockIdx.y*gridDim.x + blockIdx.x;
	unsigned long long i = bidx*blockDim.x + threadIdx.x;

	if (i >= N) {
		return;
	}

	unsigned long tgt_bit = (1ul << tgt);
	if ((i & tgt_bit) != 0) {
		qft_gpu_v3_single_state(tgt, i, width, v);
	}
}


// Implement the QFT gate by gate using the GPU.
void qft_gpu_v3_helper(int width, cuDoubleComplex *d_v, int threadsPerBlock){
	unsigned long N = (1ul << width);

	unsigned long long nblocks = (N + threadsPerBlock - 1)/threadsPerBlock;
	unsigned long long xblocks, yblocks;
	yblocks = (nblocks + BLOCKSIZE - 1)/BLOCKSIZE;
	xblocks = BLOCKSIZE;
	if (nblocks < xblocks) {
		xblocks = nblocks;
	}
	dim3 blocks(xblocks, yblocks);

	// Untuk tiap qubit...
	for (int tgt=width-1; tgt>=0; tgt--) {
		K_qft_gpu_v3_stage<<<blocks, threadsPerBlock>>>(width, d_v, tgt);
		CUT_CHECK_ERROR("K_qft_gpu_v3_stage failed.");
	}
}
