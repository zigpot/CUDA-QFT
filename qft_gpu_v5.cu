#include <cuComplex.h>
#include "cutil.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // M_PI
#include "qft_gpu_v5.h"

#define BLOCKS 32768


// Uses shared memory for the lower targets (0to8)


__device__ static void qft_gpu_v5_single_lower_state(int tgt, unsigned long i, int width, unsigned long block_base, cuDoubleComplex *sh_v){
	unsigned long tgt_bit = (1ul << tgt);
	unsigned long i_other = i^tgt_bit;

    // First half warp gets coefficients from shared memory.
	cuDoubleComplex v_i, v_iother;
	if ((i % 16) >= 8) {
		v_i = sh_v[i - block_base];
		v_iother = sh_v[i_other - block_base];
	}

    // Calculate the phase.
	unsigned long phase_coef = 1ul;
	unsigned long normal = (1ul << (width - tgt - 1));

	unsigned long ctl_bit = (1ul << tgt);
	for (int ctl = tgt + 1; ctl < width; ctl++) {
		ctl_bit = (ctl_bit << 1);
		phase_coef = (phase_coef <<1);
		if ((i & ctl_bit) != 0) {
			phase_coef = phase_coef ^ 1ul;
		}
	}
	phase_coef = phase_coef ^ (normal);

    // Second half warp gets coefficients from shared memory.
	if ((i % 16) < 8) {
		v_iother = sh_v[i_other-block_base];
		v_i = sh_v[i-block_base];
	}

    // Finish calculating the gate.
	float phi = float(phase_coef) * float(M_PI) / float(normal);
	float c = cosf(phi);
	float s = sqrtf(1.0f-c*c);
	if (phi>float(M_PI)) {
		s *= -1.0f;
	}
	cuDoubleComplex phase = {c, s};
	v_i = cuCmul(v_i, phase);

	cuDoubleComplex ai, aother;	// koefisien i dan (i^tgt_bit)

	cuDoubleComplex cuM_SQRT1_2 = make_cuDoubleComplex(M_SQRT1_2, 0);
	ai = cuCmul(cuM_SQRT1_2, cuCsub(v_iother, v_i));
	aother = cuCmul(cuM_SQRT1_2, cuCadd(v_iother, v_i));


	if ((i % 16) >= 8) {
		sh_v[i - block_base] = ai;
		sh_v[i_other - block_base] = aother;
	} else {
		sh_v[i_other - block_base] = aother;
		sh_v[i - block_base] = ai;
	}
}


// This applies the phase shifts and Hadamard transform for qubit 'tgt' and state 'i'.
// Note: This should ONLY be called with ((i & tgt_bit) == 1)
__device__ static void qft_gpu_v5_single_state(int tgt, unsigned long i, int width, cuDoubleComplex *v){
	unsigned long phase_coef = 1ul;
	unsigned long tgt_bit = (1ul << tgt);
	unsigned long i_other = i^tgt_bit;
	unsigned long normal = (1ul << (width - tgt - 1));

/*
	if ((i & tgt_bit) == 0) {
		return;
	}
*/

    // Note: Phase shifts (with the same target) commute.
	unsigned long ctl_bit = (1ul << tgt);
	for (int ctl=tgt+1; ctl<width; ctl++) {
		ctl_bit = (ctl_bit << 1);
		phase_coef = (phase_coef <<1);
		if ((i & ctl_bit) != 0) {
			phase_coef = phase_coef ^ 1ul;
		}
	}
	phase_coef = phase_coef ^ (normal);
	float phi = float(phase_coef) * float(M_PI) / float(normal);
	float c = cosf(phi);
	float s = sqrtf(1.0f - c * c);
	if (phi>float(M_PI)) {
		s *= -1.0f;
	}
	cuDoubleComplex phase = {c, s};
	cuDoubleComplex v_i = v[i];
	cuDoubleComplex v_iother = v[i_other];
	v_i = cuCmul(v_i, phase);

	cuDoubleComplex ai, aother;    // coefficients i and (i^tgt_bit)

	cuDoubleComplex cuM_SQRT1_2 = make_cuDoubleComplex(M_SQRT1_2, 0);
	ai = cuCmul(cuM_SQRT1_2, cuCsub(v_iother, v_i));
	aother = cuCmul(cuM_SQRT1_2, cuCadd(v_iother, v_i));

	v[i] = ai;
	v[i_other] = aother;
}


// This kernel performs a single stage of the QFT.
__global__ static void K_qft_gpu_v5_stage(int width, cuDoubleComplex *v, int tgt){
	unsigned long N = (1ul << width);

    // Split threads over states.
	unsigned long long bidx = blockIdx.y*gridDim.x + blockIdx.x;
	unsigned long long i = bidx*blockDim.x + threadIdx.x;

	if (i >= N) {
		return;
	}

	unsigned long tgt_bit = (1ul << tgt);
	if ((i & tgt_bit) != 0) {
		qft_gpu_v5_single_state(tgt, i, width, v);
	}
}




// This kernel performs a single stage of the QFT.
__global__ static void K_qft_gpu_v5_stage_0to8(int width, cuDoubleComplex *v){
	unsigned long N = (1ul << width);

    // Split threads over states.
	unsigned long long bidx = blockIdx.y*gridDim.x + blockIdx.x;
	unsigned long long i = bidx*blockDim.x + threadIdx.x;

	// copy in
	unsigned long block_base = bidx*blockDim.x;
	__shared__ cuDoubleComplex sh_v[512]; // Note: hardcoded shared memory size
	if (i < N) {
		sh_v[threadIdx.x] = v[i];
	}
	__syncthreads();


    // calculate
	for (int tgt = min(width-1,8); tgt >= 0; tgt--) {
		unsigned long tgt_bit = (1ul << tgt);
		if ( (i < N) && ((i & tgt_bit) != 0) ) {
			qft_gpu_v5_single_lower_state(tgt, i, width, block_base, sh_v);
		}
		__syncthreads();
	}

	// copy out
	if (i < N) {
		v[i] = sh_v[threadIdx.x];
	}
}


// Implement the QFT gate by gate using the GPU.
void qft_gpu_v5_helper(int width, cuDoubleComplex *d_v, int threadsPerBlock){
	unsigned long N = (1ul << width);

	unsigned long long nblocks = (N+threadsPerBlock-1)/threadsPerBlock;
	unsigned long long xblocks, yblocks;

	yblocks = (nblocks+BLOCKS-1)/BLOCKS;
	xblocks = BLOCKS;
	if (nblocks < xblocks) {
		xblocks = nblocks;
	}
	dim3 blocks(xblocks, yblocks);

	// For each qubit...
	int tgt;
	for (tgt=width-1; tgt>=9; tgt--) {
		K_qft_gpu_v5_stage<<<blocks, threadsPerBlock>>>(width, d_v, tgt);
		CUT_CHECK_ERROR("K_qft_gpu_v5_stage failed.");
	}
	K_qft_gpu_v5_stage_0to8<<<blocks, threadsPerBlock>>>(width, d_v);
	CUT_CHECK_ERROR("K_qft_gpu_v5_stage_0to8 failed.");
}
