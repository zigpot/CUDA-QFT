#include <cuComplex.h>
#include "cutil.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // M_PI
#include "qft_gpu_v1.h"

#define BLOCKSIZE 65535


// Ini versi asli, di-port dari kode CPU ke GPU tanpa optimisasi.

// memproses bit i dan i^(1ul << target)
// Catatan: HANYA dipanggil dengan ((i & tgt_bit) == 1)
__device__ static void phase_shift_gpu(int ctl, int tgt, unsigned long i, int width, cuDoubleComplex *v)
{
	unsigned long tgt_bit = (1ul << tgt);
	if ((i & tgt_bit) == 0) {
		return;
	}

	unsigned long ctl_bit = (1ul << ctl);

	float phi = 1.0f * float(M_PI) / float(1ul << (ctl - tgt));
	cuDoubleComplex phase = { cosf(phi), sinf(phi) };

	if ((i & ctl_bit) != 0) {
		v[i] = cuCmul(v[i], phase);
		// v[i^tgt_bit] stays unchanged
	}
}


// memproses bit i dan i^(1ul << target)
// Catatan: HANYA dipanggil dengan ((i & tgt_bit) == 1)
__device__ static void hadamard_gpu(unsigned long tgt, unsigned long i, int width, cuDoubleComplex *v)
{
	unsigned long tgt_bit = (1ul << tgt);
	if ((i & tgt_bit) == 0) {
		return;
	}

	unsigned long i_other = i^tgt_bit;
	cuDoubleComplex ai, aother;	// koefisien i dan (i^tgt_bit)
	cuDoubleComplex v_i = v[i];
	cuDoubleComplex v_iother = v[i_other];

	cuDoubleComplex cuM_SQRT1_2 = make_cuDoubleComplex(M_SQRT1_2, 0);
	ai = cuCmul(cuM_SQRT1_2, cuCsub(v_iother, v_i));
	aother = cuCmul(cuM_SQRT1_2, cuCadd(v_iother, v_i));

	v[i] = ai;
	v[i_other] = aother;
}


// Mengaplikasikan geser fase dan transformasi Hadamard untuk qubit 'tgt' dan state 'i'.
// Catatan: HANYA dipanggil dengan ((i & tgt_bit) == 1)
__device__ static void qft_gpu_v1_single_state(int tgt, unsigned long i, int width, cuDoubleComplex *v)
{
	unsigned long tgt_bit = (1ul << tgt);
	if ((i & tgt_bit) == 0) {
		return;
	}

	// Catatan: Geser fase (dengan target sama) commute.
	for (int ctl = tgt + 1; ctl < width; ctl++) {
		phase_shift_gpu(ctl, tgt, i, width, v);
	}
	hadamard_gpu(tgt, i, width, v);
}


// Kernel ini melakukan QFT single stage.
__global__ static void K_qft_gpu_v1_stage(int width, cuDoubleComplex *v, int tgt)
{
	unsigned long N = (1ul << width);

	// Membagi threads ke tiap state.
	unsigned long long bidx = blockIdx.y*gridDim.x + blockIdx.x;
	unsigned long long i = bidx*blockDim.x + threadIdx.x;

	if (i >= N) {
		return;
	}

	unsigned long tgt_bit = (1ul << tgt);
	if ((i & tgt_bit) != 0) {
		qft_gpu_v1_single_state(tgt, i, width, v);
	}
}


// Implementasi gerbang QFT dengan menggunakan GPU.
void qft_gpu_v1_helper(int width, cuDoubleComplex *d_v, int threadsPerBlock)
{
	unsigned long N = (1ul << width);

	unsigned long long nblocks = (N+threadsPerBlock-1)/threadsPerBlock;
	unsigned long long xblocks, yblocks;
	yblocks = (nblocks+BLOCKSIZE-1)/BLOCKSIZE;
	xblocks = BLOCKSIZE;
	if (nblocks < xblocks) {
		xblocks = nblocks;
	}
	dim3 blocks(xblocks, yblocks);

	// Untuk tiap qubit...
	for (int tgt = width - 1; tgt >= 0; tgt--) {
		K_qft_gpu_v1_stage<<<blocks, threadsPerBlock>>>(width, d_v, tgt);
		CUT_CHECK_ERROR("K_qft_gpu_v1_stage gagal.");
	}
}
