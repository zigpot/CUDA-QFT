#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
using std::max;
using std::min;

#include <cuComplex.h>
#include "qft_host.h"


// basic host implementation


// process bits i and i^(1ul << target)
static void phase_shift_host(int control, int target, unsigned long i, cuDoubleComplex *v){
	float phi = 1.0f * float(M_PI) / float(1ul << (control-target));
	cuDoubleComplex phase = {cosf(phi), sinf(phi)};

	unsigned long tgt_bit = (1ul << target);
	unsigned long ctl_bit = (1ul << control);

	if ((i & tgt_bit) == 0) {
		fprintf(stderr, "Logic error: phase_shift_host() should not be called with '(i & tgt_bit) == 0'.\n");
		exit(1);
	}

	if ((i & ctl_bit) != 0) {
		v[i] = cuCmul(v[i], phase);
		// v[i^tgt_bit] stays unchanged
	}
}


// memproses bit i dan i^(1ul << target)
static void hadamard_host(unsigned long target, unsigned long i, cuDoubleComplex *v){
	unsigned long tgt_bit = (1ul << target);

	if ((i & tgt_bit) == 0) {
		fprintf(stderr, "Logic error: phase_shift_host() tidak seharusnya dipanggil dengan '(i & tgt_bit) == 0'.\n");
		exit(1);
	}

	unsigned long i_other = i^tgt_bit;
	cuDoubleComplex ai, aother;	// coefficient i and (i^tgt_bit)
	cuDoubleComplex v_i = v[i];
	cuDoubleComplex v_iother = v[i_other];

	cuDoubleComplex cuM_SQRT1_2 = make_cuDoubleComplex(M_SQRT1_2, 0);
	ai = cuCmul(cuM_SQRT1_2, cuCsub(v_iother, v_i));
	aother = cuCmul(cuM_SQRT1_2, cuCadd(v_iother, v_i));

	v[i] = ai;
	v[i_other] = aother;
}

static void qft_host_kernel(int tgt, unsigned long i, int width, cuDoubleComplex *v){
	unsigned long tgt_bit = (1ul << tgt);

	if ((i & tgt_bit) == 0) {
        fprintf(stderr, "Logic error: qft_host_kernel() should not be called with '(i & tgt_bit) == 0'.\n");
		exit(1);
	}

    // Note: Phase shifts (with the same target) commute.
    //for (int ctl=width-1; ctl>tgt; ctl--) {
	for (int ctl = tgt + 1; ctl < width; ctl++) {
		phase_shift_host(ctl, tgt, i, v);
	}
	hadamard_host(tgt, i, v);
}


// For each qubit, process each state (pair of coefficients) separately.
// This is structured so that it could be implemented in a GPU kernel.
// From here, the next step would be to perform coefficient grouping to achieve
// memory coalescing.
void qft_host(int width, cuDoubleComplex *v){
	unsigned long N = (1ul << width);

    // For each qubit...
	for (int tgt = width - 1; tgt >= 0; tgt--) {
        // For each state...
		for (unsigned long i = 0; i < N; i++) {
			unsigned long tgt_bit = (1ul << tgt);
			if ((i & tgt_bit) != 0) {
				qft_host_kernel(tgt, i, width, v);
			}
		}
	}
}
