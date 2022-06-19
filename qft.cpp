#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuComplex.h>
#include "qft_gpu_launch.h"
#include "quantum_utils.h"
#include "Stopwatch.h"

#define THREADS_PER_BLOCK 512

void make_test_vector(int width, cuDoubleComplex **v){
	if (*v) {
		free(*v);
	}
	unsigned long long N = (1ull << width);
	*v = (cuDoubleComplex*)calloc(N, sizeof(cuDoubleComplex));
	if (!*v) {
		fprintf(stderr, "Error allocating memory for vector (size %llu).\n", N);
		exit(1);
	}

    // Initialize a test vector.
    // This isn't normalized. In a normalized test vector, some elements
    // would be very small. libquantum discards low-probability entries during its
    // calculations. This creates "false" error between libquantum's output and ours.
	unsigned long long i;
	for (i = 0; i < N; i++) {
		(*v)[i].x = 1.0f/float(i+1) + 1e-5f;
		(*v)[i].y  = 1.0f/float(i+1) + 1e-5f;
	}
}


int qft_test(int argc, char *argv[]){
    // Get the number of qubits from the user, or use 8 as the default.
	int width = 8;
	if (argc >= 1) {
		width = atoi(argv[0]);
	}
    // Get the GPU version from the user or use the latest as the default.
	QFT_versions version = QFT_BEST;
	if (argc >= 2) {
		version = (QFT_versions)atoi(argv[1]);
	}
	if (version == QFT_v0_HOST) {
        printf("Using GPU-QFT version: 0 (HOST reference implementation)\n");
    } else {
        printf("Using GPU-QFT version: %d\n", (int)version);
	}

	double vecsize = double(sizeof(cuDoubleComplex))*double(1ull<<width)/1024.0/1024.0;
    printf("Width: %d qubits (%llu states, %f MB).\n", width, 1ull<<width, vecsize);

	Stopwatch s(false); // measure CPU time


    // Make a new test vector and copy it into the output vector.
    // (The output vector will be modified in-place.)
	cuDoubleComplex *vin = NULL, *vout = NULL, *vtmp = NULL;
	make_test_vector(width, &vin);
	qutil_new_qvec(width, &vout);
	quantum_reg qr = qutil_collapse_qr(width, vin);
	qutil_copy_qvec(&vout, vin, width);

    // Time the GPU implementation.
	double time_gpu = qft_gpu(width, vout, THREADS_PER_BLOCK, version);
	printf("GPU time:   %f ms.\n", time_gpu*1000.0);
	fflush(stdout);

    // Time libquantum's implementation.
	s.restart();
	quantum_qft(width, &qr);
	double time_libq = s.getElapsed();
	printf("Libquantum: %f ms.\n", time_libq*1000.0);
	fflush(stdout);

	// Bandingkan waktu
	printf("Speedup:	%f x\n", time_libq/time_gpu);

    // Compare the output vectors.
    printf("Preparing vectors for comparison...\n");
    // To save memory, we clean up arrays and quregs we don't need.
	qutil_destroy_qvec(&vin);
    // Expand libquantum's vector for comparison.
	qutil_expand_qr(qr, &vtmp);
	quantum_delete_qureg(&qr);
    // Collapse and re-expand the GPU vector.
    // libquantum deletes "small" states automatically. If we don't do the same,
    // then the comparison will be skewed.
	quantum_reg qr_tmp = qutil_collapse_qr(width, vout);
	qutil_expand_qr(qr_tmp, &vout);
	printf("l2norm = %f\n", qutil_l2norm_qvecs(width, vout, vtmp));

    // Clean up the remaining vectors.
	qutil_destroy_qvec(&vout);
	qutil_destroy_qvec(&vtmp);

	return 0;
}


static void printUsage(){
    printf("Usage: qft B [V]\n");
    printf("    tests the QFT algorithm using B qubits and version V of the GPU algorithm.\n");
    printf("    V should be from 0 to 6. The default value is %d.\n", (int)QFT_BEST);
}

int main(int argc, char *argv[]){
	if (argc == 1) {
		printUsage();
		return 1;
	} else {
		return qft_test(argc-1, argv+1);
}
}
