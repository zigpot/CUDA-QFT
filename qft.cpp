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
		fprintf(stderr, "Error ketika mengalokasikan vektor (ukuran %llu).\n", N);
		exit(1);
	}

	// Inisiasi vektor uji, tidak ternormalisasi.
	unsigned long long i;
	for (i = 0; i < N; i++) {
		(*v)[i].x = 1.0f/float(i+1) + 1e-5f;
		(*v)[i].y  = 1.0f/float(i+1) + 1e-5f;
	}
}


int qft_test(int argc, char *argv[]){
	// Dapatkan jumlah qubit dari pengguna, default = 8
	int width = 8;
	if (argc >= 1) {
		width = atoi(argv[0]);
	}
	// Dapatkan versi program GPU dari pengguna, default = terbaik
	QFT_versions version = QFT_BEST;
	if (argc >= 2) {
		version = (QFT_versions)atoi(argv[1]);
	}
	if (version == QFT_v0_HOST) {
		printf("Menggunakan GPU-QFT versi: 0 (Implementasi satu core CPU)\n");
	} else {
		printf("Menggunakan GPU-QFT versi: %d\n", (int)version);
	}

	double vecsize = double(sizeof(cuDoubleComplex))*double(1ull<<width)/1024.0/1024.0;
	printf("Lebar: %d qubits (%llu states, %f MB).\n", width, 1ull<<width, vecsize);

	Stopwatch s(false); // menghitung waktu CPU time


	// Membuat vektor uji baru dan menyalinnya ke dalam vektor keluaran.
	// Vektor keluaran disalin di tempat
	cuDoubleComplex *vin = NULL, *vout = NULL, *vtmp = NULL;
	make_test_vector(width, &vin);
	qutil_new_qvec(width, &vout);
	quantum_reg qr = qutil_collapse_qr(width, vin);
	qutil_copy_qvec(&vout, vin, width);

	// Hitung waktu implementasi GPU.
	double time_gpu = qft_gpu(width, vout, THREADS_PER_BLOCK, version);
	printf("GPU time:   %f ms.\n", time_gpu*1000.0);
	fflush(stdout);

	// Hitung waktu implementasi libquantum.
	s.restart();
	quantum_qft(width, &qr);
	double time_libq = s.getElapsed();
	printf("Libquantum: %f ms.\n", time_libq*1000.0);
	fflush(stdout);

	// Bandingkan waktu
	printf("Speedup:	%f x\n", time_libq/time_gpu);

	// Bandingkan keluaran
	printf("Mempersiapkan vektor-vektor untuk perbandingan...\n");
	// Bersihkan memori dari qureg dan array yang tidak diperlukan lagi
	qutil_destroy_qvec(&vin);
	// Ekspansi vektor libquantum untuk perbandingan.
	qutil_expand_qr(qr, &vtmp);
	quantum_delete_qureg(&qr);
	// Collapse dan re-expand vektor GPU.
	// libquantum secara otomatis menghapus keadaan "kecil", bila kita tidak melakukan hal serupa,
	// maka hasil perbandingan akan berbeda
	quantum_reg qr_tmp = qutil_collapse_qr(width, vout);
	qutil_expand_qr(qr_tmp, &vout);
	printf("l2norm = %f\n", qutil_l2norm_qvecs(width, vout, vtmp));

	// Bersihkan vektor-vektor yang tersisa
	qutil_destroy_qvec(&vout);
	qutil_destroy_qvec(&vtmp);

	return 0;
}


static void printUsage(){
	printf("Penggunaan: qft B [V]\n");
	printf("	menguji algoritme QFT dengan qubit sejumlah B serta versi V daripada algoritme GPU.\n");
	printf("	V haruslah dari 0 hingga 6. Secara default, V adalah %d.\n", (int)QFT_BEST);
}

int main(int argc, char *argv[]){
	if (argc == 1) {
		printUsage();
		return 1;
	} else {
		return qft_test(argc-1, argv+1);
}
}
