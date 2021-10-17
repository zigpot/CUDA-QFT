TENTANG:
Ini adalah program yang digunakan untuk tugas akhir bertajuk: "Simulasi Komputer Kuantum Menggunakan GPGPU untuk Kasus Quantum Fourier Transform"

KOMPILASI:
nvcc -Xcompiler -fopenmp -o run qft.cpp qft_gpu_launch.cu qft_gpu_v1.cu qft_gpu_v2.cu qft_gpu_v3.cu qft_gpu_v4.cu qft_gpu_v5.cu qft_gpu_v6.cu qft_host.cpp quantum_utils.cpp Stopwatch.cpp -lquantum

PENGGUNAAN:
	./qft B [V]
	B = Jumlah qubit
	V = versi algoritme GPU (1-6)

untuk info lebih lanjut silakan hubungi: fariz.adnan.w[at]mail.ugm.ac.id
