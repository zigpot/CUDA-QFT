## About:
This is the program used for the Undergraduate Disertation titled "Simulation of Quantum Computer Using GPGPU for The Case of Quantum Fourier Transform". Originally written in NVIDIA SDK ([here](https://www.eecg.utoronto.ca/~moshovos/CUDA08/arx/QFT_report.pdf)), ported to CUDA and modified.

## Prerequisites:
CUDA 10.0.0 or later

[libquantum 1.1.1](http://www.libquantum.de/files/libquantum-1.1.1.tar.gz)

## Compilation:
	nvcc -o program.out -Xcompiler -fopenmp files here -lquantum
For instance:	
	nvcc -o qft -Xcompiler -fopenmp qft_gpu_launch.cu qft_gpu_v3.cu qft_gpu_v6.cu qft_gpu_v2.cu qft_gpu_v5.cu quantum_utils.cpp qft.cpp qft_gpu_v1.cu qft_gpu_v4.cu qft_host.cpp Stopwatch.cpp -lquantum
	Unfortunately, this is the way to do before a makefile is ready.
## Usage:
	./qft B [V]
B = Number of qubits

V = version of the GPU algorithm

## Contact Me:
for more info please contact: fariz.adnan.w[at]mail.ugm.ac.id
