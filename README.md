## About:
This is the program used for the Undergraduate Disertation titled "Simulation of Quantum Computer Using GPGPU for The Case of Quantum Fourier Transform". Originally written in NVIDIA SDK ([here](https://www.eecg.utoronto.ca/~moshovos/CUDA08/arx/QFT_report.pdf)), ported to CUDA and modified.

## Prerequisites:
CUDA 10.0.0 or later

[libquantum 1.1.1](http://www.libquantum.de/files/libquantum-1.1.1.tar.gz)

## Compilation:
	nvcc -o program.out -Xcompiler -fopenmp files here -lquantum
## Usage:
	./qft B [V]
B = Number of qubits

V = version of the GPU algorithm

## Contact Me:
for more info please contact: fariz.adnan.w[at]mail.ugm.ac.id
