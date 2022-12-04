## About:
This is the accompanying program used in the paper "Optimizing Quantum Fourier Transform Simulation on GPU". The program was originally written in NVIDIA SDK ([here](https://www.eecg.utoronto.ca/~moshovos/CUDA08/arx/QFT_report.pdf)), ported to CUDA and modified.

## Prerequisites:
CUDA 10.0.0 or later

[libquantum 1.1.1](http://www.libquantum.de/files/libquantum-1.1.1.tar.gz) (don't forget to link the newly installed library, e.g. sudo ldconfig)

## Compilation:
Simply run makefile:

	make
	
## Usage:
	./qft B [V]
B = Number of qubits

V = version of the GPU algorithm

## Contact Me:
for more info please contact: farizmade[at]yahoo.com
