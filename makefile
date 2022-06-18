# nvcc -o qft -Xcompiler -fopenmp qft.cpp qft_gpu_v1.cu qft_gpu_v2.cu qft_gpu_v3.cu qft_gpu_v4.cu qft_gpu_v5.cu qft_gpu_v6.cu quantum_utils.cpp qft_host.cpp qft_gpu_launch.cu Stopwatch.cpp -lquantum

CC := nvcc				# Nvidia CUDA compiler
qftgpus := $(wildcard qft_gpu_v*.cu)	# All

qftmake: qft.cpp quantum_utils.cpp qft_host.cpp qft_gpu_launch.cu Stopwatch.cpp
	@${CC} -o qft -Xcompiler -fopenmp qft.cpp ${qftgpus} quantum_utils.cpp qft_host.cpp qft_gpu_launch.cu Stopwatch.cpp -lquantum
