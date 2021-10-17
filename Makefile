################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= qft

# CUDA source files (compiled with cudacc)
CUFILES		:= qft_gpu_v1.cu qft_gpu_v2.cu qft_gpu_v3.cu \
		   qft_gpu_v4.cu qft_gpu_v5.cu qft_gpu_v6.cu \
		   qft_gpu_launch.cu

# CUDA dependency files
# Don't use this. These files don't seem to get compiled.
CU_DEPS		:= 

# C/C++ source files (compiled with gcc / c++)
# These files cannot contain any CUDA code.
CCFILES		:= qft_host.cpp qft.cpp quantum_utils.cpp Stopwatch.cpp


################################################################################
# Rules and targets

# Editing INCLUDES must happen *before* including common.mk.
INCLUDES += -I${HOME}/include

include ../../common/common.mk

# Editing LIBS must happen *after* including common.mk.
LIB += -L${HOME}/lib -Wl,--rpath -Wl,${HOME}/lib -lquantum
