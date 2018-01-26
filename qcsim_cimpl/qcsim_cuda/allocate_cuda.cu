
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "allocate.h"
}
#include "common_cuda.h"

unsigned int g_maxThreadsPerBlock;
unsigned int g_maxBlocksPerGrid;

int initDevice() {
	cudaError_t  cudaStatus;

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceProperties failed : %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	g_maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
	g_maxBlocksPerGrid = deviceProp.maxGridSize[0];

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed : %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	return 0;
}

double* stateAllocate(const int n) {
	cudaError_t  cudaStatus;
	size_t dim = ((size_t)1) << n;
	if (dim >= g_maxThreadsPerBlock * g_maxBlocksPerGrid) {
		fprintf(stderr, "Too many elements : %lld elements required, but allowed size are %ld * %ld \n", dim, g_maxThreadsPerBlock, g_maxBlocksPerGrid);
		return NULL;
	}

	double* ptr;
	cudaStatus = cudaMalloc((void**)&ptr, 2*dim * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed : %s\n", cudaGetErrorString(cudaStatus));
		return NULL;
	}
	return ptr;
}

void stateRelease(double* state) {
	cudaFree(state);
}

void closeDevice() {
}