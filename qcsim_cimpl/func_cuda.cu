
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>

extern "C" {
#include "func.h"
#include "random.h"
}
#include "common_cuda.h"

#define MIN(p,q) (p<q?p:q)
#define MAX(p,q) (p>q?p:q)

/*
n-qubit non-unitary operation
initialize all qubits
*/
__global__ void kernel_op_init(double* nstate, const size_t dim) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	while (i < dim) {
		nstate[2 * i] = 0.;
		nstate[2 * i + 1] = 0.;
		if (i == 0) nstate[2 * i] = 1.;

		i += blockDim.x * gridDim.x;
	}
}
void op_init(double* nstate, const size_t dim) {
	unsigned int blockCount, threadCount;
	threadCount = min((unsigned int)dim,g_maxThreadsPerBlock);
	blockCount = max((unsigned int)dim/g_maxThreadsPerBlock,1);

	kernel_op_init << < blockCount, threadCount >> > (nstate, dim);
	cudaThreadSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { 
		fprintf(stderr, "cuda op_init failed : %s\n", cudaGetErrorString(cudaStatus)); 
	}
}



/*
1qubit unitary operation
u1,u2,u3 is equivalnent to U(\theta,\phi,\lambda) in QASM
*/
__global__ void kernel_op_u(double *state, const size_t dim, const size_t targetMask,
	const double u00r, const double u00i, const double u01r, const double u01i, const double u10r, const double u10i, const double u11r, const double u11i) {

	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int maxind = dim / 2;
	const unsigned int targetMaskm = targetMask - 1;
	while (i < maxind) {
		size_t t1 = (i&targetMaskm) ^ ((i&(~targetMaskm)) << 1);
		size_t t2 = t1^targetMask;
		double a1r = state[2 * t1];
		double a1i = state[2 * t1 + 1];
		double a2r = state[2 * t2];
		double a2i = state[2 * t2 + 1];
		state[2 * t1] = u00r * a1r - u00i * a1i + u01r * a2r - u01i * a2i;
		state[2 * t1 + 1] = u00i * a1r + u00r * a1i + u01i * a2r + u01r * a2i;
		state[2 * t2] = u10r * a1r - u10i * a1i + u11r * a2r - u11i * a2i;
		state[2 * t2 + 1] = u10i * a1r + u10r * a1i + u11i * a2r + u11r * a2i;
		i += blockDim.x * gridDim.x;
	}
}
void op_u(double* state, const size_t dim, const unsigned int target, const double u1, const double u2, const double u3) {
	const size_t targetMask = ((size_t)1) << target;
	double u00r, u01r, u10r, u11r, u00i, u01i, u10i, u11i;
	unsigned int blockCount, threadCount;

	u00r = cos((u2 + u3) / 2) * cos(u1 / 2);
	u00i = -sin((u2 + u3) / 2) * cos(u1 / 2);
	u01r = -cos((u2 - u3) / 2) * sin(u1 / 2);
	u01i = sin((u2 - u3) / 2) * sin(u1 / 2);
	u10r = cos((u2 - u3) / 2) * sin(u1 / 2);
	u10i = sin((u2 - u3) / 2) * sin(u1 / 2);
	u11r = cos((u2 + u3) / 2) * cos(u1 / 2);
	u11i = sin((u2 + u3) / 2) * cos(u1 / 2);

	threadCount = min((unsigned int)dim, g_maxThreadsPerBlock);
	blockCount = max((unsigned int)dim / g_maxThreadsPerBlock, 1);

	kernel_op_u << < blockCount, threadCount >> > (state, dim, targetMask, u00r, u00i, u01r, u01i, u10r, u10i, u11r, u11i);
	cudaThreadSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda op_u failed : %s\n", cudaGetErrorString(cudaStatus));
	}
}


/*
2qubit unitary operation
control not

"target" must be different from "control"
*/
__global__ void kernel_op_cx(double *state, const size_t dim, const size_t targetMask, const size_t controlMask) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int maxind = dim / 4;
	const unsigned int mask1 = (min(targetMask, controlMask)) - 1;
	const unsigned int mask2 = (max(targetMask, controlMask)) - 1;
	for (i = 0; i < maxind; i++) {
		size_t t1, t2;
		t1 = (i&mask1) ^ ((i&(~mask1)) << 1);
		t1 = (t1&mask2) ^ ((t1&(~mask2)) << 1) ^ controlMask;
		t2 = t1^targetMask;
		double a1r = state[2 * t1];
		double a1i = state[2 * t1 + 1];
		double a2r = state[2 * t2];
		double a2i = state[2 * t2 + 1];
		state[2 * t1] = a2r;
		state[2 * t1 + 1] = a2i;
		state[2 * t2] = a1r;
		state[2 * t2 + 1] = a1i;
		i += blockDim.x * gridDim.x;
	}
}
void op_cx(double* state, const size_t dim, const unsigned int target, const unsigned int control) {
	const size_t targetMask = ((size_t)1) << target;
	const size_t controlMask = ((size_t)1) << control;
	unsigned int blockCount, threadCount;

	threadCount = min((unsigned int)dim, g_maxThreadsPerBlock);
	blockCount = max((unsigned int)dim / g_maxThreadsPerBlock, 1);

	kernel_op_cx << < blockCount, threadCount >> > (state, dim, targetMask, controlMask);
	cudaThreadSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda op_cx failed : %s\n", cudaGetErrorString(cudaStatus));
	}
}


/*
1qubit non-unitary operation
post-select 0-outcome
*/
__global__ void kernel_op_post0(const double *state, double* nstate, const size_t dim, const size_t targetMask, const double norm) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	while (i < dim) {
		if ((i&targetMask) == 0) {
			nstate[2 * i] = state[2 * i] * norm;
			nstate[2 * i + 1] = state[2 * i + 1] * norm;
		}
		else {
			nstate[2 * i] = 0;
			nstate[2 * i + 1] = 0;
		}

		i += blockDim.x * gridDim.x;
	}
}

void op_post0(const double* state, double* nstate, const size_t dim, const unsigned int target, const double norm) {
	const size_t targetMask = ((size_t)1) << target;
	unsigned int blockCount, threadCount;

	threadCount = min((unsigned int)dim, g_maxThreadsPerBlock);
	blockCount = max((unsigned int)dim / g_maxThreadsPerBlock, 1);

	kernel_op_post0 << <blockCount, threadCount >> > (state, nstate, dim, targetMask, norm);
	cudaThreadSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda op_post0 failed : %s\n", cudaGetErrorString(cudaStatus));
	}
}

/*
1qubit non-unitary operation
post-select 1-outcome
*/
__global__ void kernel_op_post1(const double *state, double* nstate, const size_t dim, const size_t targetMask, const double norm) {

	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	while (i < dim) {
		if (i&targetMask) {
			nstate[2 * i] = state[2 * i] * norm;
			nstate[2 * i + 1] = state[2 * i + 1] * norm;
		}
		else {
			nstate[2 * i] = 0;
			nstate[2 * i + 1] = 0;
		}

		i += blockDim.x * gridDim.x;
	}
}
void op_post1(const double* state, double* nstate, const size_t dim, const unsigned int target, const double norm) {
	const size_t targetMask = ((size_t)1) << target;
	unsigned int blockCount, threadCount;

	threadCount = min((unsigned int)dim, g_maxThreadsPerBlock);
	blockCount = max((unsigned int)dim / g_maxThreadsPerBlock, 1);

	kernel_op_post1 << <blockCount, threadCount >> > (state, nstate, dim, targetMask, norm);
	cudaThreadSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda op_post1 failed : %s\n", cudaGetErrorString(cudaStatus));
	}
}

/*
1qubit non-unitary operation
measurement, and return outcome
*/
unsigned int op_meas(const double* state, double* nstate, const size_t dim, const unsigned int target) {
	double prob1;
	double randomValue;
	double norm;
	unsigned int outcome;

	prob1 = stat_prob1(state, nstate, dim, target);
	randomValue = rng();
	if (randomValue > prob1) {
		outcome = 0;
		norm = 1. / sqrt(1 - prob1);
		op_post0(state, nstate, dim, target, norm);
	}
	else {
		outcome = 1;
		norm = 1. / sqrt(prob1);
		op_post1(state, nstate, dim, target, norm);
	}
	return outcome;
}

/*
calculate probability with which we obtain outcome 1
*/
template <unsigned int blockSize>
__global__ void kernel_stat_prob1_optimized(const double *g_idata, double *g_odata, const size_t N, const size_t targetMask)
{
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < N) {
		if((i/2)&targetMask)				
			sdata[tid] += g_idata[i] * g_idata[i];
		if(((i+blockSize)/2)&targetMask)	
			sdata[tid] += g_idata[i + blockSize] * g_idata[i + blockSize];
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 1024){ if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
template <unsigned int blockSize>
__global__ void kernel_stat_sum_optimized(const double *g_idata, double *g_odata, const size_t N)
{
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < N) {
		sdata[tid] += g_idata[i] + g_idata[i + blockSize];
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void kernel_stat_prob1(const double *g_idata, double *g_odata, const size_t N, const size_t targetMask)
{
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = ((i/2)&targetMask) ? g_idata[i] * g_idata[i] : 0.;
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void kernel_stat_sum(const double *g_idata, double *g_odata, const size_t N)
{
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i]; 
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

double stat_prob1(const double* state, double* workspace, const size_t dim, const unsigned int target) {
	const size_t targetMask = ((size_t)1) << target;
	double prob1 = 0.;
	unsigned int threadCount, blockCount, sharedMemSize, cursor, N;
	cudaError_t cudaStatus;

	N = (unsigned int)dim * 2;
	threadCount = min(N, g_maxThreadsPerBlock);
	blockCount = max((unsigned int)N / (2*g_maxThreadsPerBlock), 1);
	sharedMemSize = threadCount * sizeof(double);

	// mapping squared values from state to workspace 
	if (threadCount == 1024 && blockCount>1)	kernel_stat_prob1_optimized<1024> << <blockCount, threadCount, sharedMemSize >> > (state, workspace, N, targetMask);
	else										kernel_stat_prob1 << <blockCount, threadCount, sharedMemSize >> > (state, workspace, N, targetMask);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "stat_prob1 fail : %s\n", cudaGetErrorString(cudaStatus));
	}

	// accumulate workspace with length = blockCount
	cursor = 0;
	N = blockCount;
	while(N>1) {
		threadCount = min(N, g_maxThreadsPerBlock);
		blockCount = max((unsigned int)N / (2 * g_maxThreadsPerBlock), 1);
		sharedMemSize = threadCount * sizeof(double);

		double* fromPtr = workspace + cursor;
		double* toPtr = workspace + cursor + N;
		if (threadCount == 1024 && blockCount > 1)	kernel_stat_sum_optimized<1024> << <blockCount, threadCount, sharedMemSize >> > (fromPtr, toPtr , N);
		else										kernel_stat_sum << <blockCount, threadCount, sharedMemSize >> > (fromPtr, toPtr, N);
		cudaThreadSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "stat_prob1 loop fail : %s\n", cudaGetErrorString(cudaStatus));
		}

		cursor += N;
		N = blockCount;
	}

	cudaStatus = cudaMemcpy(&prob1, workspace + cursor, sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy fail : %s\n", cudaGetErrorString(cudaStatus));
	}
	//printf("prob1 : %lf\n", prob1);
	return prob1;
}

void dump_vector(const double* state, const size_t dim, FILE* outStream) {
	size_t i; 
	double norm = 0.;
	double* local = (double*)malloc(2 * dim * sizeof(double));
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(local,state,2*dim*sizeof(double),cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}
	else {
		for (i = 0; i < dim; i++) {
			fprintf(outStream, "%lld : %lf , %lf\n", i, local[2 * i], local[2 * i + 1]);
			norm += local[2 * i] * local[2 * i] + local[2 * i + 1] * local[2 * i + 1];
		}
		printf("norm :%lf\n", norm);
	}
	free(local);
}