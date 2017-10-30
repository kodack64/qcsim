#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>
#include <chrono>
#include <bitset>
#include <iostream>
#include <complex>
#include <conio.h>
#include <vector>
#include <fstream>
#include <string>

#define block 256
#define blockThread	1024

__global__ void IKernel(cuDoubleComplex *n, const cuDoubleComplex *o, unsigned const int dim,unsigned const int target)
{
	//int i = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	while (i < dim) {
		n[i] = o[i];
		i += blockDim.x*gridDim.x;
	}
}
__global__ void XKernel(cuDoubleComplex *n, const cuDoubleComplex *o, unsigned const int dim, unsigned const int target)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int shift = 1 << target;
	while (i < dim) {
		n[i] = o[i^shift];
		i += blockDim.x*gridDim.x;
	}
}
__global__ void YKernel(cuDoubleComplex *n, const cuDoubleComplex *o, unsigned const int dim, unsigned const int target)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int shift = 1 << target;
	while (i < dim) {
		int sign = 1 - ((i >> target) % 2) * 2;
		n[i] = cuCmul(
			make_cuDoubleComplex(0,sign),
			o[i^shift]
		);
		i += blockDim.x*gridDim.x;
	}
}
__global__ void ZKernel(cuDoubleComplex *n, const cuDoubleComplex *o, unsigned const int dim, unsigned const int target)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int shift = 1 << target;
	while (i < dim) {
		int sign = 1 - ((i >> target) % 2) * 2;
		n[i] = cuCmul(
			make_cuDoubleComplex(sign, 0),
			o[i]
		);
		i += blockDim.x*gridDim.x;
	}
}
__global__ void SKernel(cuDoubleComplex *n, const cuDoubleComplex *o, unsigned const int dim, unsigned const int target)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int shift = 1 << target;
	while (i < dim) {
		if ( (i>>target)%2) {
			n[i] = cuCmul(
				make_cuDoubleComplex(0, 1),
				o[i]
			);
		}
		else {
			n[i] = o[i];
		}
		i += blockDim.x*gridDim.x;
	}
}
__global__ void TKernel(cuDoubleComplex *n, const cuDoubleComplex *o, unsigned const int dim, unsigned const int target)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int shift = 1 << target;
	while (i < dim) {
		if ((i >> target) % 2) {
			n[i] = cuCmul(
				make_cuDoubleComplex(sqrt(0.5), sqrt(0.5)),
				o[i]
			);
		}
		else {
			n[i] = o[i];
		}
		i += blockDim.x*gridDim.x;
	}
}
__global__ void hadamardKernel(cuDoubleComplex *n, const cuDoubleComplex *o, unsigned const int dim, unsigned const int k)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int shift = 1 << k;
	while (i < dim) {
		int sign = 1 - ((i >> k) % 2) * 2;
		n[i] = cuCadd(
			cuCmul(make_cuDoubleComplex(sqrt(0.5), 0), o[i^shift]),
			cuCmul(make_cuDoubleComplex(sign*sqrt(0.5), 0), o[i])
		);
		i += blockDim.x*gridDim.x;
	}
}
__global__ void cnotKernel(cuDoubleComplex *n, const cuDoubleComplex *o, unsigned const int dim, unsigned const int k,unsigned const int l)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int shift = 1 << l;
	while (i < dim) {
		if ( (i>>k)%2 ) {
			n[i] = o[i^shift];
		}
		else {
			n[i] = o[i];
		}
		i += blockDim.x*gridDim.x;
	}
}

class MyCuda {
private:
	unsigned int _n;
	unsigned int _dim;
	cuDoubleComplex *stateLocal;
	cuDoubleComplex *stateOrg;
	cuDoubleComplex *stateNext;

public:
	MyCuda() :_n(0), _dim(0), stateOrg(0), stateNext(0),stateLocal(0)
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			exit(0);
		}
	}
	virtual ~MyCuda() {
		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
		}
	}

	cudaError_t init(int numQubit) {
		_n = numQubit;
		_dim = 1 << numQubit;

		cudaError_t cudaStatus;
		stateLocal = new cuDoubleComplex[_dim];
		for (unsigned int i = 0; i < _dim; i++) stateLocal[i] = make_cuDoubleComplex(0,0);
		stateLocal[0] = make_cuDoubleComplex(1,0);

		cudaStatus = cudaMalloc((void**)&stateOrg, _dim * sizeof(cuDoubleComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&stateNext, _dim * sizeof(cuDoubleComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(stateOrg, stateLocal, _dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		return cudaStatus;
	Error:
		cudaFree(stateOrg);
		cudaFree(stateNext);
		delete[] stateLocal;
		return cudaStatus;
	}

	void apply1QG(void(*kernel)(cuDoubleComplex*, const cuDoubleComplex*,const unsigned int, const unsigned int), int target) {
		assert(0 <= target && target < _n);
		// Launch a kernel on the GPU with one thread for each element.
		// <<block, thread per block>>
		// max thread per block = 2**10
		// max block = 
		if (_dim <= (1 << 10)) {
			kernel << <1, _dim >> >(stateNext, stateOrg, _dim, target);
		}
		else {

			//kernel << <min(256,_dim/1024), 1024 >> >(stateNext, stateOrg, _dim, target);
			kernel << <min(block,_dim/blockThread), blockThread >> >(stateNext, stateOrg, _dim, target);
		}

		cuDoubleComplex* stateTemp = stateOrg;
		stateOrg = stateNext;
		stateNext = stateOrg;

		/*
		// Check for any errors launching the kernel
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "apply kernel of single qubit operation launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		return cudaStatus;
		*/
	}
	void apply2QG(void(*kernel)(cuDoubleComplex*, const cuDoubleComplex*, const unsigned int, const unsigned int, const unsigned int), int control, int target) {
		assert(0 <= target && target < _n);
		// Launch a kernel on the GPU with one thread for each element.
		// <<block, thread per block>>
		// max thread per block = 2**10
		// max block = 
		if (_dim <= (1 << 10)) {
			kernel << <1, _dim >> >(stateNext, stateOrg, _dim, control, target);
		}
		else {

			//kernel << <min(256,_dim/1024), 1024 >> >(stateNext, stateOrg, _dim, target);
			kernel << <block, blockThread >> >(stateNext, stateOrg, _dim, control, target);
		}

		cuDoubleComplex* stateTemp = stateOrg;
		stateOrg = stateNext;
		stateNext = stateOrg;

		/*
		// Check for any errors launching the kernel
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "apply kernel of two qubit operation launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		return cudaStatus;
		*/
	}

	cudaError_t sync() {
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return cudaStatus;
		}
	}

	cudaError_t getStatus() {
		// Copy output vector from GPU buffer to host memory.
		cudaError_t cudaStatus = cudaMemcpy(stateLocal, stateOrg, _dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
		return cudaStatus;
	}

	void dumpState() {
		getStatus();
		std::cout << "dump" << std::endl;
		double norm = 0;
		for (unsigned int i = 0; i < _dim; i++) { 
			std::complex<double> val = std::complex<double>(stateLocal[i].x, stateLocal[i].y);
			norm += std::pow(std::abs(val),2);
			if (std::abs(val) != 0) {
				std::cout << val << "|";
				for (unsigned int j = 0; j < _n; j++) {
					std::cout << ((i >> j) % 2);
				}
				std::cout << "> ";
			}
		}
		std::cout << std::endl;
		std::cout << norm << std::endl;
		if (fabs(norm-1.0)>1e-8) {
			_getch();
		}
	}

	void close() {
		sync();
		cudaFree(stateOrg);
		cudaFree(stateNext);
		delete[] stateLocal;
	}
};

int test() {
	unsigned int n = 2;
	MyCuda* mc = new MyCuda();
	mc->init(n);
	mc->dumpState();
	mc->apply1QG(YKernel, 0);
	mc->dumpState();
	mc->apply1QG(XKernel, 0);
	mc->dumpState();
	mc->apply1QG(YKernel, 0);
	mc->dumpState();
	mc->apply1QG(hadamardKernel, 0);
	mc->dumpState();
	mc->apply2QG(cnotKernel, 0,1);
	mc->dumpState();
	mc->apply1QG(TKernel, 0);
	mc->dumpState();
	mc->close();
	delete mc;
	return 0;
}

std::vector<__int64> randomCircuitOneshot(unsigned int n, unsigned int depth) {
	std::mt19937 mt(0);
	std::vector<__int64> dur;

	MyCuda* mc = new MyCuda();
	mc->init(n);
	std::fstream ofs("gputime.txt", std::ios::app);
	ofs << n << " ";
	ofs.close();
	for (int d = 0; d < depth; d++) {
		auto start = std::chrono::system_clock::now();
		for (int i = 0; i < n; i++) {
			int r = mt() % 5;
			if (d == 0 && i == 0) r = 4;
			if (r == 0)			mc->apply1QG(XKernel, i);
			else if (r == 1)	mc->apply1QG(TKernel, i);
			else if (r==2)		mc->apply1QG(YKernel, i);
			else if(r==3)		mc->apply1QG(hadamardKernel, i);
			else if (r == 4) {
				if (i + 1 < n) {
					mc->apply2QG(cnotKernel, i, i+1);
					i++;
				}
				else {
					mc->apply1QG(SKernel,i);
				}
			}
		}
		mc->sync();
		__int64 time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
		dur.push_back(time);
		std::cout << d << " " << time << std::endl;

		std::fstream ofsa("gputime.txt", std::ios::app);
		ofsa << time << " ";
		ofsa.close();
	}
	mc->close();
	std::fstream ofse("gputime.txt", std::ios::app);
	ofse << std::endl;
	ofse.close();
	return dur;
}


int main(int argc, char** argv) {
	int n = 27;
	int depth = 100;
	if (argc > 1) {
		n = atoi(argv[1]);
		depth = atoi(argv[2]);
	}
	auto time = randomCircuitOneshot(n,depth);
	__int64 sum = 0;
	for (int i = 0; i < time.size(); i++) {
		sum += time[i];
	}
	std::cout << sum / time.size() << std::endl;

	return 0;
}