
#include <math.h>
#include "func.h"
#include "random.h"
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>

#define MIN(p,q) (p<q?p:q)
#define MAX(p,q) (p>q?q:p)

/*
n-qubit non-unitary operation
initialize all qubits
*/
void op_init(double* nstate, const size_t dim) {
	size_t i;
	for (i = 1; i < 2 * dim; i++) nstate[i] = 0.;
	nstate[0] = 1.;
}

/*
1qubit unitary operation
u1,u2,u3 is equivalnent to U(\theta,\phi,\lambda) in QASM
*/
void op_u(double* state, const size_t dim, const unsigned int target, const double u1, const double u2, const double u3) {
	double u00r, u01r, u10r, u11r, u00i, u01i, u10i, u11i;

	u00r = cos((u2 + u3) / 2) * cos(u1 / 2);
	u00i = -sin((u2 + u3) / 2) * cos(u1 / 2);
	u01r = -cos((u2 - u3) / 2) * sin(u1 / 2);
	u01i = sin((u2 - u3) / 2) * sin(u1 / 2);
	u10r = cos((u2 - u3) / 2) * sin(u1 / 2);
	u10i = sin((u2 - u3) / 2) * sin(u1 / 2);
	u11r = cos((u2 + u3) / 2) * cos(u1 / 2);
	u11i = sin((u2 + u3) / 2) * cos(u1 / 2);

	if (target == 0) {
#pragma omp parallel
		{
			const long long maxind = dim / 2;
			const int corenum = omp_get_num_threads();
			const long long block = maxind / corenum;
			const int residual = maxind%corenum;

			long long i = 0;
			const int coreind = omp_get_thread_num();
			const long long start = block*coreind + MIN(residual,coreind);
			const long long end = block*(coreind+1) + MIN(residual,coreind+1);

			__m256d r0 = _mm256_set_pd(u00r, -u00i, u01r, -u01i);
			__m256d r1 = _mm256_set_pd(u00i, u00r, u01i, u01r);
			__m256d r2 = _mm256_set_pd(u10r, -u10i, u11r, -u11i);
			__m256d r3 = _mm256_set_pd(u10i, u10r, u11i, u11r);
			for (i = start; i < end; i++) {
				double* ptr = state + i*4;
				__m256d st_in0 = _mm256_loadu_pd(ptr);
				__m256d st_out0 = _mm256_hadd_pd(_mm256_mul_pd(st_in0, r0), _mm256_mul_pd(st_in0, r1));
				__m256d st_out1 = _mm256_hadd_pd(_mm256_mul_pd(st_in0, r2), _mm256_mul_pd(st_in0, r3));
				__m256d blend = _mm256_blend_pd(st_out0, st_out1, 0b1100);
				__m256d perm = _mm256_permute2f128_pd(st_out0, st_out1, 0x21);
				__m256d st_out = _mm256_add_pd(perm, blend);
				_mm256_storeu_pd(ptr, st_out);
			}
		}
	}
	else {

#pragma omp parallel
		{
			const long long maxind = dim / 4;
			const int corenum = omp_get_num_threads();
			const long long block = maxind / corenum;
			const int residual = maxind%corenum;

			const size_t targetMask = ((size_t)2) << target;
			const size_t targetMaskm = targetMask - 1;

			long long i = 0;
			const int coreind = omp_get_thread_num();
			const long long start = block*coreind + MIN(residual, coreind);
			const long long end = block*(coreind + 1) + MIN(residual, coreind + 1);
			__m256d r00 = _mm256_set_pd(u00r, -u00i, u00r, -u00i);
			__m256d r01 = _mm256_set_pd(u01r, -u01i, u01r, -u01i);
			__m256d r10 = _mm256_set_pd(u00i, u00r, u00i, u00r);
			__m256d r11 = _mm256_set_pd(u01i, u01r, u01i, u01r);
			__m256d r20 = _mm256_set_pd(u10r, -u10i, u11r, -u11i);
			__m256d r21 = _mm256_set_pd(u10r, -u10i, u11r, -u11i);
			__m256d r30 = _mm256_set_pd(u10i, u10r, u11i, u11r);
			__m256d r31 = _mm256_set_pd(u10i, u10r, u11i, u11r);
			for (i = start ; i < end; i++ ){
				size_t t = i<<2;
				t = (t&targetMaskm) ^ ((t&(~targetMaskm)) << 1);
				double* pt1 = state + t;
				double* pt2 = state + (t^targetMask);

				__m256d st_in0 = _mm256_loadu_pd(pt1);
				__m256d st_in1 = _mm256_loadu_pd(pt2);

				__m256d st_out0 = _mm256_add_pd(_mm256_mul_pd(st_in0, r00), _mm256_mul_pd(st_in1, r01));
				__m256d st_out1 = _mm256_add_pd(_mm256_mul_pd(st_in0, r10), _mm256_mul_pd(st_in1, r11));
				__m256d st_out01 = _mm256_hadd_pd(st_out1, st_out0);
				_mm256_storeu_pd(pt1, st_out01);

				__m256d st_out2 = _mm256_add_pd(_mm256_mul_pd(st_in0, r20), _mm256_mul_pd(st_in1, r21));
				__m256d st_out3 = _mm256_add_pd(_mm256_mul_pd(st_in0, r30), _mm256_mul_pd(st_in1, r31));
				__m256d st_out23 = _mm256_hadd_pd(st_out2, st_out3);
				_mm256_storeu_pd(pt2, st_out23);

			}
		}
	}
}


/*
2qubit unitary operation
control not

"target" must be different from "control"
*/
void op_cx(double* state, const size_t dim, const unsigned int target, const unsigned int control) {
	const size_t targetMask = ((size_t)1) << target;
	const size_t controlMask = ((size_t)1) << control;
	const size_t mask1 = (((size_t)1) << MIN(target,control)) - 1;
	const size_t mask2 = (((size_t)1) << MAX(target, control)) - 1;
	const long long maxind = dim / 4;
	long long  i;
#pragma omp parallel for
	for (i = 0; i < maxind; i++) {
		size_t t1,t2;
		t1 = (i&mask1) ^ ((i&(~mask1)) << 1);
		t1 = (t1&mask2) ^ ((t1&(~mask2)) << 1) ^ controlMask;
		t2 = t1^targetMask;
		double a1r = state[2 * t1];
		double a1i = state[2 * t1 + 1];
		double a2r = state[2 * t2];
		double a2i = state[2 * t2 + 1];
		state[2 * t1]		= a2r;
		state[2 * t1 + 1]	= a2i;
		state[2 * t2]		= a1r;
		state[2 * t2 + 1]	= a1i;
	}
}

/*
calculate probability with which we obtain outcome 1
*/
double stat_prob1(const double* state, double* workspace, const size_t dim, const unsigned int target) {
	size_t i;
	const size_t targetMask = ((size_t)1) << target;
	double prob1 = 0.;
	for (i = 0; i < dim; i++) {
		if (i&targetMask) {
			prob1 += state[2 * i] * state[2 * i];
			prob1 += state[2 * i + 1] * state[2 * i + 1];
		}
	}
	return prob1;
}

/*
1qubit non-unitary operation
post-select 0-outcome
*/
void op_post0(const double* state, double* nstate, const size_t dim, const unsigned int target, const double norm) {
	size_t i;
	const size_t targetMask = ((size_t)1) << target;

	for (i = 0; i < dim; i++) {
		if ((i&targetMask) == 0) {
			nstate[2 * i] = state[2 * i] * norm;
			nstate[2 * i + 1] = state[2 * i + 1] * norm;
		}
		else {
			nstate[2 * i] = 0;
			nstate[2 * i + 1] = 0;
		}
	}
}

/*
1qubit non-unitary operation
post-select 1-outcome
*/
void op_post1(const double* state, double* nstate, const size_t dim, const unsigned int target, const double norm) {
	size_t i;
	const size_t targetMask = ((size_t)1) << target;
	for (i = 0; i < dim; i++) {
		if (i&targetMask) {
			nstate[2 * i] = state[2 * i] * norm;
			nstate[2 * i + 1] = state[2 * i + 1] * norm;
		}
		else {
			nstate[2 * i] = 0;
			nstate[2 * i + 1] = 0;
		}
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

void dump_vector(const double* state, const size_t dim, FILE* outStream) {
	size_t i;
	double norm = 0.;
	for (i = 0; i < dim; i++) {
		fprintf(outStream, "%zd : %lf , %lf\n", i, state[2 * i], state[2 * i + 1]);
		norm += state[2 * i] * state[2 * i] + state[2 * i + 1] * state[2 * i + 1];
	}
	printf("norm :%lf\n", norm);
}