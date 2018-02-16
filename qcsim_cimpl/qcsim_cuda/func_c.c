
#include <math.h>
#include "func.h"
#include "random.h"

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
	size_t i;
	const size_t targetMask = ((size_t)1) << target;
	const size_t targetMaskm = targetMask - 1;
	double u00r, u01r, u10r, u11r, u00i, u01i, u10i, u11i;

	u00r = cos((u2 + u3) / 2) * cos(u1 / 2);
	u00i = -sin((u2 + u3) / 2) * cos(u1 / 2);
	u01r = -cos((u2 - u3) / 2) * sin(u1 / 2);
	u01i = sin((u2 - u3) / 2) * sin(u1 / 2);
	u10r = cos((u2 - u3) / 2) * sin(u1 / 2);
	u10i = sin((u2 - u3) / 2) * sin(u1 / 2);
	u11r = cos((u2 + u3) / 2) * cos(u1 / 2);
	u11i = sin((u2 + u3) / 2) * cos(u1 / 2);

	for (i = 0; i < dim / 2; i++) {
		size_t t1 = (i&targetMaskm) ^ ((i&(~targetMaskm)) << 1);
		size_t t2 = t1^targetMask;
		double a1r = state[2 * t1];
		double a1i = state[2 * t1 + 1];
		double a2r = state[2 * t2];
		double a2i = state[2 * t2 + 1];
		state[2*t1]		= u00r * a1r - u00i * a1i + u01r * a2r - u01i * a2i;
		state[2*t1+1]	= u00i * a1r + u00r * a1i + u01i * a2r + u01r * a2i;
		state[2*t2]		= u10r * a1r - u10i * a1i + u11r * a2r - u11i * a2i;
		state[2*t2+1]	= u10i * a1r + u10r * a1i + u11i * a2r + u11r * a2i;
	}
}


/*
2qubit unitary operation
control not

"target" must be different from "control"
*/
void op_cx(double* state, const size_t dim, const unsigned int target, const unsigned int control) {
	size_t i;
	size_t tmp;
	const size_t targetMask = ((size_t)1) << target;
	const size_t targetMaskm = targetMask - 1;
	const size_t controlMask = ((size_t)1) << control;
	const size_t controlMaskm = controlMask-1;
	// does not work when cont < target
	for (i = 0; i < dim / 4; i++) {
		size_t t1;
		t1 = (i&targetMaskm) ^ ((i&(~targetMaskm)) << 1);
		t1 = (t1&controlMaskm) ^ ((t1&(~controlMaskm)) << 1) ^ controlMask;
		size_t t2 = t1^targetMask;
		double a1r = state[2 * t1];
		double a1i = state[2 * t1 + 1];
		double a2r = state[2 * t2];
		double a2i = state[2 * t2 + 1];
		state[2 * t1] = a2r;
		state[2 * t1 + 1] = a2i;
		state[2 * t2] = a1r;
		state[2 * t2 + 1] = a1i;
	}
	/*
	for (i = 0; i < dim; i++) {
		if (i&controlMask) {
			tmp = i^targetMask;
			nstate[2 * i] = state[2 * tmp];
			nstate[2 * i + 1] = state[2 * tmp + 1];
		}
		else {
			nstate[2 * i] = state[2 * i];
			nstate[2 * i + 1] = state[2 * i + 1];
		}
	}
	*/
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
		fprintf(outStream, "%lld : %lf , %lf\n", i, state[2 * i], state[2 * i + 1]);
		norm += state[2 * i] * state[2 * i] + state[2 * i + 1] * state[2 * i + 1];
	}
	printf("norm :%lf\n", norm);
}