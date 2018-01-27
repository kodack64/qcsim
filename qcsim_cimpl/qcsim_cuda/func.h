#pragma once

#include <stdio.h>

/*
n-qubit non-unitary operation
initialize all qubits
*/
void op_init(double* nstate, const size_t dim);

/*
1qubit unitary operation
u1,u2,u3 is equivalnent to U(\theta,\phi,\lambda) in QASM
*/
void op_u(const double* state, double* nstate, const size_t dim, const unsigned int target, const double u1, const double u2, const double u3);

/*
2qubit unitary operation
control not

"target" must be different from "control"
*/
void op_cx(const double* state, double* nstate, const size_t dim, const unsigned int target, const unsigned int control);

/*
1qubit non-unitary operation
post-select 0-outcome
*/
void op_post0(const double* state, double* nstate, const size_t dim, const unsigned int target, const double norm);

/*
1qubit non-unitary operation
post-select 1-outcome
*/
void op_post1(const double* state, double* nstate, const size_t dim, const unsigned int target, const double norm);

/*
1qubit non-unitary operation
measurement, and return outcome
*/
unsigned int op_meas(const double* state, double* nstate, const size_t dim, const unsigned int target);

/*
calculate probability with which we obtain outcome 1
*/
double stat_prob1(const double* state, double* workspace, const size_t dim, const unsigned int target);

/*
dump all values
*/
void dump_vector(const double* state, const size_t dim, FILE* fp);

