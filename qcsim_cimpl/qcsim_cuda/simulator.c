

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdlib.h>
#include <stdarg.h>
#include "simulator.h"
#include "func.h"
#include "allocate.h"

#ifdef DLL_EXPORT
unsigned int g_numQubit;
size_t g_dim;
double* g_state;
double* g_stateBuf;

void init(const unsigned int n) {
	g_numQubit = n;
	g_dim = ((size_t)1) << n;
	initDevice();
	g_state = stateAllocate(n);
	g_stateBuf = stateAllocate(n);
	op_init(g_state,g_dim);
}
void u(const unsigned int target, const double u1, const double u2, const double u3) {
	op_u(g_state,g_stateBuf,g_dim,target,u1,u2,u3);
	double* ptr = g_state;
	g_state = g_stateBuf;
	g_stateBuf = ptr;
}
void cx(const unsigned int target, const unsigned int control) {
	op_cx(g_state, g_stateBuf, g_dim, target, control);
	double* ptr = g_state;
	g_state = g_stateBuf;
	g_stateBuf = ptr;
}
int meas(const unsigned int target) {
	int res = op_meas(g_state, g_stateBuf, g_dim, target);
	double* ptr = g_state;
	g_state = g_stateBuf;
	g_stateBuf = ptr;
	return res;
}
void release() {
	closeDevice();
	stateRelease(g_state);
	stateRelease(g_stateBuf);
}
#endif








int simulateConsoleStream() {
	return simulateFileStream(stdin, stdout);
}

int simulateFileStream(FILE* inStream,FILE* outStream) {
	unsigned int n;
	unsigned int operatorType;
	unsigned int targetQubit, controlQubit;
	unsigned int outcome;
	size_t dim;
	double u1, u2, u3;
	double *state, *stateBuf, *tempPtr;
	int ret;
	unsigned int swapFlag;

	// obtain number of qubits
	ret = fscanf(inStream,"%d", &n);
	if (ret == EOF) goto Error_InvalidMessage;
	dim = ((size_t)1) << n;

	// alloc memory for state vector
	// (assume 0r,0i,1r,1i,...)
	initDevice();
	state = stateAllocate(n);
	if (state == NULL) goto Error_MemoryError;
	stateBuf = stateAllocate(n);
	if (stateBuf == NULL) goto Error_MemoryError;
	op_init(state, dim);


	// loop until obtain end message
	while (1) {
		ret = fscanf(inStream,"%d", &operatorType);
		if (ret == EOF) goto Error_InvalidMessage;
		swapFlag = 0;

		// update quantum state
		// unitary op
		if (operatorType == 0) {
			ret = fscanf(inStream, "%d %lf %lf %lf", &targetQubit, &u1, &u2, &u3);
			if (ret == EOF) goto Error_InvalidMessage;
			if (!(0 <= targetQubit && targetQubit < n)) goto Error_OutOfRange;

			op_u(state, stateBuf, dim, targetQubit, u1, u2, u3);
			swapFlag = 1;
		}

		// cx op
		else if (operatorType == 1) {
			ret = fscanf(inStream, "%d %d", &controlQubit, &targetQubit);
			if (ret == EOF) goto Error_InvalidMessage;
			if (!(0 <= targetQubit && targetQubit < n)) goto Error_OutOfRange;
			if (!(0 <= controlQubit && controlQubit < n)) goto Error_OutOfRange;
			if (controlQubit == targetQubit) goto Error_SameControlTarget;

			op_cx(state, stateBuf, dim, targetQubit, controlQubit);
			swapFlag = 1;
		}

		// meas op
		else if (operatorType == 2) {
			ret = fscanf(inStream, "%d", &targetQubit);
			if (ret == EOF) goto Error_InvalidMessage;
			if (!(0 <= targetQubit && targetQubit < n)) goto Error_OutOfRange;

			outcome = op_meas(state, stateBuf, dim, targetQubit);
			fprintf(outStream, "%d\n", outcome);
			swapFlag = 1;
		}

		// init op
		else if (operatorType == 3) {
			op_init(state, dim);
			swapFlag = 0;
		}

		// state dump
		else if (operatorType == 4) {
			dump_vector(state, dim, outStream);
			swapFlag = 0;
		}

		// exit
		else if (operatorType == 5) {
			break;
		}
		else goto Error_InvalidMessage;


		// swap buffer
		if (swapFlag) {
			tempPtr = state;
			state = stateBuf;
			stateBuf = tempPtr;
		}
	}

	// release
	stateRelease(state);
	stateRelease(stateBuf);
	closeDevice();
	return 0;

Error_MemoryError:
	fprintf(stderr, "Cannot allocate memory\n");
	return 1;
Error_InvalidMessage:
	fprintf(stderr, "Message format is invalid\n");
	stateRelease(state);
	stateRelease(stateBuf);
	closeDevice();
	return 2;
Error_OutOfRange:
	fprintf(stderr, "Invalid qubit ID is specified\n");
	stateRelease(state);
	stateRelease(stateBuf);
	closeDevice();
	return 3;
Error_SameControlTarget:
	fprintf(stderr, "Control and target qubits are the same in cnot\n");
	stateRelease(state);
	stateRelease(stateBuf);
	closeDevice();
	return 4;
}



