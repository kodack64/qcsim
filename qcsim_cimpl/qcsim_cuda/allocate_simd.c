
#include "allocate.h"
#include <stdlib.h>

int initDevice() {
	return 0;
}
double* stateAllocate(const int n) {
	size_t dim = ((size_t)1) << n;
	return (double *)malloc(2 * dim * sizeof(double));
}
void stateRelease(double* state) {
	free(state);
}
void closeDevice() {
}