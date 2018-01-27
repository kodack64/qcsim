#pragma once

#include<stdio.h>

int simulateConsoleStream();
int simulateFileStream(FILE* inStream, FILE* outStream);

#ifdef DLL_EXPORT
__declspec(dllexport) void init(const unsigned int n);
__declspec(dllexport) void u(const unsigned int n, const double u1, const double u2, const double u3);
__declspec(dllexport) void cx(const unsigned int target,const unsigned int control);
__declspec(dllexport) int meas(const unsigned int target);
__declspec(dllexport) void release();
#endif
