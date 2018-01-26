#pragma once

#ifndef DLL_EXPORT
int initDevice();
double* stateAllocate(const int n);
void stateRelease(double* state);
void closeDevice();
#else
__declspec(dllexport) int initDevice();
__declspec(dllexport) double* stateAllocate(const int n);
__declspec(dllexport) void stateRelease(double* state);
__declspec(dllexport) void closeDevice();
#endif 

