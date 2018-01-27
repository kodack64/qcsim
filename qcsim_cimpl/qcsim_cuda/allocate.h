#pragma once

int initDevice();
double* stateAllocate(const int n);
void stateRelease(double* state);
void closeDevice();
