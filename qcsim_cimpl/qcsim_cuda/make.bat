gcc -c -O2 main.c simulator.c
nvcc -c -O2 -Xcompiler "/wd 4819" -arch=compute_60 -code=sm_60 allocate_cuda.cu func_cuda.cu
nvcc main.o simulator.o allocate_cuda.obj func_cuda.obj -o qcsim_cuda.exe

rem gcc main.c simulator.c allocate_c.c func_c.c -O2 -o qcsim_c.exe
rem gcc main.c simulator.c allocate_omp.c func_omp.c -O2 -fopenmp -o qcsim_omp.exe

