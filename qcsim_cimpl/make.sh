
gcc -O2 -shared -Wall -Wl,-soname,simulator -DDLL_EXPORT -fPIC simulator.c allocate_c.c func_c.c -o qcsim_c.so
gcc -O2 -shared -Wall -fopenmp -Wl,-soname,simulator -DDLL_EXPORT -fPIC simulator.c allocate_omp.c func_omp.c -o qcsim_omp.so
gcc -O2 -shared -Wall -march=native -Wl,-soname,simulator -DDLL_EXPORT -fPIC simulator.c allocate_simd.c func_simd.c -o qcsim_simd.so
gcc -O2 -shared -Wall -march=native -fopenmp -Wl,-soname,simulator -DDLL_EXPORT -fPIC simulator.c allocate_simdomp.c func_simdomp.c -o qcsim_simdomp.so
