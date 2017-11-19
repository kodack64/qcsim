# qcsim
Simulate quantum circuits and qasm with various classical backend devices

---

### Usage
Simulate quantum operations or execute qasm. See sample code.

### Backend
- cpu : calculate with numpy function
- cython : calculate with cythonized code
- cythonomp : calculate with cythonized code with OpenMP
- gpu : calculate with GPU
- ibmq : submit qasm to IBMQuantumExperience

### Requirements
- numpy
- cython
- cupy
- IBMQuantumExperience

### Note
- Result is not tested yet
