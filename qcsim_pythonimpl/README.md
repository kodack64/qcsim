# qcsim
Quantum circuit simulator with python

---

### Usage

For python:
```python
from qcsim.cpu import CpuSimulator
from qcsim.gpu import GpuSimulator
from qcsim.cython import CythonSimulator
from qcsim.cythonomp import CythonOmpSimulator
import numpy as np

sim = GpuSimulator(n=3)
sim.apply("H",0)
sim.apply("CX",0,1)
sim.apply("X",2)
sim.apply("Xrot",1,theta=np.pi/8)
sim.apply("MeasZ0",1,update=True)
a = sim.trace()
```

For QASM:
```python
from qcsim.qasm import QasmSimulator
qasmText = \
"""
OPENQASM 2.0;
include \"qelib1.inc\";
include \"qelib2.inc\";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[2];
zrot(pi/4) q[0];
xxrot(pi/4) q[0],q[2];
gate hoge d0,d1{
    xxrot(pi/4) d0,d1;
    zrot(pi/4) d1;
}
hoge q[0],q[1];
m0 q[0];
if(c == 0) hoge q[0],q[1];
measure q -> c;
"""
sim = QasmSimulator(data=qasmText,backendName="cythonomp")
sim.execute()
vec = sim.getState()
```
qiskit-sdk is used for converting QASM to Json.

Submit QASM to IBMQuantumExperience:
```python
from qcsim.qasm import QasmSimulator
qasmText = \
"""
OPENQASM 2.0;
include \"qelib1.inc\";
include \"qelib2.inc\";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[2];
measure q -> c;
"""
sim = QasmSimulator(data=qasmText,backendName="ibmqx")
sim.execute()
sample = sim.getSample(1024)
```
To access IBMQX, rename "apikey.env.sample" to "apikey.env" and paste API token.

### Backend
- cpu : use numpy-native function
- cython : use cython-native function
- cythonomp : use cython-native function with OpenMP
- gpu : use GPU with CuPy
- ibmq : submit qasm to IBMQuantumExperience

### Requirements
- numpy
- cython
- cupy
- IBMQuantumExperience
- qiskit-sdk
