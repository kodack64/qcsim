
import subprocess
res = subprocess.Popen(["python","setup.py","build_ext","-i"],stdout = subprocess.PIPE).communicate()

from qcsim.cpu import CpuSimulator
from qcsim.gpu import GpuSimulator
from qcsim.cython import CythonSimulator
from qcsim.cythonomp import CythonOmpSimulator
from qcsim.qasm import QasmSimulator
import numpy as np

def gpuTest():
    sim = GpuSimulator(3)
    sim.apply("CX",0,1)
    sim.apply("Y",1)
    sim.apply("X",2)
    sim.apply("Xrot",1,theta=np.pi/8)
    a = sim.trace()
    print("braket rep: ",sim)

def qasmTextTest():
    qasmText = \
    """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    include \"qelib2.inc\";
    qreg q1[3];
    creg c1[3];
    creg c2[3];
    h q1[0];
    cx q1[0],q1[2];
    //rx(pi) q1[0];
    zrot(pi/4) q1[0];
    xxrot(pi/4) q1[0],q1[2];
    gate test d0,d1{
        xxrot(pi/4) d0,d1;
        zrot(pi/4) d1;
    }
    test q1[0],q1[1];
    //m0 q1[0];
    //measure q1 -> c2;
    if(c2 == 3) test q1[0],q1[1];
    """
    sim = QasmSimulator(data=qasmText,backendName="gpu",verbose=False)
    sim.execute()
    sam = sim.getSample(sampleCount=10)
    print("sample: ",sam)

def qasmFileTest():
    qasmText = \
"""
OPENQASM 2.0;
include \"qelib1.inc\";
include \"qelib2.inc\";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
rz(pi/4) q[0];
x q[0];
rx(pi/16) q[1];
cz q[0],q[1];
t q[1];
//xxrot(pi/4) q[0],q[2];
gate test d0,d1{
    xxrot(pi/4) d0,d1;
    zrot(pi/4) d1;
}
test q[0],q[1];
//measure q -> c;
"""
    fname = "sample.qasm"
    fout = open(fname,"w")
    fout.write(qasmText)
    fout.close()

    backends = ["cpu","gpu","cython","cythonomp","ibmqx"]
    stateVecs = []
    for backend in backends:
        sim = QasmSimulator(file=fname,backendName=backend,verbose=False)
        sim.execute()
        sam = sim.getSample(1024)
        print("sample: ",sam)

        if backend != "ibmqx":
            tr = sim.getTrace()
            vec = sim.getState()
            vecStr = sim.getStringRep()
            stateVecs.append(vec)
            #print("state vec: ", vec/vec[0])
            #print("trace: ",tr)
            print("braket rep:", vecStr)

    for ind in range(len(stateVecs)):
        mind = np.argmax(np.abs(stateVecs[ind])>1e-10)
        stateVecs[ind] /= stateVecs[ind][mind]
        print(stateVecs[ind])
    for y in range(len(stateVecs)):
        for x in range(len(stateVecs)):
            print("OK " if (np.abs(stateVecs[y] - stateVecs[x])<1e-10).all() else "NG ",end="")
        print()

#gpuTest()
#qasmTextTest()
qasmFileTest()
