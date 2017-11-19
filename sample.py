
from qcsim.gpu import GpuSimulator
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
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[2];
zrot(pi/4) q[0];
xxrot(pi/4) q[0],q[2];
gate test d0,d1{
    xxrot(pi/4) d0,d1;
    zrot(pi/4) d1;
}
test q[0],q[1];
if(c == 3) test q[0],q[1];
measure q -> c;
"""
    fname = "sample.qasm"
    fout = open(fname,"w")
    fout.write(qasmText)
    fout.close()

    sim = QasmSimulator(file=fname,backendName="cython",verbose=False)
    sim.execute()
    tr = sim.getTrace()
    vec = sim.getState()
    vecStr = sim.getStringRep()
    sam = sim.getSample(1024)
    print("state vec: ", vec)
    print("trace: ",tr)
    print("braket rep:", vecStr)
    print("sample: ",sam)

#gpuTest()
#qasmTextTest()
qasmFileTest()
