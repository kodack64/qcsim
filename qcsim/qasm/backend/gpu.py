
import numpy as np
from qcsim.qasm.backend.base import QasmBackendBase
from qcsim.gpu import GpuSimulator

class QasmBackendGpu(QasmBackendBase):
    name = "gpu"
    basisGates = "x,y,z,h,s,t,cx,cz,m0,m1,xrot,yrot,zrot,xxrot"
    def __init__(self,verbose=False):
        self.vec = None
        self.sim = None
        self.verbose = verbose
    def simulate(self,circuitJson):
        numQubit = circuitJson["header"]["number_of_qubits"]
        self.sim = GpuSimulator(numQubit,verbose=self.verbose)
        for operation in circuitJson["operations"]:
            gateTargets = operation["qubits"]
            gateOps = operation["name"]
            mapper = {
                "x":"X",
                "y":"Y",
                "z":"Z",
                "h":"H",
                "s":"S",
                "t":"T",
                "cx":"CX",
                "cz":"CZ",
                "m0":"MeasZ0",
                "m1":"MeasZ1",
                "xrot":"Xrot",
                "yrot":"Yrot",
                "zrot":"Zrot",
                "xxrot":"XXrot"
            }
            if(gateOps in ["x","y","z","h","s","t","cx","cz","m0","m1"]):
                if(len(gateTargets)==1):
                    self.sim.apply(mapper[gateOps],gateTargets[0])
                elif(len(gateTargets)==2):
                    self.sim.apply(mapper[gateOps],gateTargets[0],gateTargets[1])
            elif(gateOps in ["xrot","yrot","zrot","xxrot"]):
                theta = operation["params"][0]
                if(len(gateTargets)==1):
                    self.sim.apply(mapper[gateOps],gateTargets[0],theta=theta)
                elif(len(gateTargets)==2):
                    self.sim.apply(mapper[gateOps],gateTargets[0],gateTargets[1],theta=theta)
            else:
                print(" !!! {} is not supported !!!".format(gateOps))
    def getState(self):
        self.vec = self.sim.asnumpy()
        return self.vec
    def getTrace(self):
        return self.sim.trace()
    def getSample(self,sampleCount=1):
        if(self.vec is None):
            self.vec = self.sim.asnumpy()
        prob = abs(self.vec)**2
        return np.random.choice(len(self.vec),sampleCount,p=prob)
    def __str__(self):
        return str(self.sim)
