
import numpy as np
from qcsim.qasm.backend.base import QasmBackendBase
from qcsim.cythonomp import CythonOmpSimulator

class QasmBackendCythonOmp(QasmBackendBase):
    name = "cythonomp"
    basisGates = "cx,m0,m1"
    def __init__(self,verbose=False):
        self.vec = None
        self.sim = None
        self.verbose = verbose
    def simulate(self,circuitJson,qasm):

        numQubit = circuitJson["header"]["number_of_qubits"]

        if "number_of_clbits" in circuitJson["header"].keys():
            numBit = circuitJson["header"]["number_of_clbits"]
            clbitsArray = np.zeros(numBit)

        self.sim = CythonOmpSimulator(numQubit,verbose=self.verbose)

        for operation in circuitJson["operations"]:

            gateOps = operation["name"]

            gateTargets = operation["qubits"]

            if "conditional" in operation.keys():
                condition = operation["conditional"]
                condVal = int(condition["val"],0)
                condMask = int(condition["mask"],0)
                flag = True
                for ind in range(numBit):
                    if( (condMask>>ind) %2==1):
                        flag = flag and (condVal%2 == clbitsArray[ind])
                        condVal//=2
                if(not flag):
                    continue

            if "clbits" in operation.keys():
                measureTargets = operation["clbits"]

            if "params" in operation.keys():
                params = operation["params"]

            if(gateOps in ["x","y","z","h","s","t","cx","cz","m0","m1"]):
                if(len(gateTargets)==1):
                    self.sim.apply(gateOps,gateTargets[0])
                elif(len(gateTargets)==2):
                    self.sim.apply(gateOps,gateTargets[0],gateTargets[1])

            elif(gateOps in ["measure"]):
                trace = self.sim.trace()
                prob = self.sim.apply("M0",gateTargets[0],update=False)/trace
                if(np.random.rand()<prob):
                    self.sim.update()
                    clbitsArray[measureTargets[0]] = 0
                else:
                    self.sim.apply("M1",gateTargets[0])
                    clbitsArray[measureTargets[0]] = 1
                self.sim.normalize()

            elif(gateOps in ["U"]):
                self.sim.apply("U",gateTargets[0],param=params)

            else:
                print(" !!! {} is not supported !!!".format(gateOps))
                print(operation)

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
        return self.sim.tostr()
