
import qiskit._openquantumcompiler as openquantumcompiler
from qcsim.qasm.backend.gpu import QasmBackendGpu
from qcsim.qasm.backend.cpu import QasmBackendCpu
from qcsim.qasm.backend.cython import QasmBackendCython
from qcsim.qasm.backend.cythonomp import QasmBackendCythonOmp
from qcsim.qasm.backend.ibmqx import QasmBackendIbmqx

class QasmSimulator:
    def __init__(self,data = None, file = None, backendName = "cpu",verbose=False, APIToken = None):
        backends = [QasmBackendGpu, QasmBackendCpu, QasmBackendCython, QasmBackendCythonOmp, QasmBackendIbmqx]

        backendNames = dict([(sim.name,sim) for sim in backends])
        backendBasisGates = dict([(sim.name,sim.basisGates) for sim in backends ])

        if(data is None and file is None):
            raise Exception("data or file is required")
        if(data is not None and file is not None):
            raise Exception("data or file is required")
        if(data is not None):
            text = data
        else:
            try:
                text = open(file).read()
            except:
                raise Exception("file not found")

        if(backendName not in backendNames):
            raise Exception("unknown backends : choose among "+str(backendNames.keys()))
        basisGates =backendBasisGates[backendName]

        circuit_dag = openquantumcompiler.compile(text,basis_gates=basisGates)
        circuit_json = openquantumcompiler.dag2json(circuit_dag,basis_gates=basisGates)
        self.backendName = backendName
        self.qasmtxt = circuit_dag.qasm(qeflag=True)
        self.circuit = circuit_json
        self.backend = backendNames[backendName]
        self.verbose = verbose
        self.APIToken = APIToken

    def execute(self):
        self.simulator = self.backend(verbose= self.verbose)
        self.simulator.simulate(self.circuit,self.qasmtxt)
    def getState(self):
        if(self.simulator is None):
            raise Exception("simulation is not executed")
        return self.simulator.getState()
    def getTrace(self):
        if(self.simulator is None):
            raise Exception("simulation is not executed")
        return self.simulator.getTrace()
    def getSample(self,sampleCount=1):
        if(self.simulator is None):
            raise Exception("simulation is not executed")
        return self.simulator.getSample(sampleCount)
    def getStringRep(self):
        return str(self.simulator)
