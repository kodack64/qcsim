
import numpy as np
from qcsim.qasm.backend.base import QasmBackendBase
from IBMQuantumExperience import IBMQuantumExperience as IBMQ

class QasmBackendIbmqx(QasmBackendBase):
    name = "ibmqx"
    basisGates = "u1,u2,u3,cx,id"
    def __init__(self,verbose=False):
        self.verbose = verbose
    def simulate(self,circuitJson,qasm):
        self.qasm = qasm
        envs = {}
        try:
            fin = open("apikey.env")
            for line in fin:
                elem = line.split("=")
                if(len(elem)==2):
                    envs[elem[0].strip()] = elem[1].strip()
            API_KEY = envs["API_KEY"]
        except:
            raise Exception("No Api token is set. Please specify api key at \"apikey.env\"")
        self.api = IBMQ(API_KEY)

    def getState(self):
        print(" !!! real simulator cannot calculate state")
        return None

    def getTrace(self):
        print(" !!! real simulator cannot calculate trace")
        return None

    def getSample(self,sampleCount=1):
        print(self.qasm)
        res = self.api.run_experiment(qasm = self.qasm,backend="simulator",shots=sampleCount)
        if( "error" in res):
            return res["error"]
        else:
            return res["result"]["measure"]

    def __str__(self):
        print(" !!! real simulator cannot provide string representation")
        return ""
