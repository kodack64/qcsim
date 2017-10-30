class QasmBackendBase:
    name = ""
    basisGates = ""
    def __init__(self,verbose=False):
        raise NotImplementedError("abstract class")
    def simulate(self,circuitJson):
        raise NotImplementedError("abstract class")
    def getState(self):
        raise NotImplementedError("abstract class")
    def getTrace(self):
        raise NotImplementedError("abstract class")
    def getSample(self,sampleCount):
        raise NotImplementedError("abstract class")
