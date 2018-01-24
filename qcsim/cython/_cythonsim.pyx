
"""
class of quantum circuit simulator
"""

import numpy as np
cimport numpy as np
cimport cython


ctypedef unsigned int uint

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double calcTrace(double complex[:] s,int dim):
    cdef int i
    cdef double sum = 0.
    for i in range(dim):
        sum += np.real(s[i]*np.conj(s[i]))
    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void applyU(double complex[:] s, double complex[:] ns,int dim, int target, double complex[:] u):
    cdef int mask = 1LL<<target
    cdef int i
    for i in range(dim):
        if(i&mask == 0):
            ns[i] = u[0]*s[i] + u[1]*s[i^mask]
        else:
            ns[i] = u[2]*s[i^mask] + u[3]*s[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void applyCX(double complex[:] s, double complex[:] ns,int dim, int target, int control):
    cdef int mask1 = 1<<target
    cdef int mask2 = 1<<control
    cdef int i
    for i in range(dim):
        if(i&mask1 == 0):
            ns[i] = s[i]
        else:
            ns[i] = s[i^mask2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void applyMeasure(double complex[:] s, double complex[:] ns,int dim, int target, int value):
    cdef int mask = 1<<target
    cdef int check = value<<target
    cdef int i
    for i in range(dim):
        if(i&mask == check):
            ns[i] = s[i]
        else:
            ns[i] = 0

cdef class CythonSimulator:
    cdef int n
    cdef int dim
    cdef np.ndarray state
    cdef np.ndarray nstate
    cdef double currentTrace
    cdef int verbose

    def __init__(self,int n,int verbose=0):
        """
        @param n : number of qubit
        @param verbose : output verbose comments for debug (default: false)
        @return None
        """
        self.n = n
        self.dim = 2**n
        self.state = np.zeros(self.dim,dtype=np.complex128)
        self.state[0]=1.
        self.nstate = np.zeros_like(self.state)
        self.verbose = verbose
        self.currentTrace = -1

    def apply(self,gate,target,control=None,theta=None,param=None,update=True):
        """
        apply quantum gate to the qubit(s)

        @param gate : string or cupy kernel of applying gate
        @param target : target qubit index or indices
        @param control : control qubit index or indices (default: empty list)
        @param theta : rotation angle, used in rotation gate (default: None)
        @param theta : rotation angle, used in rotation gate (default: None)
        @param params : description of unitary operation (default: empty list)
        @update : The calculated state is placed in buffer-state. If update is Ture, swap current state with buffer after calculation. (default: True)
        """
        if target not in [list,np.ndarray]:
            target = [target]
        if control not in [list,np.ndarray]:
            control = [control]
        gate = gate.upper()

        if(gate.upper() == "U"):
            #print(param)
            if(len(param)==3):
                u0 = np.exp(-1j*(param[1]+param[2])/2.) * np.cos(param[0]/2.)
                u1 = -np.exp(-1j*(param[1]-param[2])/2.) * np.sin(param[0]/2.)
                u2 = np.exp(1j*(param[1]-param[2])/2.) * np.sin(param[0]/2.)
                u3 = np.exp(1j*(param[1]+param[2])/2.) * np.cos(param[0]/2.)
                param = [u0,u1,u2,u3]
            #print(param)
            applyU(self.state,self.nstate,self.dim,target[0],np.array(param))
        elif(gate.upper() == "CX"):
            applyCX(self.state,self.nstate,self.dim,target[0],control[0])
        elif(gate.upper() == "M0"):
            applyMeasure(self.state,self.nstate,self.dim,target[0],0)
        elif(gate.upper() == "M1"):
            applyMeasure(self.state,self.nstate,self.dim,target[0],1)
        else:
            raise Exception("not implemented {}".format())

        if(update):
            self.update()
        else:
            return self.trace(buffer=True)

    def update(self):
        """
        swap buffer-state with the current state
        """
        self.state,self.nstate = self.nstate,self.state
        self.currentTrace = -1

    def trace(self,int buffer=0):
        """
        take trace of the quantum state
        @buffer : calculate trace of buffer-state (default: False)
        """
        if(self.verbose): print("Calculate trace")
        if(buffer):
            return np.real(calcTrace(self.nstate,self.dim))
        else:
            val = calcTrace(self.state,self.dim)
            self.currentTrace = val
            return np.real(val)

    def normalize(self,double eps=1e-16):
        """
        normalize quantum state
        @eps : if trace is smaller than eps, raise error for avoiding Nan (default: 1e-16)
        """
        if(self.currentTrace < 0):
            self.trace()
        valtrace = np.real(self.currentTrace)
        if(valtrace<eps):
            raise ValueError("Trace is too small : {}".format(valtrace))
        self.state/=np.sqrt(self.currentTrace)
        if(self.verbose): print("Normalize")

    def asnumpy(self):
        """
        recieve quantum state as numpy matrix.
        Do nothing in numpy
        """
        return self.state

    def tostr(self,eps=1e-10):
        """
        overload string function
        return bra-ket representation of current quantum state (very slow when n is large)
        """
        fst = True
        ret = ""
        for ind in range(self.dim):
            val = self.state[ind]
            if(abs(val)<eps):
                continue
            else:
                if(fst):
                    fst = False
                else:
                    ret += " + "
                ret += str(val) + "|" + format(ind,"b").zfill(self.n)[::-1]+ ">"
        return ret

    def __bound(self,ind):
        return (0<=ind and ind<self.n)
