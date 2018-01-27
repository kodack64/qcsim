
from ctypes import *
import numpy as np
import time
import matplotlib.pyplot as plt

path = "./x64/ReleaseDLL/"
dllNames = ["qcsim_c.dll","qcsim_omp.dll","qcsim_cuda.dll"]
maxn = 27

for dllName in dllNames:
	dll = cdll.LoadLibrary(path+dllName)
	times = []
	nran = np.arange(1,maxn+1)
	for n in nran:
		dll.init(c_int(n))
		
		st = time.time()
		depth = 0
		gc = 0
		while( (time.time()-st) < 1. ):
			for ind in range(n):
				r1,r2,r3 = np.random.rand(),np.random.rand(),np.random.rand()
				dll.u(ind,c_double(r1*np.pi),c_double(r2*np.pi),c_double(r3*np.pi))
				gc+=1
			ind = depth%2
			while(ind+1 < n):
				dll.cx(ind,ind+1)
				gc+=1
				ind+=2
			depth += 1
		res = ""
		for ind in range(n):
			res += str(dll.meas(ind))
			gc+=1
		gpt = (time.time()-st)/gc
		print(res)
		print(dllName,n,gpt)
		dll.release()
		times.append(gpt)
	plt.plot(nran,times,label=dllName)
plt.yscale("log")
plt.legend()
plt.xlabel("Num of qubit")
plt.ylabel("Time (sec)")
plt.savefig("stat.png")
plt.show()
