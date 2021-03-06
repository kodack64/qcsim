
from ctypes import *
import numpy as np
import time
import os

path = "./x64/Release_DLL/"
dllNames = ["qcsim_c.dll","qcsim_omp.dll","qcsim_cuda.dll","qcsim_simd.dll","qcsim_simdomp.dll"]
maxn = 27


for dllName in dllNames:
	print(dllName.replace(".dll",""))
	if("posix" in os.name):
		dll = CDLL(dllName.replace("dll","so"))
	else:
		dll = cdll.LoadLibrary(path+dllName)
	fout = open("stat_{}.txt".format(dllName.replace(".dll","")),"w")
	for n in np.arange(2,maxn+1,1):
		dll.init(c_int(n))
		
		st = time.time()
		gc = 0
		while( (time.time()-st) < 1. ):
			for ind in range(n):
				dll.u(c_int(ind),c_double(1.57),c_double(0.2),c_double(0.1))
				gc+=1
		gpt1 = (time.time()-st)/gc

		st = time.time()
		gc=0
		while( (time.time()-st) < 1. ):
			for ind in range(n-1):
				dll.cx(c_int(ind),c_int(ind+1))
				gc+=1
		gpt2 = (time.time()-st)/gc

		fout.write("{} {} {}\n".format(n,gpt1,gpt2))
		print("{} {} {}".format(n,gpt1,gpt2))
		dll.release()
	fout.close()

