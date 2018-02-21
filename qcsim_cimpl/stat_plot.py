
import glob
import numpy as np
import matplotlib.pyplot as plt

flist = glob.glob("stat_qcsim_*.txt")
flist.sort()

alldat = {}
keys = []
for fname in flist:
	fin = open(fname)
	dataname = fname.replace(".txt","").split("_")[-1]
	dat = []
	for line in fin:
		elem = line.split(" ")
		n = int(elem[0])
		t_u = float(elem[1])
		t_cx = float(elem[2])
		dat.append([n,t_u,t_cx])
	dat = np.array(dat).T
	alldat[dataname]=dat
	keys.append(dataname)
keys.sort()

plt.subplot(1,2,1)
for key in keys:
	dat = alldat[key]
	plt.plot(dat[0],dat[1],label=key)
plt.legend()
plt.title("single qubit unitary")
plt.xlabel("#qubit")
plt.ylabel("time (sec)")
plt.ylim(1e-6,1e1)
plt.yscale("log")

plt.subplot(1,2,2)
for key in keys:
	dat = alldat[key]
	plt.plot(dat[0],dat[2],label=key)
plt.legend()
plt.title("CNOT")
plt.xlabel("#qubit")
plt.ylabel("time (sec)")
plt.ylim(1e-6,1e1)
plt.yscale("log")

plt.savefig("stat.png")
plt.show()
