
OPENQASM 2.0;
include "qelib1.inc";
include "qelib2.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
rz(pi/4) q[0];
x q[0];
rx(pi/16) q[1];
cz q[0],q[1];
t q[1];
//xxrot(pi/4) q[0],q[2];
//gate test d0,d1{
//    xxrot(pi/4) d0,d1;
//    zrot(pi/4) d1;
//}
//test q[0],q[1];
measure q -> c;
