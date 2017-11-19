
OPENQASM 2.0;
include "qelib1.inc";
include "qelib2.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[2];
zrot(pi/4) q[0];
xxrot(pi/4) q[0],q[2];
gate test d0,d1{
    xxrot(pi/4) d0,d1;
    zrot(pi/4) d1;
}
test q[0],q[1];
if(c == 3) test q[0],q[1];
measure q -> c;
