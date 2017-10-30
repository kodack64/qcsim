
    OPENQASM 2.0;
    include "qelib1.inc";
    include "qelib2.inc";
    qreg q1[3];
    h q1[0];
    cx q1[0],q1[2];
    zrot(pi/4) q1[0];
    xxrot(pi/4) q1[0],q1[2];
    gate test d0,d1{
        xxrot(pi/4) d0,d1;
        zrot(pi/4) d1;
    }
    test q1[0],q1[1];
    m0 q1[0];
    