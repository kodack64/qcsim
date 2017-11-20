
"""
list of kernel function for gate operation
"""

import cupy as cp

class KernelList:
    ker_X = cp.ElementwiseKernel(
    in_params = "raw T x, int32 k",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    """,
    operation = """
    y = x[i^mask];
    """,
    name = "X",
    )


    ker_Y = cp.ElementwiseKernel(
    in_params = "raw T x, int32 k",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    thrust::complex<double> img = thrust::complex<double>(0,1);
    """,
    operation = """
    if(i&mask) y = -img*x[i^mask];
    else y = img*x[i^mask];
    """,
    name = "Y",
    )


    ker_Z = cp.ElementwiseKernel(
    in_params = "T x, int32 k",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    """,
    operation = """
    if(i&mask) y = -x;
    else y = x;
    """,
    name = "Z",
    )


    ker_H = cp.ElementwiseKernel(
    in_params = "raw T x, int32 k",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    thrust::complex<double> sq2 = thrust::complex<double>(1/sqrt(2.),0);
    """,
    operation = """
    if(i&mask) y = (-x[i] + x[i^mask])*sq2;
    else y = (x[i] + x[i^mask])*sq2;
    """,
    name = "H",
    )

    ker_S = cp.ElementwiseKernel(
    in_params = "T x, int32 k",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    thrust::complex<double> img = thrust::complex<double>(0,1);
    """,
    operation = """
    if(i&mask) y = img*x;
    else y = x;
    """,
    name = "S",
    )

    ker_T = cp.ElementwiseKernel(
    in_params = "S x, int32 k",
    out_params = "S y",
    loop_prep = """
    int mask = 1<<k;
    thrust::complex<double> ph = thrust::complex<double>(1/sqrt(2.),1/sqrt(2.));
    """,
    operation = """
    if(i&mask) y = ph*x;
    else y = x;
    """,
    name = "T",
    )

    ker_CX = cp.ElementwiseKernel(
    in_params = "raw T x, int32 k, int32 u",
    out_params = "T y",
    loop_prep = """
    int mask_k = 1<<k;
    int mask_u = 1<<u;
    """,
    operation = """
    if(i&mask_k) y = x[i^mask_u];
    else y = x[i];
    """,
    name = "CX",
    )

    ker_CZ = cp.ElementwiseKernel(
    in_params = "T x, int32 k, int32 u",
    out_params = "T y",
    loop_prep = """
    int mask_k = 1<<k;
    int mask_u = 1<<u;
    """,
    operation = """
    if((i&mask_k) && (i&mask_u)) y = -x;
    else y = x;
    """,
    name = "CZ",
    )

    ker_Toffoli = cp.ElementwiseKernel(
    in_params = "raw T x, int32 k, int32 u, int32 t",
    out_params = "T y",
    loop_prep = """
    int mask_k = 1<<k;
    int mask_u = 1<<u;
    int mask_t = 1<<t;
    """,
    operation = """
    if((i&mask_k) && (i&mask_u)) y = x[i^mask_t];
    else y = x[i];
    """,
    name = "Toffoli",
    )

    ker_Xrot = cp.ElementwiseKernel(
    in_params = "raw T x, int32 k, float64 theta",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    thrust::complex<double> c = thrust::complex<double>(cos(theta),0);
    thrust::complex<double> s = thrust::complex<double>(0,sin(theta));
    """,
    operation = """
    y = c*x[i]+s*x[i^mask];
    """,
    name = "Xrot",
    )

    ker_Yrot = cp.ElementwiseKernel(
    in_params = "raw T x, int32 k, float64 theta",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    thrust::complex<double> c = thrust::complex<double>(cos(theta),0);
    thrust::complex<double> s = thrust::complex<double>(sin(theta),0);
    """,
    operation = """
    if(i&mask) y = c*x[i]-s*x[i^mask];
    else y = c*x[i]+s*x[i^mask];
    """,
    name = "Yrot",
    )

    ker_Zrot = cp.ElementwiseKernel(
    in_params = "T x, int32 k, float64 theta",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    thrust::complex<double> c1 = thrust::complex<double>(cos(theta),sin(theta));
    thrust::complex<double> c2 = thrust::complex<double>(cos(theta),-sin(theta));
    """,
    operation = """
    if(i&mask) y = c2*x;
    else y = c1*x;
    """,
    name = "Zrot",
    )

    ker_XXrot = cp.ElementwiseKernel(
    in_params = "raw T x, int32 k, int32 u,float64 theta",
    out_params = "T y",
    loop_prep = """
    int mask_ku = (1<<k)+(1<<u);
    thrust::complex<double> c = thrust::complex<double>(cos(theta),0);
    thrust::complex<double> s = thrust::complex<double>(0,sin(theta));
    """,
    operation = """
    y = c*x[i] + s*x[i^mask_ku];
    """,
    name = "XXrot",
    )



    ker_MeasZ0 = cp.ElementwiseKernel(
    in_params = "T x, int32 k",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    """,
    operation = """
    if(i&mask) y = 0;
    else y = x;
    """,
    name = "MeasZ0",
    )

    ker_MeasZ1 = cp.ElementwiseKernel(
    in_params = "T x, int32 k",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    """,
    operation = """
    if(i&mask) y = x;
    else y = 0;
    """,
    name = "MeasZ1",
    )

    ker_U = cp.ElementwiseKernel(
    in_params = "raw T x, int32 k, float64 t0, float64 t1, float64 t2",
    out_params = "T y",
    loop_prep = """
    int mask = 1<<k;
    double t12p = (t1+t2)/2;
    double t12m = (t1-t2)/2;
    thrust::complex<double> u00 = thrust::complex<double>(cos(t12p),sin(-t12p)) * cos(t0/2);
    thrust::complex<double> u01 = thrust::complex<double>(-cos(t12m),-sin(-t12m)) * sin(t0/2);
    thrust::complex<double> u10 = thrust::complex<double>(cos(t12m),sin(t12m)) * sin(t0/2);
    thrust::complex<double> u11 = thrust::complex<double>(cos(t12p),sin(t12p)) * cos(t0/2);
    """,
    operation = """
    if(i&mask) y = u10*x[i^mask] + u11*x[i];
    else y = u00*x[i] + u01*x[i^mask];
    """,
    name = "U",
    )

    ker_trace = cp.ReductionKernel(
    "T x",
    "T y",
    "x*thrust::conj(x)",
    "a+b",
    "y = a",
    "0",
    "trace"
    )

    onePauli = [ker_X,ker_Y,ker_Z]
    oneClifford = onePauli + [ker_H,ker_S]
    oneGate = oneClifford + [ker_T]

    twoPauli = []
    twoClifford = [ker_CX,ker_CZ]
    twoGate = twoPauli + twoClifford + []

    threePauli = []
    threeClifford = []
    threeGate = [ker_Toffoli]

    pauli = onePauli + twoPauli + threePauli
    clifford = oneClifford + twoClifford
    discreteGate = oneGate + twoGate + threeGate

    oneRot = [ker_Xrot,ker_Yrot,ker_Zrot]
    twoRot = [ker_XXrot]
    matchgate = [ker_Zrot,ker_XXrot]
    continuusGate = oneRot+twoRot

    genericGate = [ker_U]

    measurement = [ker_MeasZ0,ker_MeasZ1]

    allGate = discreteGate + continuusGate + measurement + genericGate
    allGateName = [g.name for g in allGate]

    # require target
    oneDiscrete = oneGate
    # require control, target
    twoDiscrete = twoGate
