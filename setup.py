from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

if os.name=="posix":

    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("qcsim.cython._cythonsim", ["./qcsim/cython/_cythonsim.pyx"],language='c')],
        include_dirs = [numpy.get_include()]
    )
    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("qcsim.cythonomp._cythonompsim", ["./qcsim/cythonomp/_cythonompsim.pyx"],extra_compile_args=['-fopenmp'],extra_link_args=['-lgomp'],language='c')],
        include_dirs = [numpy.get_include()]
    )

elif os.name=="nt":
    setup(
    cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension(
            "qcsim.cython._cythonsim",
            ["./qcsim/cython/_cythonsim.pyx"],
            extra_compile_args=['/Ot', '/favor:INTEL64', '/EHsc', '/GA'],
            language='c'
         )],
        include_dirs = [numpy.get_include()]
    )
    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension(
            "qcsim.cythonomp._cythonompsim",
            ["./qcsim/cythonomp/_cythonompsim.pyx"],
            extra_compile_args=['/Ot', '/favor:INTEL64', '/EHsc', '/GA', '/openmp'],
            language='c'
        )],
        include_dirs = [numpy.get_include()]
    )

else:
    raise Exception("Unsupported environement")
