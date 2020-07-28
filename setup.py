from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension('im2col', ['layers/im2col.pyx'],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
        extra_link_args=['-fopenmp']),
    Extension('pooling_cy', ['layers/pooling_cy.pyx'],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
        extra_link_args=['-fopenmp']),
    Extension('relu_cy', ['layers/relu_cy.pyx'],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
        extra_link_args=['-fopenmp']),
    Extension('batch_norm_stats_cy', ['layers/batch_norm_stats_cy.pyx'],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
        extra_link_args=['-fopenmp'])
]

setup(
    ext_modules = cythonize(extensions),
)