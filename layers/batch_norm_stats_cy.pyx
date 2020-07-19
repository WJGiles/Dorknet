#cython: language_level=3
import numpy as np
cimport cython
cimport numpy as np
from cython.parallel import prange, parallel

"""
Note that cython infers when you need to use an omp reduction in parallel regions,
so the incrementing inside pranges in some of functions below is actually safe.
"""

ctypedef np.float32_t dtype_t


@cython.boundscheck(False)
@cython.wraparound(False)
def channelwise_mean_and_var_4d(dtype_t[:,:,:,:] A):
    cdef int b, c, iRow, iCol, i
    cdef int batch_size = A.shape[0]
    cdef int num_channels = A.shape[1]
    cdef int num_rows = A.shape[2]
    cdef int num_cols = A.shape[3]
    cdef dtype_t num_features = <dtype_t>(batch_size*num_rows*num_cols)
    cdef dtype_t[:] mean = np.zeros(num_channels, dtype=np.float32)
    cdef dtype_t[:] var = np.zeros(num_channels, dtype=np.float32)
                                          
    with nogil, parallel():
        for c in prange(num_channels, schedule='static'):
            for b in prange(batch_size, schedule='static'):
                for iRow in range(num_rows):
                    for iCol in range(num_cols):
                        mean[c] += A[b,c,iRow,iCol]

    for i in range(num_channels):
        mean[i] /= num_features

    with nogil, parallel():
        for c in prange(num_channels, schedule='static'):
            for b in prange(batch_size, schedule='static'):
                for iRow in range(num_rows):
                    for iCol in range(num_cols):
                        var[c] += (A[b,c,iRow,iCol] - mean[c])**2

    for i in range(num_channels):
        var[i] /= num_features

    return mean.base, var.base