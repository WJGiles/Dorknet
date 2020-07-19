#cython: language_level=3
import numpy as np
cimport cython
cimport numpy as np
from cython.parallel import prange, parallel

ctypedef np.float32_t dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
def relu_4d_forward_train(dtype_t[:,:,:,:] X):
    cdef int b, c, iRow, iCol
    cdef int batch_size = X.shape[0]
    cdef int num_channels = X.shape[1]
    cdef int num_rows = X.shape[2]
    cdef int num_cols = X.shape[3]
    cdef dtype_t[:,:,:,:] out = np.zeros((batch_size,
                                          num_channels,
                                          num_rows,
                                          num_cols), dtype=np.float32)
    cdef dtype_t[:,:,:,:] pos_locs = np.zeros((batch_size,
                                          num_channels,
                                          num_rows,
                                          num_cols), dtype=np.float32)
                                          
    with nogil, parallel():
        for b in prange(batch_size, schedule='static'):
            for c in prange(num_channels, schedule='static'):
                for iRow in range(num_rows):
                    for iCol in range(num_cols):
                        if X[b,c,iRow,iCol] > 0.0:
                            out[b,c,iRow,iCol] = X[b,c,iRow,iCol]
                            pos_locs[b,c,iRow,iCol] = 1.0
                        else:
                            out[b,c,iRow,iCol] = 0.0
                            pos_locs[b,c,iRow,iCol] = 0.0

    return out.base, pos_locs.base

@cython.boundscheck(False)
@cython.wraparound(False)
def relu_4d_forward_test(dtype_t[:,:,:,:] X):
    cdef int b, c, iRow, iCol
    cdef int batch_size = X.shape[0]
    cdef int num_channels = X.shape[1]
    cdef int num_rows = X.shape[2]
    cdef int num_cols = X.shape[3]
    cdef dtype_t[:,:,:,:] out = np.zeros((batch_size,
                                          num_channels,
                                          num_rows,
                                          num_cols), dtype=np.float32)
                                          
    with nogil, parallel():
        for b in prange(batch_size, schedule='static'):
            for c in prange(num_channels, schedule='static'):
                for iRow in range(num_rows):
                    for iCol in range(num_cols):
                        if X[b,c,iRow,iCol] > 0.0:
                            out[b,c,iRow,iCol] = X[b,c,iRow,iCol]
                        else:
                            out[b,c,iRow,iCol] = 0.0

    return out.base

@cython.boundscheck(False)
@cython.wraparound(False)
def relu_2d_forward_train(dtype_t[:,:] X):
    cdef int b, c
    cdef int batch_size = X.shape[0]
    cdef int num_channels = X.shape[1]
    cdef dtype_t[:,:] out = np.zeros((batch_size,
                                          num_channels),
                                          dtype=np.float32)
    cdef dtype_t[:,:] pos_locs = np.zeros((batch_size,
                                               num_channels),
                                               dtype=np.float32)
                                          
    with nogil, parallel():
        for b in prange(batch_size, schedule='static'):
            for c in prange(num_channels, schedule='static'):
                        if X[b,c] > 0.0:
                            out[b,c] = X[b,c]
                            pos_locs[b,c] = 1.0
                        else:
                            out[b,c] = 0.0
                            pos_locs[b,c] = 0.0

    return out.base, pos_locs.base

@cython.boundscheck(False)
@cython.wraparound(False)
def relu_2d_forward_test(dtype_t[:,:] X):
    cdef int b, c
    cdef int batch_size = X.shape[0]
    cdef int num_channels = X.shape[1]
    cdef dtype_t[:,:] out = np.zeros((batch_size,
                                          num_channels),
                                          dtype=np.float32)
                                          
    with nogil, parallel():
        for b in prange(batch_size, schedule='static'):
            for c in prange(num_channels, schedule='static'):
                        if X[b,c] > 0.0:
                            out[b,c] = X[b,c]
                        else:
                            out[b,c] = 0.0

    return out.base