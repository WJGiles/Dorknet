import numpy as np
cimport cython
cimport numpy as np
from cython.parallel import prange, parallel

ctypedef np.float32_t dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
def pool(dtype_t[:,:,:,:] X, int stride):
    cdef Py_ssize_t i, j, k, l, m, n, p, q
    cdef dtype_t max_val
    cdef dtype_t[:,:,:,:] pooled_out = np.zeros((X.shape[0],
                                                 X.shape[1],
                                                 int(X.shape[2]/stride),
                                                 int(X.shape[3]/stride)),
                                                 dtype=np.float32)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
                for k in range(0, X.shape[2], stride): # row
                    for l in range(0, X.shape[3], stride): # col
                        max_val = X[i, j, k, l]
                        # Pooling region
                        for m in range(stride):
                            for n in range(stride):
                                if X[i, j, k + m, l + n] > max_val:
                                    max_val = X[i, j, k + m, l + n]
                        p = int(k/stride)
                        q = int(l/stride)
                        pooled_out[i, j, p, q] = max_val
    return pooled_out.base

@cython.boundscheck(False)
@cython.wraparound(False)
def pool_train(dtype_t[:,:,:,:] X, int stride):
    cdef Py_ssize_t i, j, k, l, m, n, p, q, r, s
    cdef dtype_t max_val
    cdef dtype_t[:,:,:,:] pooled_out = np.zeros((X.shape[0],
                                                 X.shape[1],
                                                 int(X.shape[2]/stride),
                                                 int(X.shape[3]/stride)),
                                                 dtype=np.float32)
    cdef int[:,:,:,:] max_locs = np.zeros((X.shape[0],
                                           X.shape[1],
                                           X.shape[2],
                                           X.shape[3]),
                                           dtype=np.int32)

    for i in range(X.shape[0]):# batch
        for j in range(X.shape[1]): # channel
                for k in range(0, X.shape[2], stride): # row
                    for l in range(0, X.shape[3], stride): # col
                        max_val = X[i, j, k, l]
                        r = 0
                        s = 0
                        # Pooling region
                        for m in range(stride):
                            for n in range(stride):
                                if X[i, j, k + m, l + n] > max_val:
                                    max_val = X[i, j, k + m, l + n]
                                    r = m
                                    s = n
                        p = <Py_ssize_t>(k/stride)
                        q = <Py_ssize_t>(l/stride)
                        pooled_out[i, j, p, q] = max_val
                        max_locs[i, j, k + r, l + s] = 1
    return pooled_out.base, max_locs.base

@cython.boundscheck(False)
@cython.wraparound(False)
def pool_backward(int[:,:,:,:] max_locs, dtype_t[:,:,:,:] upstream_dx, int stride):
    cdef Py_ssize_t i, j, k, l, m, n
    cdef dtype_t[:,:,:,:] out = np.zeros((max_locs.shape[0],
                                          max_locs.shape[1],
                                          max_locs.shape[2],
                                          max_locs.shape[3]),
                                          dtype=np.float32)
    for i in range(max_locs.shape[0]):
        for j in range(max_locs.shape[1]):
            for k in range(max_locs.shape[2]):
                for l in range(max_locs.shape[3]):
                    if max_locs[i, j, k, l] == 1:
                        m = <Py_ssize_t>(k/stride)
                        n = <Py_ssize_t>(l/stride)
                        out[i, j, k, l] = upstream_dx[i, j, m, n]

    return out.base