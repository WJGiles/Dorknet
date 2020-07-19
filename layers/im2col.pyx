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
def im2col_cy(dtype_t[:,:,:,:] X, int f_rows, int f_cols, int stride):
    cdef Py_ssize_t i, j, k, l, m, n
    cdef dtype_t full_num_row_patches = ((X.shape[2] - f_rows)/stride) + 1
    cdef dtype_t full_num_col_patches = ((X.shape[3] - f_cols)/stride) + 1
    cdef Py_ssize_t num_row_patches = <Py_ssize_t>(full_num_row_patches)
    cdef Py_ssize_t num_col_patches = <Py_ssize_t>(full_num_col_patches)
    cdef dtype_t[:,:] f_patches = np.empty((X.shape[0]*num_row_patches*num_col_patches,
                                            X.shape[1]*f_rows*f_cols),
                                           dtype=np.float32)

    with nogil, parallel():
        for i in prange(X.shape[0], schedule='static'):
            for j in prange(0, num_row_patches, schedule='static'):  #row_patch_ix:
                for k in prange(0, num_col_patches, schedule='static'):  #col_patch_ix:
                    for l in range(X.shape[1]): # channels
                        for m in range(f_rows):
                            for n in range(f_cols):
                                f_patches[num_col_patches*(i*num_row_patches + j) + k,
                                          f_cols*(l*f_rows + m) + n] = X[i, l, j*stride + m, k*stride + n]

    return f_patches.base, full_num_row_patches, full_num_col_patches

@cython.boundscheck(True)
@cython.wraparound(True)
def depthwise_im2col_cy(dtype_t[:,:,:,:] X, int f_rows, int f_cols, int stride):
    """
    X has shape (batch_size, channels, height, width)
    """
    print("depthwise_im2col_cy")
    cdef Py_ssize_t c, b, iRow, iCol, iFrow, iFcol
    #row_patch_ix = range(0, X.shape[2] - f_rows + 1, stride)
    #col_patch_ix = range(0, X.shape[3] - f_cols + 1, stride)
    cdef int batch_size = X.shape[0]
    cdef int num_channels = X.shape[1]
    cdef int num_rows = X.shape[2]
    cdef int num_cols = X.shape[3]
    cdef int num_row_patches = <Py_ssize_t>len(range(0, X.shape[2] - f_rows + 1, stride))
    cdef int num_col_patches = <Py_ssize_t>len(range(0, X.shape[3] - f_cols + 1, stride))
    cdef dtype_t[:,:,:] out = np.empty((num_channels,
                                        batch_size*num_row_patches*num_col_patches,
                                        f_rows*f_cols), dtype=np.float32)
    with nogil, parallel():
        for c in prange(num_channels, schedule='static'):
            for b in prange(batch_size, schedule='static'):
                for iRow in range(num_row_patches):
                    for iCol in range(num_col_patches):
                        for iFrow in range(f_rows):
                            for iFcol in range(f_cols):
                                out[c,
                                    num_col_patches*(num_row_patches*b + iRow) + iCol,
                                    f_cols*iFrow + iFcol] = (
                                        X[b, c, stride*iRow+iFrow, stride*iCol+iFcol]
                                    )

    return out.base, num_row_patches, num_col_patches

@cython.boundscheck(False)
@cython.wraparound(False)
def depthwise_im2col_einsum_cy(dtype_t[:,:,:,:] X, int f_rows, int f_cols, int stride):
    """
    X has shape (batch_size, channels, height, width)
    """
    cdef Py_ssize_t b, c, iRow, iCol, iFrow, iFcol
    #row_patch_ix = range(0, X.shape[2] - f_rows + 1, stride)
    #col_patch_ix = range(0, X.shape[3] - f_cols + 1, stride)
    cdef int batch_size = X.shape[0]
    cdef int num_channels = X.shape[1]
    cdef int num_rows = X.shape[2]
    cdef int num_cols = X.shape[3]
    cdef int num_row_patches = <Py_ssize_t>len(range(0, X.shape[2] - f_rows + 1, stride))
    cdef int num_col_patches = <Py_ssize_t>len(range(0, X.shape[3] - f_cols + 1, stride))
    cdef dtype_t[:,:,:,:] out = np.empty((batch_size,
                                        num_channels,
                                        num_row_patches*num_col_patches,
                                        f_rows*f_cols), dtype=np.float32)
    with nogil, parallel():
        for b in prange(batch_size, schedule='static'):
            for c in prange(num_channels, schedule='static'):
                for iRow in range(num_row_patches):
                    for iCol in range(num_col_patches):
                        for iFrow in range(f_rows):
                            for iFcol in range(f_cols):
                                out[b,
                                    c,
                                    num_col_patches*iRow + iCol,
                                    f_cols*iFrow + iFcol] = (
                                        X[b, c, stride*iRow+iFrow, stride*iCol+iFcol]
                                    )

    return out.base, num_row_patches, num_col_patches

@cython.boundscheck(False)
@cython.wraparound(False)
def depthwise_conv_cy(dtype_t[:,:,:,:] X, dtype_t[:,:,:] f, int f_rows, int f_cols, int stride):
    """
    X has shape (batch_size, channels, height, width)
    f has shape = (channels, f_rows, f_cols)
    """
    cdef Py_ssize_t b, c, iRow, iCol, iFrow, iFcol
    cdef int batch_size = X.shape[0]
    cdef int num_channels = X.shape[1]
    cdef int num_rows = X.shape[2]
    cdef int num_cols = X.shape[3]
    
    cdef dtype_t full_num_row_patches = ((X.shape[2] - f_rows)/stride) + 1
    cdef dtype_t full_num_col_patches = ((X.shape[3] - f_cols)/stride) + 1
    cdef Py_ssize_t num_row_patches = <Py_ssize_t>(full_num_row_patches)
    cdef Py_ssize_t num_col_patches = <Py_ssize_t>(full_num_col_patches)


    cdef dtype_t[:,:,:,:] out = np.zeros((batch_size,
                                        num_channels,
                                        num_row_patches,
                                        num_col_patches), dtype=np.float32)
    with nogil, parallel():
        for b in prange(batch_size, schedule='static'):
            for c in prange(num_channels, schedule='static'):
                for iRow in range(num_row_patches):
                    for iCol in range(num_col_patches):
                        for iFrow in range(f_rows):
                            for iFcol in range(f_cols):
                                out[b, c, iRow, iCol] += X[b, c, stride*iRow+iFrow, stride*iCol+iFcol]*f[c, iFrow, iFcol]

    return out.base, full_num_row_patches, full_num_col_patches

@cython.boundscheck(False)
@cython.wraparound(False)
def depthwise_backward_direct_cy(dtype_t[:,:,:,:] upstream_dx, dtype_t[:,:,:,:] X, dtype_t[:,:,:] w,
                                 dtype_t full_num_row_patches, dtype_t full_num_col_patches, int stride, int padding):
    cdef Py_ssize_t iBatch, iChan, iRow, iCol, iFrow, iFcol
    cdef int batch_size = X.shape[0]
    cdef int num_channels = X.shape[1]
    cdef int f_rows = w.shape[1]
    cdef int f_cols = w.shape[2]
    cdef int num_padded_rows = <Py_ssize_t>(stride*(full_num_row_patches - 1) + f_rows)
    cdef int num_padded_cols = <Py_ssize_t>(stride*(full_num_col_patches - 1) + f_cols)
    cdef Py_ssize_t num_row_patches = <Py_ssize_t>(full_num_row_patches)
    cdef Py_ssize_t num_col_patches = <Py_ssize_t>(full_num_col_patches)
    cdef dtype_t[:,:,:,:] padded_dx = np.zeros((batch_size, num_channels,
                                                num_padded_rows, num_padded_cols),
                                                dtype=np.float32)
    cdef dtype_t[:,:,:,:] dw = np.zeros((batch_size, w.shape[0], w.shape[1], w.shape[2]), dtype=np.float32)

    with nogil, parallel():
        for iBatch in prange(batch_size):
            for iChan in prange(num_channels):
                for iRow in range(num_row_patches):
                    for iCol in range(num_col_patches):
                        for iFrow in range(f_rows):
                            for iFcol in range(f_cols):
                                dw[iBatch, iChan, iFrow, iFcol] += (
                                    upstream_dx[iBatch, iChan, iRow, iCol]*X[iBatch, iChan,
                                                                            stride*iRow + iFrow,
                                                                            stride*iCol + iFcol]
                                )
                                padded_dx[iBatch, iChan, stride*iRow + iFrow, stride*iCol + iFcol] += (
                                    upstream_dx[iBatch, iChan, iRow, iCol]*w[iChan, iFrow, iFcol]
                                )
    if padding > 0:
        dx = padded_dx[:,:,padding:-padding,padding:-padding].copy()
        return dx.base, dw.base
    else:
        return padded_dx.base, dw.base

@cython.boundscheck(False)
@cython.wraparound(False)
def depthwise_row2im_einsum_cy(dtype_t[:,:,:,:] X, int batch_size, int num_row_patches, int num_col_patches,
                            int f_rows, int f_cols, int num_channels, int stride, int padding):
    # X = (chans, batch_size*num_row_patches*num_col_patches, flattened_patches)
    # out = (batch, chans, row, col)
    cdef Py_ssize_t b, c, iRow, iCol, iFrow, iFcol
    cdef int num_padded_rows = stride*(num_row_patches - 1) + f_rows
    cdef int num_padded_cols = stride*(num_col_patches - 1) + f_cols
    cdef dtype_t[:,:,:,:] padded_out = np.zeros((batch_size, num_channels, num_padded_rows, num_padded_cols),
                                                 dtype=np.float32)
    with nogil, parallel():
        for b in prange(batch_size, schedule='static'):
            for c in prange(num_channels, schedule='static'):
                for iRow in prange(num_row_patches, schedule='static'):
                    for iCol in range(num_col_patches):
                        for iFrow in range(f_rows):
                            for iFcol in range(f_cols):
                                padded_out[b, c, stride*iRow + iFrow, stride*iCol + iFcol] += ( 
                                    X[b, c, num_col_patches*iRow + iCol, f_cols*iFrow + iFcol]
                                )
    if padding > 0:
        unpadded_out = padded_out[:,:,padding:-padding,padding:-padding].copy()
        return unpadded_out.base
    else:
        return padded_out.base

@cython.boundscheck(False)
@cython.wraparound(False)
def row2im_cy(dtype_t[:,:] X, int batch_size, dtype_t full_num_row_patches, dtype_t full_num_col_patches,
              int f_rows, int f_cols, int num_channels, int stride, int padding):
    cdef Py_ssize_t iBatch, iRow, iCol, iChan, iFrow, iFcol
    cdef int num_padded_rows = <Py_ssize_t>(stride*(full_num_row_patches - 1) + f_rows)
    cdef int num_padded_cols = <Py_ssize_t>(stride*(full_num_col_patches - 1) + f_cols)
    cdef Py_ssize_t num_row_patches = <Py_ssize_t>(full_num_row_patches)
    cdef Py_ssize_t num_col_patches = <Py_ssize_t>(full_num_col_patches)
    cdef dtype_t[:,:,:,:] padded_out = np.zeros((batch_size, num_channels, num_padded_rows, num_padded_cols),
                                            dtype=np.float32)
    cdef dtype_t[:,:,:,:] unpadded_out

    with nogil, parallel():
        for iBatch in prange(batch_size, schedule='static'):
            for iRow in prange(num_row_patches, schedule='static'):
                for iCol in prange(num_col_patches, schedule='static'):
                    for iChan in range(num_channels):
                        for iFrow in range(f_rows):
                            for iFcol in range(f_cols):
                                padded_out[iBatch, iChan, stride*iRow + iFrow, stride*iCol + iFcol] += (
                                    X[num_col_patches*(num_row_patches*iBatch + iRow) + iCol, f_cols*(f_rows*iChan + iFrow) + iFcol]
                                )
    if padding > 0:
        unpadded_out = padded_out[:,:,padding:-padding,padding:-padding].copy()
        return unpadded_out.base
    else:
        return padded_out.base

@cython.boundscheck(True)
@cython.wraparound(True)
def depthwise_row2im_cy(dtype_t[:,:,:] X, int batch_size, int num_row_patches, int num_col_patches,
                        int f_rows, int f_cols, int num_channels, int stride, int padding):
    print("depthwise_row2im_cy")
    cdef Py_ssize_t iChan, iBatch, iRow, iCol, iFrow, iFcol
    cdef int num_padded_rows = stride*(num_row_patches - 1) + f_rows
    cdef int num_padded_cols = stride*(num_col_patches - 1) + f_cols
    cdef dtype_t[:,:,:,:] padded_out = np.zeros((batch_size, num_channels,
                                                 num_padded_rows, num_padded_cols),
                                                 dtype=np.float32)
    cdef dtype_t[:,:,:,:] unpadded_out
    with nogil, parallel():
        for iChan in prange(num_channels, schedule='static'):
            for iBatch in prange(batch_size, schedule='static'):
                for iRow in prange(num_row_patches, schedule='static'):
                    for iCol in prange(num_col_patches, schedule='static'):
                        for iFrow in range(f_rows):
                            for iFcol in range(f_cols):
                                padded_out[iBatch, iChan, iRow + iFrow, iCol + iFcol] += ( 
                                    X[iChan, num_col_patches*(num_row_patches*iBatch + iRow) + iCol, iFrow*f_cols + iFcol]
                                )
    if padding > 0:
        unpadded_out = padded_out[:,:,padding:-padding,padding:-padding].copy()
        return unpadded_out.base
    else:
        return padded_out.base