import numpy as np
import cupy as cp
import im2col
import sys
import itertools, time
#import line_profiler
from regularisers import l2

profile = lambda x: x

class ConvLayer:
    def __init__(self, layer_name, filter_block_shape=None, stride=1, padding=1,
                 with_bias=True, weight_regulariser=None, weight_initialiser="normal"):
        self.is_on_gpu = False
        self.layer_name = layer_name
        self.stride = stride
        self.padding = padding
        self.patches = None
        self.weight_regulariser = weight_regulariser
        self.weight_initialiser = weight_initialiser
        if filter_block_shape:
            self.num_filters, self.filter_chans, self.f_rows, self.f_cols = filter_block_shape
            if self.weight_initialiser == "glorot_uniform":
                limit = np.sqrt(6.0 / (self.filter_chans + self.num_filters))
                weights = np.random.uniform(low=-limit,high=limit,size=filter_block_shape).astype(np.float32)
            elif self.weight_initialiser == "normal":
                weights = 0.01*np.random.randn(*filter_block_shape).astype(np.float32)
            self.learned_params = {"weights": weights}
            self.grads = {"weights": np.zeros_like(weights)}
            self.with_bias = with_bias
            if with_bias:
                bias = np.zeros(self.num_filters).astype(np.float32)
                self.learned_params.update({"bias": bias})
                self.grads.update({"bias": np.zeros_like(bias)})
        else:
            self.num_filters = None
            self.learned_params = {}
            self.grads = {}

    def __repr__(self):
        out = "ConvLayer({}, ".format(self.layer_name)
        if self.num_filters is not None:
            out += "filter_block_shape=({},{},{},{}), ".format(self.num_filters,
                                                               self.filter_chans,
                                                               self.f_rows,
                                                               self.f_rows)
        out += "stride={}, padding={}, with_bias={}, weight_regulariser={})".format(self.stride,
                                                             self.padding,
                                                             self.with_bias,
                                                             self.weight_regulariser)
        return out

    def to_gpu(self):
        if self.is_on_gpu:
            print("Layer already on GPU, ignoring request")
        else:
            self.im2col_kernel, self.row2im_kernel = self.get_kernels()
            # move learned_params and grads to gpu
            for k, v in self.learned_params.items():
                self.learned_params[k] = cp.asarray(self.learned_params[k])
            for k, v  in self.grads.items():
                self.grads[k] = cp.asarray(self.learned_params[k])
            self.is_on_gpu = True

    @profile
    def forward(self, X, test_mode=False):

        self.input_shape = X.shape
        if self.padding > 0:
            X = self.pad_input(X)

        flat_filter = self.learned_params["weights"].reshape((-1, self.f_rows*self.f_cols*self.filter_chans))
        
        if self.is_on_gpu:
            self.num_row_patches = ((X.shape[2] - self.f_rows)/self.stride) + 1
            self.num_col_patches = ((X.shape[3] - self.f_cols)/self.stride) + 1
            self.patches = cp.zeros((X.shape[0]*int(self.num_row_patches)*int(self.num_col_patches),
                            self.f_rows*self.f_cols*self.filter_chans)).astype(cp.float32)
            numThreadBlocks = int((X.shape[0]*X.shape[1]*self.num_row_patches*self.num_col_patches + 1024 - 1) / 1024)
            self.im2col_kernel((numThreadBlocks,), (1024,), (X, self.patches, self.f_rows, self.f_cols,
                                                   X.shape[0], self.filter_chans, X.shape[-2], X.shape[-1], 
                                                   int(self.num_row_patches), int(self.num_col_patches), self.stride))
            out = cp.dot(cp.asarray(self.patches), cp.asarray(flat_filter.T))
        else:
            self.patches, self.num_row_patches, self.num_col_patches = im2col.im2col_cy(
                                                         X,
                                                         self.f_rows,
                                                         self.f_cols,
                                                         self.stride
                                                         )
            out = cp.dot(self.patches, flat_filter.T) 
        if self.with_bias:
            out += self.learned_params['bias'].reshape(1,-1)
        out = out.reshape((X.shape[0],int(self.num_row_patches),int(self.num_col_patches),self.num_filters)).transpose(0,3,1,2)
        return out

    @profile
    def backward(self, upstream_dx):
        if self.with_bias:
            self.grads["bias"] = np.sum(upstream_dx, axis=(0,2,3))
        upstream_dx = upstream_dx.transpose(0,2,3,1) # Undo the transpose from forward pass
        upstream = upstream_dx.reshape(self.patches.shape[0], -1)
        if self.is_on_gpu:
            self.grads["weights"] = cp.dot(upstream.T, self.patches).reshape(self.learned_params["weights"].shape)
        else:
            self.grads["weights"] = np.dot(upstream.T, self.patches).reshape(self.learned_params["weights"].shape)
        if self.weight_regulariser:
            self.grads["weights"] += self.weight_regulariser.backward(self.learned_params["weights"])
        upstream = upstream_dx.reshape(-1, self.num_filters)
        flat_filter = self.learned_params["weights"].reshape((-1, self.f_rows*self.f_cols*self.filter_chans))
        if self.is_on_gpu:
            dx_rows = cp.dot(upstream, flat_filter)
            num_padded_rows = int(self.stride*(self.num_row_patches - 1) + self.f_rows)
            num_padded_cols = int(self.stride*(self.num_col_patches - 1) + self.f_cols)
            padded_dx = cp.zeros((self.input_shape[0], self.input_shape[1], num_padded_rows, num_padded_cols), dtype=cp.float32)
            numThreadBlocks = int((padded_dx.shape[0]*padded_dx.shape[1]*num_padded_rows*num_padded_cols + 1024 - 1) / 1024)
            self.row2im_kernel((numThreadBlocks,), (1024,), (padded_dx, dx_rows, self.f_rows, self.f_cols, self.input_shape[0], self.input_shape[1], 
                                                   num_padded_rows, num_padded_cols, int(self.num_row_patches), int(self.num_col_patches),
                                                   self.stride))
            # unpad
            if self.padding > 0:
                dx = padded_dx[:,:,self.padding:-self.padding,self.padding:-self.padding]
                return dx
            else:
                return padded_dx

        else:
            dx_rows = np.dot(upstream, flat_filter)
            dx = im2col.row2im_cy(dx_rows, self.input_shape[0],
                                self.num_row_patches, self.num_col_patches, 
                                self.f_rows, self.f_cols, self.input_shape[1], 
                                self.stride, self.padding)
            dx = np.array(dx)
            return dx

    def row2im(self, X, batch_size, num_row_patches, num_col_patches, f_rows, f_cols, num_channels, stride, padding):
        num_padded_rows = stride*(num_row_patches - 1) + f_rows
        num_padded_cols = stride*(num_col_patches - 1) + f_cols
        padded_out = np.zeros((batch_size, num_channels, num_padded_rows, num_padded_cols))
        for iBatch in range(batch_size):
            for iRow in range(num_row_patches):
                for iCol in range(num_col_patches):
                    for iChan in range(num_channels):
                        for iFrow in range(f_rows):
                            for iFcol in range(f_cols):
                                padded_out[iBatch, iChan, stride*iRow + iFrow, stride*iCol + iFcol] += (
                                    X[num_col_patches*(num_row_patches*iBatch + iRow) + iCol, f_cols*(f_rows*iChan + iFrow) + iFcol]
                                )
        unpadded_out = padded_out[:,:,padding:-padding,padding:-padding]
        return unpadded_out

    def pad_input(self, X):
        xp = cp.get_array_module(X)
        return xp.pad(
                X, ((0, 0), (0, 0), 
                    (self.padding, self.padding),
                    (self.padding, self.padding)), 
                    "constant"
                )

    @classmethod
    def im2col_stack(cls, X, f_rows, f_cols, stride):
        """
        X has shape (batch_size, channels, height, width)
        """
        row_patch_ix = range(0, X.shape[2] - f_rows + 1, stride)
        col_patch_ix = range(0, X.shape[3] - f_cols + 1, stride)
        f_patches = []
        for b, i, j in itertools.product(range(X.shape[0]),
                                         row_patch_ix,
                                         col_patch_ix):
            f_patches.append(X[b, :, i:i+f_rows, j:j+f_cols].flatten())
        patches = np.stack(f_patches)

        return patches, len(row_patch_ix), len(col_patch_ix)

    #@profile
    def im2col(self, X, f_rows, f_cols, stride):
        """
        X has shape (batch_size, channels, height, width)
        """
        row_patch_ix = range(0, X.shape[2] - f_rows + 1, stride)
        col_patch_ix = range(0, X.shape[3] - f_cols + 1, stride)
        num_row_patches = len(row_patch_ix)
        num_col_patches = len(col_patch_ix)
        patches = np.empty((X.shape[0]*num_row_patches*num_col_patches,X.shape[1]*f_rows*f_cols))

        for i, j in itertools.product(row_patch_ix, col_patch_ix):
            indices = np.arange(i*num_row_patches + j, patches.shape[0], num_row_patches*num_col_patches)
            patches[indices, :] = X[:, :, i:i+f_rows, j:j+f_cols].reshape(X.shape[0],X.shape[1]*f_rows*f_cols)

        return patches, num_row_patches, num_col_patches

    def get_kernels(self):
        im2col_kernel = cp.RawKernel(
        """ extern "C" __global__ void im2col(float* X, float* Z, int f_rows, int f_cols,
                                            int batch_size, int num_channels, int X_num_rows, int X_num_cols,
                                            int num_row_patches, int num_col_patches, int stride) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid < batch_size*num_channels*num_row_patches*num_col_patches) {
                    int x_chan = (tid/(num_row_patches*num_col_patches));
                    int z_batch = int(tid/(num_channels*num_row_patches*num_col_patches))*num_channels*num_row_patches*num_col_patches;
                    int z_batch_index = tid%(num_channels*num_row_patches*num_col_patches);
                    int z_batch_col = int(z_batch_index/(num_row_patches*num_col_patches));
                    int z_batch_row = (z_batch_index%(num_row_patches*num_col_patches))*num_channels;
                    int x_start = x_chan*X_num_rows*X_num_cols + ((tid%(num_row_patches*num_col_patches))/num_row_patches)*X_num_cols*stride + stride*(tid%num_row_patches);
                    for (int i_f=0; i_f < f_rows*f_cols; i_f++) {
                        Z[f_rows*f_cols*(z_batch + z_batch_row + z_batch_col) + i_f] = X[x_start + (i_f/f_rows)*X_num_cols + i_f%f_cols];
                    }
                }
        } """, "im2col")

        row2im_kernel = cp.RawKernel(
        """ extern "C" __global__ void row2im(float* X, float* Z, int f_rows, int f_cols,
                                            int batch_size, int num_channels, int X_num_rows, int X_num_cols,
                                            int num_row_patches, int num_col_patches, int stride) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid < batch_size*num_channels*num_row_patches*num_col_patches) {
                    int x_chan = (tid/(num_row_patches*num_col_patches));
                    int z_batch = int(tid/(num_channels*num_row_patches*num_col_patches))*num_channels*num_row_patches*num_col_patches;
                    int z_batch_index = tid%(num_channels*num_row_patches*num_col_patches);
                    int z_batch_col = int(z_batch_index/(num_row_patches*num_col_patches));
                    int z_batch_row = (z_batch_index%(num_row_patches*num_col_patches))*num_channels;
                    int x_start = x_chan*X_num_rows*X_num_cols + ((tid%(num_row_patches*num_col_patches))/num_row_patches)*X_num_cols*stride + stride*(tid%num_row_patches);
                    for (int i_f=0; i_f < f_rows*f_cols; i_f++) {
                        atomicAdd(&X[x_start + (i_f/f_rows)*X_num_cols + i_f%f_cols], Z[f_rows*f_cols*(z_batch + z_batch_row + z_batch_col) + i_f]);
                        //p[z_batch + z_batch_row + z_batch_col] = tid; // X[x_start + (i_f/f_rows)*X_num_cols + i_f%f_cols];
                    }
                }
        } """, "row2im")

        return im2col_kernel, row2im_kernel

    def regulariser_forward(self):
        out = 0
        if self.weight_regulariser:
            out += self.weight_regulariser.forward(self.learned_params["weights"])
        return out

    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__ 
        base_dset.attrs["with_bias"] = self.with_bias
        base_dset.attrs["num_filters"] = self.num_filters
        base_dset.attrs["filter_chans"] = self.filter_chans
        base_dset.attrs["f_rows"] = self.f_rows
        base_dset.attrs["f_cols"] = self.f_cols
        base_dset.attrs["stride"] = self.stride
        base_dset.attrs["padding"] = self.padding
        dset = open_f.create_dataset(self.layer_name + "/weights", 
                                     self.learned_params['weights'].shape,
                                     dtype=self.learned_params['weights'].dtype)
        dset[:] = cp.asnumpy(self.learned_params["weights"])
        if self.weight_regulariser is not None:
            dset.attrs["weight_regulariser_type"] = np.string_(self.weight_regulariser.type)
            dset.attrs["weight_regulariser_strength"] = np.string_(self.weight_regulariser.strength)
        if self.with_bias:
            dset_bias = open_f.create_dataset(self.layer_name + "/bias",
                                              self.learned_params['bias'].shape,
                                              dtype=self.learned_params['bias'].dtype)
            dset_bias[:] = cp.asnumpy(self.learned_params['bias'])
        
        if save_grads:
            dset_grads = open_f.create_dataset(self.layer_name + "/grads/weights",
                                               self.grads['weights'].shape,
                                               dtype=self.learned_params['weights'].dtype)
            dset_grads[:] = cp.asnumpy(self.grads["weights"])
            if self.with_bias:
                dset_grads_bias = open_f.create_dataset(self.layer_name + "/grads/bias",
                                                        self.grads['bias'].shape,
                                                        dtype=self.learned_params['bias'].dtype)
                dset_grads_bias[:] = cp.asnumpy(self.grads["bias"])

    def load_from_h5(self, open_f, load_grads=True):
        self.num_filters = open_f[self.layer_name + '/layer_info'].attrs['num_filters']
        self.filter_chans = open_f[self.layer_name + '/layer_info'].attrs['filter_chans']
        self.with_bias = open_f[self.layer_name + '/layer_info'].attrs['with_bias']
        self.f_rows = open_f[self.layer_name + '/layer_info'].attrs['f_rows']
        self.f_cols = open_f[self.layer_name + '/layer_info'].attrs['f_cols']
        self.stride = open_f[self.layer_name + '/layer_info'].attrs['stride']
        self.padding = open_f[self.layer_name + '/layer_info'].attrs['padding']

        weight_regulariser_type = open_f[self.layer_name + '/weights'].attrs.get("weight_regulariser_type", None)
        if weight_regulariser_type:
            weight_regulariser_strength = open_f[self.layer_name + '/weights'].attrs["weight_regulariser_strength"]
            if weight_regulariser_type == b"l2":
                self.weight_regulariser = l2.l2(strength=float(weight_regulariser_strength))

        self.learned_params['weights'] = open_f[self.layer_name + '/weights'][:]
        if self.with_bias:
            self.learned_params['bias'] = open_f[self.layer_name + '/bias'][:]
        if load_grads:
            self.grads['weights'] = open_f[self.layer_name + '/grads/weights'][:]
            if self.with_bias:
                self.grads['bias'] = open_f[self.layer_name + '/grads/bias'][:]
