import im2col
import numpy as np
import cupy as cp
import itertools

from numpy.lib.function_base import select
from .layer import Layer
from regularisers import l2

class DepthwiseConvLayer(Layer):
    def __init__(self, layer_name, filter_block_shape=None,
                 stride=1, padding=1, with_bias=True,
                 weight_regulariser=None, weight_initialiser="normal"):
        """
        filter_block_shape = (num_incoming_channels, num_filter_rows, num_filter_cols)
        """
        super().__init__(layer_name)
        self.stride = stride
        self.padding = padding
        self.with_bias = with_bias
        self.weight_regulariser = weight_regulariser
        self.weight_initialiser = weight_initialiser
        if filter_block_shape is not None:
            self.num_filters, self.f_rows, self.f_cols = filter_block_shape
            if self.weight_initialiser == "glorot_uniform":
                limit = np.sqrt(6.0 / (2*self.num_filters))
                weights = np.random.uniform(low=-limit,high=limit,size=filter_block_shape).astype(np.float32)
            elif self.weight_initialiser == "normal":
                weights = 0.01*np.random.randn(*filter_block_shape).astype(np.float32)
            self.learned_params = {"weights": weights}
            self.grads = {"weights": np.zeros_like(weights).astype(np.float32)}
            if with_bias:
                bias = np.zeros(self.num_filters).astype(np.float32)
                self.learned_params.update({"bias": bias})
                self.grads.update({"bias": np.zeros_like(bias, dtype=np.float32)})
        else:
            self.num_filters = None
            self.learned_params = {}
            self.grads = {}

    def __repr__(self):
        out = "DepthwiseConvLayer({}, ".format(self.layer_name)
        if self.num_filters is not None:
            out += "filter_block_shape=({}, {}, {}), ".format(self.num_filters,
                                                              self.f_rows,
                                                              self.f_cols)
        out += "stride={}, padding={}, with_bias={}, weight_regulariser={})".format(self.stride,
                                                             self.padding,
                                                             self.with_bias,
                                                             repr(self.weight_regulariser))
        return out

    def to_gpu(self):
        super().to_gpu()
        self.forward_kernel, self.backward_kernel = self.get_kernels()

    def pad_input(self, X):
        xp = cp.get_array_module(X)
        return xp.pad(
                X, ((0, 0), (0, 0), 
                    (self.padding, self.padding),
                    (self.padding, self.padding)), 
                    "constant"
                )

    def forward(self, X, test_mode=False):
        if self.is_on_gpu:
            return self.forward_cp(X, test_mode=test_mode)
        else:
            return self.forward_cy(X, test_mode=test_mode)

    def forward_cy(self, X, test_mode=False):
        X = self.pad_input(X)
        if not test_mode:
            self.X = X
        out, self.num_row_patches, self.num_col_patches = im2col.depthwise_conv_cy(
                            X, self.learned_params["weights"],
                            self.learned_params["weights"].shape[-2],
                            self.learned_params["weights"].shape[-1], self.stride
                        )
        if self.with_bias:
            out += self.learned_params["bias"][np.newaxis, :, np.newaxis, np.newaxis]
        return out

    def forward_cp(self, X, test_mode=False):
        X = self.pad_input(X)
        if not test_mode:
            self.X = X

        self.num_row_patches = ((X.shape[2] - self.f_rows)/self.stride) + 1
        self.num_col_patches = ((X.shape[3] - self.f_cols)/self.stride) + 1

        out = cp.zeros((X.shape[0], X.shape[1], int(self.num_row_patches), int(self.num_col_patches)), dtype=np.float32)
        numThreadBlocks = int((X.shape[0]*X.shape[1]*self.num_row_patches*self.num_col_patches + 1024 - 1) / 1024)
        self.forward_kernel((numThreadBlocks,), (1024,), (cp.asarray(self.learned_params["weights"]), X, out, 
                                             X.shape[0],X.shape[1],X.shape[2],X.shape[3],
                                             self.f_rows, self.f_cols, int(self.num_row_patches), int(self.num_col_patches),
                                             self.stride))

        if self.with_bias:
            out += self.learned_params["bias"][np.newaxis, :, np.newaxis, np.newaxis]
        return out

    def get_kernels(self):
        forward_kernel = cp.RawKernel("""
            extern "C" __global__
            void forward_conv(const float* f, const float* x, float* z,
                    int batch_size, int num_chans, int num_rows, int num_cols,
                    int f_rows, int f_cols, 
                    int num_row_patches, int num_col_patches,
                    int stride) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                if(tid < num_row_patches*num_col_patches*num_chans*batch_size) {
                    int x_chan = tid/(num_row_patches*num_col_patches);
                    int x_start = x_chan*num_rows*num_cols + ((tid%(num_row_patches*num_col_patches))/num_row_patches)*num_cols*stride + stride*(tid%num_row_patches);
                    for (int i_f=0;i_f<f_rows*f_cols;i_f++) {
                        z[tid] += f[(x_chan%num_chans)*f_rows*f_cols + i_f]*x[x_start + (i_f/f_rows)*num_cols + i_f%f_cols];
                    }
                }
            }
            """, 'forward_conv')
        backward_kernel = cp.RawKernel("""
            extern "C" __global__ 
            void backward_conv(float* upstream_dx, float* X, float* w, float* padded_dx, float* dw,
                            int X_num_rows, int X_num_cols, int X_num_chans, int X_batch_size,
                            int num_row_patches, int num_col_patches, 
                            int f_rows, int f_cols, int stride)
                {
                    int tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid < num_row_patches*num_col_patches*X_num_chans*X_batch_size) {         
                        int x_chan = (tid/(num_row_patches*num_col_patches));
                        int x_start = x_chan*X_num_rows*X_num_cols + ((tid%(num_row_patches*num_col_patches))/num_row_patches)*X_num_cols*stride + stride*(tid%num_row_patches);

                        for (int i_f=0;i_f<f_rows*f_cols;i_f++) {
                            atomicAdd(&dw[(x_chan%X_num_chans)*f_rows*f_cols + i_f], upstream_dx[tid]*X[x_start + (i_f/f_rows)*X_num_cols + i_f%f_cols]);
                            atomicAdd(&padded_dx[x_start + (i_f/f_rows)*X_num_cols + i_f%f_cols], upstream_dx[tid]*w[(x_chan%X_num_chans)*f_rows*f_cols + i_f]);
                        }
                    }
                }""",
            "backward_conv")
        
        return forward_kernel, backward_kernel

    def forward_old(self, X, test_mode=False):
        X = self.pad_input(X)
        self.patches, self.num_row_patches, self.num_col_patches = im2col.depthwise_im2col_cy(X,
                                                                self.learned_params["weights"].shape[1],
                                                                self.learned_params["weights"].shape[2],
                                                                self.stride)
        out = np.zeros((X.shape[0], X.shape[1], self.num_row_patches, self.num_col_patches), dtype=np.float32)
        flat_filters = self.learned_params["weights"].reshape(
                -1, self.learned_params["weights"].shape[-2]*self.learned_params["weights"].shape[-1]
            )
        for i in range(self.patches.shape[0]):
            channel_out = np.dot(self.patches[i, :, :], flat_filters[i, :].T)
            for j in range(X.shape[0]):
                out[j, i, :, :] = channel_out[j*self.num_row_patches*self.num_col_patches:
                                             (j+1)*self.num_row_patches*self.num_col_patches].reshape(
                                                 self.num_row_patches, self.num_col_patches
                                                 )
        if self.with_bias:
            out += self.learned_params["bias"][np.newaxis, :, np.newaxis, np.newaxis]
        return out

    def forward_einsum(self, X, test_mode=False):
        X = self.pad_input(X)
        self.patches, self.num_row_patches, self.num_col_patches = im2col.depthwise_im2col_einsum_cy(X,
                                                                        self.learned_params["weights"].shape[1],
                                                                        self.learned_params["weights"].shape[2],
                                                                        self.stride)
        flat_w = self.learned_params["weights"].reshape(self.learned_params["weights"].shape[0],-1).T # Should probably just store in the right shape...
        out = np.einsum("ijkl,lj->ijk", self.patches, flat_w, optimize='greedy').reshape(X.shape[0],
                                                                X.shape[1],
                                                                self.num_row_patches,
                                                                self.num_col_patches)
        if self.with_bias:
            out += self.learned_params["bias"][np.newaxis, :, np.newaxis, np.newaxis]
        return out

    def backward(self, upstream_dx):
        if self.is_on_gpu:
            return self.backward_cp(upstream_dx)
        else:
            return self.backward_cy(upstream_dx)

    def backward_cy(self, upstream_dx):
        if self.with_bias:
            self.grads["bias"] = np.sum(upstream_dx, axis=(0,2,3))
        
        dx, dw = im2col.depthwise_backward_direct_cy(upstream_dx, self.X, self.learned_params["weights"],
                                                     self.num_row_patches, self.num_col_patches, self.stride,
                                                     self.padding)
        self.grads["weights"] = np.sum(dw, axis=0)
        if self.weight_regulariser:
            self.grads["weights"] += self.weight_regulariser.backward(self.learned_params["weights"])
        return np.array(dx)

    def backward_cp(self, upstream_dx):
        if self.with_bias:
            self.grads["bias"] = cp.asnumpy(cp.sum(upstream_dx, axis=(0,2,3)))

        padded_dx = cp.zeros(self.X.shape, dtype=cp.float32)
        dw = cp.zeros(self.learned_params["weights"].shape, dtype=cp.float32)
        numThreadBlocks = int((upstream_dx.shape[0]*upstream_dx.shape[1]*self.num_row_patches*self.num_col_patches + 1024 - 1) / 1024)
        self.backward_kernel((numThreadBlocks,), (1024,), (upstream_dx, self.X, cp.asarray(self.learned_params["weights"]),
                                                 padded_dx, dw, 
                                                 self.X.shape[-2], self.X.shape[-1], self.X.shape[1], self.X.shape[0],
                                                 int(self.num_row_patches), int(self.num_col_patches), 
                                                 self.learned_params["weights"].shape[-2], 
                                                 self.learned_params["weights"].shape[-1],
                                                 self.stride, self.padding))
        
        self.grads["weights"] = dw
        if self.weight_regulariser:
            self.grads["weights"] += self.weight_regulariser.backward(self.learned_params["weights"])

        if self.padding > 0:
            dx = padded_dx[:,:,self.padding:-self.padding,self.padding:-self.padding]
            return dx
        else:
            return padded_dx

    def backward_old(self, upstream_dx):
        if self.with_bias:
            self.grads["bias"] = np.sum(upstream_dx, axis=(0,2,3))
        dx_rows = np.zeros((upstream_dx.shape[1],
                            upstream_dx.shape[0]*self.num_row_patches*self.num_col_patches,
                            self.learned_params["weights"].shape[-2]*self.learned_params["weights"].shape[-1]),
                            dtype=np.float32)
        dw = np.zeros_like(self.learned_params["weights"], dtype=np.float32)
        for out_c in range(upstream_dx.shape[1]):
            upstream = upstream_dx[:, out_c, :, :].reshape(-1,1)
            dw[out_c, :, :] = np.dot(upstream.T, self.patches[out_c]).reshape(self.learned_params["weights"].shape[-2],
                                                                              self.learned_params["weights"].shape[-1])
            dx_rows[out_c, :, :] = np.dot(upstream, self.learned_params["weights"][out_c,:,:].reshape(1,-1))
        self.grads["weights"] = dw

        dx = im2col.depthwise_row2im_cy(dx_rows, upstream_dx.shape[0], self.num_row_patches, self.num_col_patches,
                                        self.learned_params["weights"].shape[-2], self.learned_params["weights"].shape[-1],
                                        self.num_filters, self.stride, self.padding)

        return np.array(dx)
    
    def backward_einsum(self, upstream_dx):
        if self.with_bias:
            self.grads["bias"] = np.sum(upstream_dx, axis=(0,2,3))
        upstream = upstream_dx.reshape(upstream_dx.shape[0], upstream_dx.shape[1],
                                       upstream_dx.shape[2]*upstream_dx.shape[3])
        self.grads["weights"] = np.einsum("ijk,ijkl->jl",
                                          upstream,
                                          self.patches, optimize='greedy').reshape(
                                              self.learned_params["weights"].shape
                                        )
        upstream = upstream.reshape(*upstream.shape, 1)
        w_flat = self.learned_params["weights"].reshape(self.learned_params["weights"].shape[0],
                                                        self.learned_params["weights"].shape[1]*self.learned_params["weights"].shape[2],
                                                        1)
        dx_rows = np.einsum("ijkl,jlm->ijkl", upstream, w_flat, optimize='greedy')
        dx = im2col.depthwise_row2im_einsum_cy(dx_rows, upstream_dx.shape[0], self.num_row_patches, self.num_col_patches,
                                        self.learned_params["weights"].shape[-2], self.learned_params["weights"].shape[-1],
                                        self.num_filters, self.stride, self.padding)
        return dx

    def depthwise_im2col_np(self, X, f_rows, f_cols, stride):
        row_patch_ix = range(0, X.shape[2] - f_rows + 1, stride)
        col_patch_ix = range(0, X.shape[3] - f_cols + 1, stride)
        channels_patched = []

        for c in range(X.shape[1]):
            f_patches = []
            for b, i, j in itertools.product(range(X.shape[0]),
                                            row_patch_ix,
                                            col_patch_ix):
                f_patches.append(X[b, c, i:i+f_rows, j:j+f_cols].flatten())
            channels_patched.append(np.stack(f_patches))
        patches = np.stack(channels_patched)

        return patches, len(row_patch_ix), len(col_patch_ix)

    def depthwise_row2im_np(self, X, batch_size, num_row_patches, num_col_patches,
                         f_rows, f_cols, num_channels, stride, padding):
        num_padded_rows = stride*(num_row_patches - 1) + f_rows
        num_padded_cols = stride*(num_col_patches - 1) + f_cols
        padded_out = np.zeros((batch_size, num_channels, num_padded_rows, num_padded_cols), dtype=np.float32)
        for iChan in range(num_channels):
            for iBatch in range(batch_size):
                for iRow in range(num_row_patches):
                    for iCol in range(num_col_patches):
                        for iFrow in range(f_rows):
                            for iFcol in range(f_cols):
                                padded_out[iBatch, iChan, stride*iRow + iFrow, stride*iCol + iFcol] += ( 
                                    X[iChan, num_col_patches*(num_row_patches*iBatch + iRow) + iCol, iFrow*f_cols + iFcol]
                                )
        if padding > 0:
            unpadded_out = padded_out[:,:,padding:-padding,padding:-padding]
        else:
            unpadded_out = padded_out
        return unpadded_out

    def save_to_h5(self, open_f, save_grads=True):
        base_dset = open_f.create_dataset(self.layer_name + "/layer_info", dtype=np.float32)
        base_dset.attrs["type"] = self.__class__.__name__ 
        base_dset.attrs["stride"] = self.stride
        base_dset.attrs["padding"] = self.padding
        base_dset.attrs["with_bias"] = self.with_bias
        base_dset.attrs["num_filters"] = self.num_filters
        base_dset.attrs["f_rows"] = self.f_rows
        base_dset.attrs["f_cols"] = self.f_cols
        dset = open_f.create_dataset(self.layer_name + "/weights", 
                                     self.learned_params['weights'].shape,
                                     dtype=self.learned_params['weights'].dtype)
        dset[:] = cp.asnumpy(self.learned_params['weights'])
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
        self.f_cols = open_f[self.layer_name + '/layer_info'].attrs['f_cols']
        self.f_rows = open_f[self.layer_name + '/layer_info'].attrs['f_rows']
        self.num_filters = open_f[self.layer_name + '/layer_info'].attrs['num_filters']
        self.stride = open_f[self.layer_name + '/layer_info'].attrs['stride']
        self.padding = open_f[self.layer_name + '/layer_info'].attrs['padding']
        self.with_bias = open_f[self.layer_name + '/layer_info'].attrs['with_bias']

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