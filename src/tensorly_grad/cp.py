from tensorly.decomposition._cp import initialize_cp
from .optimize_wrapper import minimize
import numpy as np
import sparse
import warnings
from .tenalg import _mttkrp_numba, _mttkrp_from_raw, mttkrp
try:
    from numba import njit
    from numba.typed import List
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False


_oldstack = sparse.stack
from functools import wraps
@wraps(sparse.stack)
def _stack(arrays, *args, **kwargs):
    if all(isinstance(arr, np.ndarray) for arr in arrays):
        return np.stack(arrays, *args, **kwargs)
    return _oldstack(arrays, *args, **kwargs)
sparse.stack = _stack
del _stack


def make_numba_list(native_list):
    numba_list = List()
    for x in native_list:
        numba_list.append(x)
    return numba_list


class CPTensorOptimizeWrapper:
    def __init__(self, shape, rank):
        self.shape = shape
        self.rank = rank
        self.array_size = np.sum(shape)*rank

        self.factor_array_indices = [0]*(1 + len(shape))
        current_index = 0
        for mode, l in enumerate(shape):
            self.factor_array_indices[mode+1] = current_index + l*rank
            current_index = self.factor_array_indices[mode+1]

    def flatten_cp_tensor(self, cp_tensor):
        """Convert a CPTensor into a one-dimensional array.
        """
        weights, factors = cp_tensor
        if weights is not None:
            warnings.warn("The CP tensor have weights, which is currently not supported. Multiply them into the factor matrices and set the weights to None.")
        
        array = np.empty(self.array_size)
        for mode, factor in enumerate(factors):
            current_index = self.factor_array_indices[mode]
            next_index = self.factor_array_indices[mode+1]
            array[current_index:next_index] = factor.ravel()

        return array

    def unflatten_cp_tensor(self, arr):
        """Convert a one-dimensional array into a CPTensor.
        """
        factors = [
            arr[start_idx:stop_idx].reshape(length, self.rank)
            for length, start_idx, stop_idx in zip(self.shape, self.factor_array_indices, self.factor_array_indices[1:])
        ]
        return (None, factors)




def cp_loss_and_grad(cp_tensor, indices, tensor_data, importance_weights=None):
    indices = np.asarray(indices)
    weights, factors = cp_tensor
    if weights is not None:
        warnings.warn("The CP tensor have weights, which is currently not supported. Multiply them into the factor matrices and set the weights to None.")

    cp_elements = construct_cp_elements(cp_tensor, indices)
    error = cp_elements - tensor_data
    
    grads = [_mttkrp_from_raw(indices, error, factors, skip_mode) for skip_mode, _ in enumerate(factors)]

    return 0.5*np.sum(error**2), (None, grads)


def construct_cp_elements(cp_tensor, indices):
    """

    Arguments
    ---------
    cp_tensor
    indices : iterable of array(int)
        The first element is an array of indices along the first mode,
        the second element is an array of indices along the second mode, and so forth.

        Generally obtained by calling ``np.nonzero`` on an array.
    """
    weights, factors = cp_tensor
    rank = factors[0].shape[1]

    rank_one_elements = np.ones((len(indices[0]), rank))
    if weights is not None:
        rank_one_elements *= weights[np.newaxis, :]
        
    for mode, factor in enumerate(factors):
        rank_one_elements *= factor[indices[mode], :]

    return rank_one_elements.sum(1)


def cp_wopt(tensor, rank, importance_weights=None, init="random", method='l-bfgs-b', **kwargs):
    wrapper = CPTensorOptimizeWrapper(tensor.shape, rank)
    if init == "svd" and isinstance(tensor, sparse.SparseArray):
        warnings.warn("SVD init can be time-consuming and is not recommended for sparse tensors")
    
    weights, factors = initialize_cp(tensor, rank, init=init)
    n_factors = len(factors)
    for factor in factors:
        factor *= weights[np.newaxis, :]**(1/n_factors)

    x0 = wrapper.flatten_cp_tensor((None, factors))

    res = minimize(
        lambda x: cp_loss_and_grad(x, tensor.nonzero(), tensor.data, importance_weights.data),
        x0,
        jac=True, 
        obj_to_array=wrapper.flatten_cp_tensor,
        array_to_obj=wrapper.unflatten_cp_tensor,
        method=method,
        **kwargs
    )
    x0 = res.x
    return res.obj, res


@njit(cache=False,)
def _construct_cp_elements(factor_matrices, indices):
    rank_one_elements = np.empty(indices.shape[0])
    element_row = np.ones(factor_matrices[0].shape[1], dtype=np.float64)
    for i, index in enumerate(indices):
        element_row[:] = 1.
        for mode, fm in enumerate(factor_matrices):
            fm_elements = fm[index[mode]]
            element_row[:] *= fm_elements

        rank_one_elements[i] = element_row.sum()
    return rank_one_elements


@njit(cache=False,)
def _cp_loss_and_grad(factor_matrices, indices, tensor_data, importance_weights,):
    cp_elements = _construct_cp_elements(factor_matrices, indices)
    error = importance_weights * (cp_elements - tensor_data)
    
    grads = [_mttkrp_numba(indices.T, error, factor_matrices, skip_mode) for skip_mode, _ in enumerate(factor_matrices)]
    return 0.5*np.sum(error**2), grads


@njit(cache=False,)
def _cp_sgd(factor_matrices, indices, tensor_data, importance_weights, batch_size, learning_rate, maxiter, momentum):
    # factor_matrices -> y
    # optimal_factors -> x
    loss = []
    num_samples = tensor_data.shape[0]
    prev_factors = [np.zeros(fm.shape, dtype=fm.dtype) for fm in factor_matrices]
    aux_factors = [fm.copy() for fm in factor_matrices]
    num_modes = len(factor_matrices)
    
    iterations_per_epoch = int(np.floor(num_samples / batch_size))
    num_epochs = int(np.ceil(maxiter / iterations_per_epoch))
    theta = theta_next = 0
    for epoch in range(num_epochs):
        permutation = np.random.permutation(num_samples)

        # If it's the final epoch, then we need to adjust the number of iterations
        if epoch == num_epochs - 1:
            iterations_per_epoch = maxiter - iterations_per_epoch * (num_epochs - 1)
        
        for it in range(iterations_per_epoch):

            selected_elements = permutation[it*batch_size:(it+1)*batch_size]
            selected_indices = indices[selected_elements]
            selected_data = tensor_data[selected_elements]
            selected_weights = importance_weights[selected_elements]
            
            error, grads = _cp_loss_and_grad(aux_factors, selected_indices, selected_data, selected_weights)
            
            loss.append(error)
            
            theta_next = 0.5 + 0.5*np.sqrt(1 + 4*theta)
            beta = (theta - 1)/theta_next
            theta = theta_next
            for mode in range(num_modes):
                prev_factors[mode] = factor_matrices[mode]
                factor_matrices[mode] = factor_matrices[mode] - (learning_rate * grads[mode] / batch_size) / np.log2(2 + it)
                aux_factors[mode] = (1 + momentum*beta) * factor_matrices[mode] - momentum*beta*prev_factors[mode]
            
        
    return (None, factor_matrices), loss


def cp_sgd(tensor, rank, importance_weights=None, init="random", batch_size=100, learning_rate=1e-1, maxiter=100_000, momentum=1):
    tensor_data = tensor.data
    importance_weights = importance_weights.data
    indices = np.asarray(tensor.nonzero()).T

    weights, factors = initialize_cp(tensor, rank, init=init)
    n_factors = len(factors)
    for factor in factors:
        factor *= weights[np.newaxis, :]**(1/n_factors)

    return _cp_sgd(make_numba_list(factors), indices, tensor_data, importance_weights, batch_size, learning_rate, maxiter, momentum)

