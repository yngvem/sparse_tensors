import numpy as np
from numba import njit



def make_numba_list(native_list):
    numba_list = List()
    for x in native_list:
        numba_list.append(x)
    return numba_list


def _mttkrp_numpy(nnz_indices, data, factor_matrices, skip_mode):
    rank = factor_matrices[0].shape[1]
    mttkrp = np.empty_like(factor_matrices[skip_mode])
    temp = np.zeros((1, rank)) + data[:, np.newaxis]

    for curr_mode, factor_matrix in enumerate(factor_matrices):
        if curr_mode == skip_mode:
            continue
        selected_elements = factor_matrix[nnz_indices[curr_mode], :]
        temp *= selected_elements
    
    for r in range(rank):
        mttkrp[:, r] = np.bincount(
            nnz_indices[skip_mode], temp[:, r], minlength=factor_matrices[skip_mode].shape[0]
        )
    return mttkrp


@njit(cache=False, )
def single_idx_mttkrp(index, tensor_element, factor_matrices, skip_mode):
    out = np.zeros((factor_matrices[0].shape[1])) + tensor_element
    for curr_mode, factor_matrix in enumerate(factor_matrices):
        if curr_mode != skip_mode:
            out *= factor_matrix[index[curr_mode]]
    return out


@njit(cache=False,)
def _mttkrp_numba(nnz_indices, data, factor_matrices, skip_mode):
    if nnz_indices.shape[0] != len(factor_matrices):
        raise ValueError
    rank = factor_matrices[0].shape[1]
    mttkrp = np.zeros_like(factor_matrices[skip_mode])
    temp = np.zeros((factor_matrices[0].shape[1]))

    for i, index in enumerate(nnz_indices.T):
        mttkrp[index[skip_mode]] += single_idx_mttkrp(index, data[i], factor_matrices, skip_mode)
    return mttkrp


def _mttkrp_from_raw(nnz_indices, data, factor_matrices, skip_mode):
    if USE_NUMBA:
        numba_factors = make_numba_list(factor_matrices)
        return _mttkrp_numba(np.array(nnz_indices), data, numba_factors, skip_mode)
    else:
        return _mttkrp_numpy(nnz_indices, data, factor_matrices, skip_mode)


def mttkrp(tensor, factor_matrices, skip_mode):
    nonzero = np.array(tensor.nonzero())
    return _mttkrp_from_raw(nonzero, tensor.data, factor_matrices, skip_mode)

