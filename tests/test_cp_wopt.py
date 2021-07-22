
import numpy as np
from tensorly.cp_tensor import cp_to_tensor
from sparse_tensors.cp_grad import (
    cp_wopt, construct_cp_elements, CPTensorOptimizeWrapper,
    cp_loss_and_grad, _cp_loss_and_grad, make_numba_list,
    cp_sgd, _construct_cp_elements, make_numba_list
)
import sparse
from scipy.optimize import check_grad


def test_construct_cp_elements():
    shape = (10, 20, 15)
    A = 1 + np.random.standard_normal((shape[0], 3))
    B = 1 + np.random.standard_normal((shape[1], 3))
    C = 1 + np.random.standard_normal((shape[2], 3))
    cp_tensor = None, [A, B, C]
    X = cp_to_tensor((None, (A, B, C)))
    all_indices = np.unravel_index(np.arange(np.prod(shape)), shape)

    np.testing.assert_allclose(
        construct_cp_elements(est_cp, all_indices),
        cp_to_tensor(est_cp).ravel()
    )
    np.testing.assert_allclose(
        _construct_cp_elements(make_numba_list(est_cp[1]), np.array(all_indices).T),
        cp_to_tensor(est_cp).ravel()
    )

def test_cp_wopt_can_impute():
    A = 1 + np.random.standard_normal((10, 3))
    B = 1 + np.random.standard_normal((20, 3))
    C = 1 + np.random.standard_normal((15, 3))

    X = cp_to_tensor((None, (A, B, C)))
    X_zeroed = X.copy()
    mask = np.zeros((10, 20, 15))

    masked_indices_i = np.random.randint(0, 10, 5)
    masked_indices_j = np.random.randint(0, 20, 5)
    masked_indices_k = np.random.randint(0, 15, 5)
    X_zeroed[masked_indices_i, masked_indices_j, masked_indices_k] = 0

    weights = np.ones_like(X_zeroed)

    X_zeroed = sparse.COO.from_numpy(X_zeroed)
    weights[masked_indices_i, masked_indices_j, masked_indices_k] = 0

    weights = sparse.COO.from_numpy(weights)

    est_cp, res = cp_wopt(X_zeroed, 3, weights, method="l-bfgs-b", options=dict(ftol=1e-100, gtol=1e-8))
    X_hat = cp_to_tensor(est_cp)
    np.testing.assert_allclose(
        X_hat[masked_indices_i, masked_indices_j, masked_indices_k],
        X[masked_indices_i, masked_indices_j, masked_indices_k],
    )


def notest_cp_wopt_can_impute_large_tensor():
    shape = 20, 30, 40, 50
    A = 1 + np.random.standard_normal((shape[0], 3))
    B = 1 + np.random.standard_normal((shape[1], 3))
    C = 1 + np.random.standard_normal((shape[2], 3))
    D = 1 + np.random.standard_normal((shape[3], 3))
    cp_tensor = (None, (A, B, C, D))

    total_n_elements = np.prod(shape)
    fraction_missing = 0.99
    n_elements = int((1 - fraction_missing)*total_n_elements)
    avail_indices_i = np.random.randint(0, shape[0], n_elements)
    avail_indices_j = np.random.randint(0, shape[1], n_elements)
    avail_indices_k = np.random.randint(0, shape[2], n_elements)
    avail_indices_l = np.random.randint(0, shape[3], n_elements)
    avail_indices = (avail_indices_i, avail_indices_j, avail_indices_k, avail_indices_l)
    tensor_elements = construct_cp_elements(cp_tensor, avail_indices)

    X_zeroed = sparse.COO(avail_indices, tensor_elements, shape)
    importance_weights = sparse.COO(avail_indices, np.ones_like(tensor_elements), shape)
    
    est_cp, res = cp_wopt(X_zeroed, 3, importance_weights, options=dict(ftol=0, gtol=1e-8, iprint=99))
    sampled_elements = construct_cp_elements(est_cp, avail_indices)
    np.testing.assert_allclose(
        sampled_elements,
        tensor_elements,
    )
    

def test_cp_sgd_can_impute():
    A = 1 + np.random.standard_normal((10, 3))
    B = 1 + np.random.standard_normal((20, 3))
    C = 1 + np.random.standard_normal((15, 3))

    X = cp_to_tensor((None, (A, B, C)))
    X_zeroed = X.copy()
    mask = np.zeros((10, 20, 15))

    #masked_indices_i = np.random.randint(0, 10, 5)
    #masked_indices_j = np.random.randint(0, 20, 5)
    #masked_indices_k = np.random.randint(0, 15, 5)
    #X_zeroed[masked_indices_i, masked_indices_j, masked_indices_k] = 0

    weights = np.ones_like(X_zeroed)

    X_zeroed = sparse.COO.from_numpy(X_zeroed)
    #weights[masked_indices_i, masked_indices_j, masked_indices_k] = 0

    weights = sparse.COO.from_numpy(weights)

    est_cp, res = cp_sgd(X_zeroed, 3, weights, maxiter=10000, learning_rate=1e3, batch_size=10*20*15)
    X_hat = cp_to_tensor(est_cp)
    np.testing.assert_allclose(
        X_hat,# [masked_indices_i, masked_indices_j, masked_indices_k]
        X,# [masked_indices_i, masked_indices_j, masked_indices_k]
    )

def test_cp_grad():
    shape = 2, 3, 4, 5
    rank = 3
    A = 1 + np.random.standard_normal((shape[0], rank))
    B = 1 + np.random.standard_normal((shape[1], rank))
    C = 1 + np.random.standard_normal((shape[2], rank))
    D = 1 + np.random.standard_normal((shape[3], rank))
    cp_tensor = (None, (A, B, C, D))

    total_n_elements = np.prod(shape)
    fraction_missing = 0.5
    n_elements = int((1 - fraction_missing)*total_n_elements)
    avail_indices_i = np.random.randint(0, shape[0], n_elements)
    avail_indices_j = np.random.randint(0, shape[1], n_elements)
    avail_indices_k = np.random.randint(0, shape[2], n_elements)
    avail_indices_l = np.random.randint(0, shape[3], n_elements)
    avail_indices = (avail_indices_i, avail_indices_j, avail_indices_k, avail_indices_l)
    tensor_elements = construct_cp_elements(cp_tensor, avail_indices)

    X_zeroed = sparse.COO(avail_indices, tensor_elements, shape)
    importance_weights = sparse.COO(avail_indices, np.ones_like(tensor_elements), shape)
    
    

    wrapper = CPTensorOptimizeWrapper(X_zeroed.shape, rank)
    def f(x):
        return cp_loss_and_grad(wrapper.unflatten_cp_tensor(x), X_zeroed.nonzero(), X_zeroed.data, importance_weights.data)[0]
    
    def grad(x):
        return wrapper.flatten_cp_tensor(cp_loss_and_grad(wrapper.unflatten_cp_tensor(x), X_zeroed.nonzero(), X_zeroed.data, importance_weights.data)[1])
    

    assert check_grad(f, grad, wrapper.flatten_cp_tensor(cp_tensor)) < 1e-4


def test_cp_numba_grad():
    shape = 2, 3, 4, 5
    rank = 3
    A = 1 + np.random.standard_normal((shape[0], rank))
    B = 1 + np.random.standard_normal((shape[1], rank))
    C = 1 + np.random.standard_normal((shape[2], rank))
    D = 1 + np.random.standard_normal((shape[3], rank))
    cp_tensor = (None, (A, B, C, D))

    total_n_elements = np.prod(shape)
    fraction_missing = 0.5
    n_elements = int((1 - fraction_missing)*total_n_elements)
    avail_indices = np.unravel_index(np.random.choice(total_n_elements, n_elements, replace=False), shape)
    tensor_elements = construct_cp_elements(cp_tensor, avail_indices)

    X_zeroed = sparse.COO(avail_indices, tensor_elements, shape)
    importance_weights = sparse.COO(avail_indices, np.ones_like(tensor_elements), shape)
    

    indices = np.array(avail_indices).T

    wrapper = CPTensorOptimizeWrapper(X_zeroed.shape, rank)
    def f(x):
        return _cp_loss_and_grad(
            make_numba_list(wrapper.unflatten_cp_tensor(x)[1]),
            indices,
            tensor_elements,
            importance_weights.data
        )[0]
    

    def grad(x):
        factor_matrices = wrapper.unflatten_cp_tensor(x)[1]
        factor_matrices = make_numba_list(factor_matrices)
        loss, grad = _cp_loss_and_grad(factor_matrices, indices, tensor_elements, importance_weights.data)
        return wrapper.flatten_cp_tensor((None, grad))
    

    assert check_grad(f, grad, wrapper.flatten_cp_tensor(cp_tensor)) < 1e-4

    all_indices = np.array(np.unravel_index(np.arange(total_n_elements), shape))
    all_tensor_elements = construct_cp_elements(cp_tensor, all_indices)
    all_weights = np.ones_like(all_tensor_elements)
    def f(x):
        return _cp_loss_and_grad(
            make_numba_list(wrapper.unflatten_cp_tensor(x)[1]),
            all_indices.T,
            all_tensor_elements,
            all_weights
        )[0]
    
    def grad(x):
        factor_matrices = wrapper.unflatten_cp_tensor(x)[1]
        factor_matrices = make_numba_list(factor_matrices)
        loss, grad = _cp_loss_and_grad(factor_matrices, all_indices.T, all_tensor_elements, all_weights)
        return wrapper.flatten_cp_tensor((None, grad))
    
    assert np.abs(f(wrapper.flatten_cp_tensor(cp_tensor))) < 1e-10
    assert np.linalg.norm(grad(wrapper.flatten_cp_tensor(cp_tensor))) < 1e-10
    