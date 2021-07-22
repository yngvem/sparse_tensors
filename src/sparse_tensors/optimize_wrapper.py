from scipy import optimize


def minimize(fun, x0, jac=None, obj_to_array=None, array_to_obj=None, **kwargs):
    if obj_to_array is None and array_to_obj is None:
        def obj_to_array(x): return x
        def array_to_obj(x): return x
    elif obj_to_array is None or array_to_obj is None:
        raise ValueError("If one of `obj_to_array` and `array_to_obj` is not None, then the other cannot be None")

    if callable(jac):
        def jac_(x):
            return obj_to_array(jac(array_to_obj(x)))
    elif jac:
        def fun_(x):
            f, grad = fun(array_to_obj(x))
            return f, obj_to_array(grad)
        jac_ = jac
    else:
        def fun_(x):
            return fun(array_to_obj(x))
        jac_ = jac
    
    res = optimize.minimize(fun_, x0, jac=jac_, **kwargs)
    res.obj = array_to_obj(res.x)
    return res
