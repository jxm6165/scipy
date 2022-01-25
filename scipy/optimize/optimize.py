# __docformat__ = "restructuredtext en"
# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************

# A collection of optimization algorithms. Version 0.5
# CHANGES
#  Added fminbound (July 2001)
#  Added brute (Aug. 2002)
#  Finished line search satisfying strong Wolfe conditions (Mar. 2004)
#  Updated strong Wolfe conditions line search to use
#  cubic-interpolation (Mar. 2004)


# Minimization routines

__all__ = ['fmin', 'fmin_powell', 'fmin_bfgs', 'fmin_sr1', 'fmin_ncg', 'fmin_cg',
           'fminbound', 'brent', 'golden', 'bracket', 'rosen', 'rosen_der',
           'rosen_hess', 'rosen_hess_prod', 'brute', 'approx_fprime',
           'line_search', 'check_grad', 'OptimizeResult', 'show_options',
           'OptimizeWarning']

__docformat__ = "restructuredtext en"

import warnings
import sys
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
                   asarray, sqrt, Inf, asfarray, isinf)
import numpy as np
from .linesearch import (line_search_wolfe1, line_search_wolfe2,
                         line_search_wolfe2 as line_search,
                         LineSearchWarning)
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import MapWrapper
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                             'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}


class MemoizeJac:
    """ Decorator that caches the return values of a function returning `(fun, grad)`
        each time it is called. """

    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None:
            self.x = np.asarray(x).copy()
            fg = self.fun(x, *args)
            self.jac = fg[1]
            self._value = fg[0]

    def __call__(self, x, *args):
        """ returns the the function value """
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.jac


class OptimizeResult(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class OptimizeWarning(UserWarning):
    pass


def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in SciPy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)


def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper


def is_array_scalar(x):
    """Test whether `x` is either a scalar or an array scalar.

    """
    return np.size(x) == 1


_epsilon = sqrt(np.finfo(float).eps)


def vecnorm(x, ord=2):
    if ord == Inf:
        return np.amax(np.abs(x))
    elif ord == -Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x) ** ord, axis=0) ** (1.0 / ord)


def _prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    """
    Creates a ScalarFunction object for use with scalar minimizers
    (BFGS/LBFGSB/SLSQP/TNC/CG/etc).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    jac : {callable,  '2-point', '3-point', 'cs', None}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        If one of `{'2-point', '3-point', 'cs'}` is selected then the gradient
        is calculated with a relative step for finite differences. If `None`,
        then two-point finite differences with an absolute step is used.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` functions).
    bounds : sequence, optional
        Bounds on variables. 'new-style' bounds are required.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    hess : {callable,  '2-point', '3-point', 'cs', None}
        Computes the Hessian matrix. If it is callable, it should return the
        Hessian matrix:

            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

        Alternatively, the keywords {'2-point', '3-point', 'cs'} select a
        finite difference scheme for numerical estimation.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.

    Returns
    -------
    sf : ScalarFunction
    """
    if callable(jac):
        grad = jac
    elif jac in FD_METHODS:
        # epsilon is set to None so that ScalarFunction is made to use
        # rel_step
        epsilon = None
        grad = jac
    else:
        # default (jac is None) is to do 2-point finite differences with
        # absolute step size. ScalarFunction has to be provided an
        # epsilon value that is not None to use absolute steps. This is
        # normally the case from most _minimize* methods.
        grad = '2-point'
        epsilon = epsilon

    if hess is None:
        # ScalarFunction requires something for hess, so we give a dummy
        # implementation here if nothing is provided, return a value of None
        # so that downstream minimisers halt. The results of `fun.hess`
        # should not be used.
        def hess(x, *args):
            return None

    if bounds is None:
        bounds = (-np.inf, np.inf)

    # ScalarFunction caches. Reuse of fun(x) during grad
    # calculation reduces overall function evaluations.
    sf = ScalarFunction(fun, x0, args, grad, hess,
                        finite_diff_rel_step, bounds, epsilon=epsilon)

    return sf


def _clip_x_for_func(func, bounds):
    # ensures that x values sent to func are clipped to bounds

    # this is used as a mitigation for gh11403, slsqp/tnc sometimes
    # suggest a move that is outside the limits by 1 or 2 ULP. This
    # unclean fix makes sure x is strictly within bounds.
    def eval(x):
        x = _check_clip_x(x, bounds)
        return func(x)

    return eval


def _check_clip_x(x, bounds):
    if (x < bounds[0]).any() or (x > bounds[1]).any():
        warnings.warn("Values in x were outside bounds during a "
                      "minimize step, clipping to bounds", RuntimeWarning)
        x = np.clip(x, bounds[0], bounds[1])
        return x

    return x


def rosen(x):
    """
    The Rosenbrock function.

    The function computed is::

        sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be computed.

    Returns
    -------
    f : float
        The value of the Rosenbrock function.

    See Also
    --------
    rosen_der, rosen_hess, rosen_hess_prod

    Examples
    --------
    >>> from scipy.optimize import rosen
    >>> X = 0.1 * np.arange(10)
    >>> rosen(X)
    76.56

    For higher-dimensional input ``rosen`` broadcasts.
    In the following example, we use this to plot a 2D landscape.
    Note that ``rosen_hess`` does not broadcast in this manner.

    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> x = np.linspace(-1, 1, 50)
    >>> X, Y = np.meshgrid(x, x)
    >>> ax = plt.subplot(111, projection='3d')
    >>> ax.plot_surface(X, Y, rosen([X, Y]))
    >>> plt.show()
    """
    x = asarray(x)
    r = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
               axis=0)
    return r


def rosen_der(x):
    """
    The derivative (i.e. gradient) of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the derivative is to be computed.

    Returns
    -------
    rosen_der : (N,) ndarray
        The gradient of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_hess, rosen_hess_prod

    Examples
    --------
    >>> from scipy.optimize import rosen_der
    >>> X = 0.1 * np.arange(9)
    >>> rosen_der(X)
    array([ -2. ,  10.6,  15.6,  13.4,   6.4,  -3. , -12.4, -19.4,  62. ])

    """
    x = asarray(x)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = (200 * (xm - xm_m1 ** 2) -
                 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


def rosen_hess(x):
    """
    The Hessian matrix of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Hessian matrix is to be computed.

    Returns
    -------
    rosen_hess : ndarray
        The Hessian matrix of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_der, rosen_hess_prod

    Examples
    --------
    >>> from scipy.optimize import rosen_hess
    >>> X = 0.1 * np.arange(4)
    >>> rosen_hess(X)
    array([[-38.,   0.,   0.,   0.],
           [  0., 134., -40.,   0.],
           [  0., -40., 130., -80.],
           [  0.,   0., -80., 200.]])

    """
    x = atleast_1d(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros(len(x), dtype=x.dtype)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H


def rosen_hess_prod(x, p):
    """
    Product of the Hessian matrix of the Rosenbrock function with a vector.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Hessian matrix is to be computed.
    p : array_like
        1-D array, the vector to be multiplied by the Hessian matrix.

    Returns
    -------
    rosen_hess_prod : ndarray
        The Hessian matrix of the Rosenbrock function at `x` multiplied
        by the vector `p`.

    See Also
    --------
    rosen, rosen_der, rosen_hess

    Examples
    --------
    >>> from scipy.optimize import rosen_hess_prod
    >>> X = 0.1 * np.arange(9)
    >>> p = 0.5 * np.arange(9)
    >>> rosen_hess_prod(X, p)
    array([  -0.,   27.,  -10.,  -95., -192., -265., -278., -195., -180.])

    """
    x = atleast_1d(x)
    Hp = np.zeros(len(x), dtype=x.dtype)
    Hp[0] = (1200 * x[0] ** 2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
    Hp[1:-1] = (-400 * x[:-2] * p[:-2] +
                (202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]) * p[1:-1] -
                400 * x[1:-1] * p[2:])
    Hp[-1] = -400 * x[-2] * p[-2] + 200 * p[-1]
    return Hp


def _wrap_function(function, args):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    # A copy of x is sent to the user function (gh13740)
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        ncalls[0] += 1
        return function(np.copy(x), *(wrapper_args + args))

    return ncalls, function_wrapper


def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None, initial_simplex=None):
    """
    Minimize a function using the downhill simplex algorithm.

    This algorithm only uses function values, not derivatives or second
    derivatives.

    Parameters
    ----------
    func : callable func(x,*args)
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func, i.e., ``f(x,*args)``.
    xtol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    ftol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : number, optional
        Maximum number of function evaluations to make.
    full_output : bool, optional
        Set to True if fopt and warnflag outputs are desired.
    disp : bool, optional
        Set to True to print convergence messages.
    retall : bool, optional
        Set to True to return list of solutions at each iteration.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    initial_simplex : array_like of shape (N + 1, N), optional
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.

    Returns
    -------
    xopt : ndarray
        Parameter that minimizes function.
    fopt : float
        Value of function at minimum: ``fopt = func(xopt)``.
    iter : int
        Number of iterations performed.
    funcalls : int
        Number of function calls made.
    warnflag : int
        1 : Maximum number of function evaluations made.
        2 : Maximum number of iterations reached.
    allvecs : list
        Solution at each iteration.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'Nelder-Mead' `method` in particular.

    Notes
    -----
    Uses a Nelder-Mead simplex algorithm to find the minimum of function of
    one or more variables.

    This algorithm has a long history of successful use in applications.
    But it will usually be slower than an algorithm that uses first or
    second derivative information. In practice, it can have poor
    performance in high-dimensional problems and is not robust to
    minimizing complicated functions. Additionally, there currently is no
    complete theory describing when the algorithm will successfully
    converge to the minimum, or how fast it will if it does. Both the ftol and
    xtol criteria must be met for convergence.

    Examples
    --------
    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fmin(f, 1)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 17
             Function evaluations: 34
    >>> minimum[0]
    -8.8817841970012523e-16

    References
    ----------
    .. [1] Nelder, J.A. and Mead, R. (1965), "A simplex method for function
           minimization", The Computer Journal, 7, pp. 308-313

    .. [2] Wright, M.H. (1996), "Direct Search Methods: Once Scorned, Now
           Respectable", in Numerical Analysis 1995, Proceedings of the
           1995 Dundee Biennial Conference in Numerical Analysis, D.F.
           Griffiths and G.A. Watson (Eds.), Addison Wesley Longman,
           Harlow, UK, pp. 191-208.

    """
    opts = {'xatol': xtol,
            'fatol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'return_all': retall,
            'initial_simplex': initial_simplex}

    res = _minimize_neldermead(func, x0, args, callback=callback, **opts)
    if full_output:
        retlist = res['x'], res['fun'], res['nit'], res['nfev'], res['status']
        if retall:
            retlist += (res['allvecs'],)
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_neldermead(func, x0, args=(), callback=None,
                         maxiter=None, maxfev=None, disp=False,
                         return_all=False, initial_simplex=None,
                         xatol=1e-4, fatol=1e-4, adaptive=False, bounds=None,
                         **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
    xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    adaptive : bool, optional
        Adapt algorithm parameters to dimensionality of problem. Useful for
        high-dimensional minimization [1]_.
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.

        Note that this just clips all vertices in simplex based on
        the bounds.

    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277

    """
    if 'ftol' in unknown_options:
        warnings.warn("ftol is deprecated for Nelder-Mead,"
                      " use fatol instead. If you specified both, only"
                      " fatol is used.",
                      DeprecationWarning)
        if (np.isclose(fatol, 1e-4) and
                not np.isclose(unknown_options['ftol'], 1e-4)):
            # only ftol was probably specified, use it.
            fatol = unknown_options['ftol']
        unknown_options.pop('ftol')
    if 'xtol' in unknown_options:
        warnings.warn("xtol is deprecated for Nelder-Mead,"
                      " use xatol instead. If you specified both, only"
                      " xatol is used.",
                      DeprecationWarning)
        if (np.isclose(xatol, 1e-4) and
                not np.isclose(unknown_options['xtol'], 1e-4)):
            # only xtol was probably specified, use it.
            xatol = unknown_options['xtol']
        unknown_options.pop('xtol')

    _check_unknown_options(unknown_options)
    maxfun = maxfev
    retall = return_all

    fcalls, func = _wrap_function(func, args)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2 / dim
        psi = 0.75 - 1 / (2 * dim)
        sigma = 1 - 1 / dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    x0 = asfarray(x0).flatten()

    if bounds is not None:
        lower_bound, upper_bound = bounds.lb, bounds.ub
        # check bounds
        if (lower_bound > upper_bound).any():
            raise ValueError("Nelder Mead - one of the lower bounds is greater than an upper bound.")
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds",
                          OptimizeWarning, 3)

    if bounds is not None:
        x0 = np.clip(x0, lower_bound, upper_bound)

    if initial_simplex is None:
        N = len(x0)

        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt) * y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.asfarray(initial_simplex).copy()
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    if retall:
        allvecs = [sim[0]]

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf

    if bounds is not None:
        sim = np.clip(sim, lower_bound, upper_bound)

    one2np1 = list(range(1, N + 1))
    fsim = np.empty((N + 1,), float)

    for k in range(N + 1):
        fsim[k] = func(sim[k])

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
            break

        xbar = np.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        if bounds is not None:
            xr = np.clip(xr, lower_bound, upper_bound)
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            if bounds is not None:
                xe = np.clip(xe, lower_bound, upper_bound)
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    if bounds is not None:
                        xc = np.clip(xc, lower_bound, upper_bound)
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    if bounds is not None:
                        xcc = np.clip(xcc, lower_bound, upper_bound)
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        if bounds is not None:
                            sim[j] = np.clip(sim[j], lower_bound, upper_bound)
                        fsim[j] = func(sim[j])

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print('Warning: ' + msg)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print('Warning: ' + msg)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x, final_simplex=(sim, fsim))
    if retall:
        result['allvecs'] = allvecs
    return result


def approx_fprime(xk, f, epsilon, *args):
    """Finite-difference approximation of the gradient of a scalar function.

    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the gradient of `f`.
    f : callable
        The function of which to determine the gradient (partial derivatives).
        Should take `xk` as first argument, other arguments to `f` can be
        supplied in ``*args``. Should return a scalar, the value of the
        function at `xk`.
    epsilon : array_like
        Increment to `xk` to use for determining the function gradient.
        If a scalar, uses the same finite difference delta for all partial
        derivatives. If an array, should contain one value per element of
        `xk`.
    \\*args : args, optional
        Any other arguments that are to be passed to `f`.

    Returns
    -------
    grad : ndarray
        The partial derivatives of `f` to `xk`.

    See Also
    --------
    check_grad : Check correctness of gradient function against approx_fprime.

    Notes
    -----
    The function gradient is determined by the forward finite difference
    formula::

                 f(xk[i] + epsilon[i]) - f(xk[i])
        f'[i] = ---------------------------------
                            epsilon[i]

    The main use of `approx_fprime` is in scalar function optimizers like
    `fmin_bfgs`, to determine numerically the Jacobian of a function.

    Examples
    --------
    >>> from scipy import optimize
    >>> def func(x, c0, c1):
    ...     "Coordinate vector `x` should be an array of size two."
    ...     return c0 * x[0]**2 + c1*x[1]**2

    >>> x = np.ones(2)
    >>> c0, c1 = (1, 200)
    >>> eps = np.sqrt(np.finfo(float).eps)
    >>> optimize.approx_fprime(x, func, [eps, np.sqrt(200) * eps], c0, c1)
    array([   2.        ,  400.00004198])

    """
    xk = np.asarray(xk, float)

    f0 = f(xk, *args)
    if not np.isscalar(f0):
        try:
            f0 = f0.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    return approx_derivative(f, xk, method='2-point', abs_step=epsilon,
                             args=args, f0=f0)


def check_grad(func, grad, x0, *args, **kwargs):
    """Check the correctness of a gradient function by comparing it against a
    (forward) finite-difference approximation of the gradient.

    Parameters
    ----------
    func : callable ``func(x0, *args)``
        Function whose derivative is to be checked.
    grad : callable ``grad(x0, *args)``
        Gradient of `func`.
    x0 : ndarray
        Points to check `grad` against forward difference approximation of grad
        using `func`.
    args : \\*args, optional
        Extra arguments passed to `func` and `grad`.
    epsilon : float, optional
        Step size used for the finite difference approximation. It defaults to
        ``sqrt(np.finfo(float).eps)``, which is approximately 1.49e-08.

    Returns
    -------
    err : float
        The square root of the sum of squares (i.e., the 2-norm) of the
        difference between ``grad(x0, *args)`` and the finite difference
        approximation of `grad` using func at the points `x0`.

    See Also
    --------
    approx_fprime

    Examples
    --------
    >>> def func(x):
    ...     return x[0]**2 - 0.5 * x[1]**3
    >>> def grad(x):
    ...     return [2 * x[0], -1.5 * x[1]**2]
    >>> from scipy.optimize import check_grad
    >>> check_grad(func, grad, [1.5, -1.5])
    2.9802322387695312e-08

    """
    step = kwargs.pop('epsilon', _epsilon)
    if kwargs:
        raise ValueError("Unknown keyword arguments: %r" %
                         (list(kwargs.keys()),))
    return sqrt(sum((grad(x0, *args) -
                     approx_fprime(x0, func, step, *args)) ** 2))


def approx_fhess_p(x0, p, fprime, epsilon, *args):
    # calculate fprime(x0) first, as this may be cached by ScalarFunction
    f1 = fprime(*((x0,) + args))
    f2 = fprime(*((x0 + epsilon * p,) + args))
    return (f2 - f1) / epsilon


class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found

    """

    extra_condition = kwargs.pop('extra_condition', None)

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            kwargs2 = {}
            for key in ('c1', 'c2', 'amax'):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval,
                                     extra_condition=extra_condition,
                                     **kwargs2)

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-5, norm=Inf,
              epsilon=_epsilon, maxiter=None, full_output=0, disp=1,
              retall=0, callback=None):
    """
    Minimize a function using the BFGS algorithm.

    Parameters
    ----------
    f : callable ``f(x,*args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable ``f'(x,*args)``, optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    gtol : float, optional
        Gradient norm must be less than `gtol` before successful termination.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray, optional
        If `fprime` is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration. Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True, return ``fopt``, ``func_calls``, ``grad_calls``, and
        ``warnflag`` in addition to ``xopt``.
    disp : bool, optional
        Print convergence message if True.
    retall : bool, optional
        Return a list of results at each iteration if True.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e., the inverse Hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
        3 : NaN result encountered.
    allvecs : list
        The value of `xopt` at each iteration. Only returned if `retall` is
        True.

    Notes
    -----
    Optimize the function, `f`, whose gradient is given by `fprime`
    using the quasi-Newton method of Broyden, Fletcher, Goldfarb,
    and Shanno (BFGS).

    See Also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See ``method='BFGS'`` in particular.

    References
    ----------
    Wright, and Nocedal 'Numerical Optimization', 1999, p. 198.

    Examples
    --------
    >>> from scipy.optimize import fmin_bfgs
    >>> def quadratic_cost(x, Q):
    ...     return x @ Q @ x
    ...
    >>> x0 = np.array([-3, -4])
    >>> cost_weight =  np.diag([1., 10.])
    >>> # Note that a trailing comma is necessary for a tuple with single element
    >>> fmin_bfgs(quadratic_cost, x0, args=(cost_weight,))
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 7                   # may vary
            Function evaluations: 24        # may vary
            Gradient evaluations: 8         # may vary
    array([ 2.85169950e-06, -4.61820139e-07])

    >>> def quadratic_cost_grad(x, Q):
    ...     return 2 * Q @ x
    ...
    >>> fmin_bfgs(quadratic_cost, x0, quadratic_cost_grad, args=(cost_weight,))
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 7
            Function evaluations: 8
            Gradient evaluations: 8
    array([ 2.85916637e-06, -4.54371951e-07])

    """
    opts = {'gtol': gtol,
            'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}

    res = _minimize_bfgs(f, x0, args, fprime, callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['jac'], res['hess_inv'],
                   res['nfev'], res['njev'], res['status'])
        if retall:
            retlist += (res['allvecs'],)
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def fmin_sr1(f, x0, fprime=None, args=(), gtol=1e-5, norm=Inf,
             epsilon=_epsilon, maxiter=None, full_output=0, disp=1,
             retall=0, callback=None):
    """
    Minimize a function using the BFGS algorithm.

    Parameters
    ----------
    f : callable ``f(x,*args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable ``f'(x,*args)``, optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    gtol : float, optional
        Gradient norm must be less than `gtol` before successful termination.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray, optional
        If `fprime` is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration. Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True, return ``fopt``, ``func_calls``, ``grad_calls``, and
        ``warnflag`` in addition to ``xopt``.
    disp : bool, optional
        Print convergence message if True.
    retall : bool, optional
        Return a list of results at each iteration if True.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e., the inverse Hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
        3 : NaN result encountered.
    allvecs : list
        The value of `xopt` at each iteration. Only returned if `retall` is
        True.

    Notes
    -----
    Optimize the function, `f`, whose gradient is given by `fprime`
    using the quasi-Newton method of Broyden, Fletcher, Goldfarb,
    and Shanno (BFGS).

    See Also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See ``method='BFGS'`` in particular.

    References
    ----------
    Wright, and Nocedal 'Numerical Optimization', 1999, p. 198.

    Examples
    --------
    >>> from scipy.optimize import fmin_bfgs
    >>> def quadratic_cost(x, Q):
    ...     return x @ Q @ x
    ...
    >>> x0 = np.array([-3, -4])
    >>> cost_weight =  np.diag([1., 10.])
    >>> # Note that a trailing comma is necessary for a tuple with single element
    >>> fmin_bfgs(quadratic_cost, x0, args=(cost_weight,))
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 7                   # may vary
            Function evaluations: 24        # may vary
            Gradient evaluations: 8         # may vary
    array([ 2.85169950e-06, -4.61820139e-07])

    >>> def quadratic_cost_grad(x, Q):
    ...     return 2 * Q @ x
    ...
    >>> fmin_bfgs(quadratic_cost, x0, quadratic_cost_grad, args=(cost_weight,))
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 7
            Function evaluations: 8
            Gradient evaluations: 8
    array([ 2.85916637e-06, -4.54371951e-07])

    """
    opts = {'gtol': gtol,
            'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}

    res = _minimize_sr1(f, x0, args, fprime, callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['jac'], res['hess_inv'],
                   res['nfev'], res['njev'], res['status'])
        if retall:
            retlist += (res['allvecs'],)
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, errHistory=None,
                   disp=False, return_all=False, finite_diff_rel_step=None,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        errHistory.append(old_fval)
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                     old_fval, old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remaines for more safety
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        else:
            rhok = 1. / rhok_inv

        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])

    fval = old_fval
    errHistory.append(old_fval)

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result



def _minimize_obfgs(fun, x0, args=(), jac=None, callback=None,
                     gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, err=[], timeplot=[],
                     disp=False, return_all=False, vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, mu=None,
                     dirNorm=True,Hk_mat=None,
                     **unknown_options):
    '''
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    '''

    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    N = len(x0)
    I = np.eye(N, dtype=int)

    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    import time
    start = time.time()
    k = len(Hk_mat)

    if k == 0:
        N = len(x0)
        I = np.eye(N, dtype=int)
        Hk = I
    else:
        Hk = Hk_mat[0]

    gfk = myfprime(xk)
    pk = -np.dot(Hk, gfk)



    if dirNorm == True:
        pk = pk / vecnorm(pk, 2)  # direction normalization

    sk = alpha_k[0] * pk
    xkp1 = xk + sk

    #sk_vec.append(sk)

    gfkp1 = myfprime(xkp1)
    yk = gfkp1 - gfk + sk
    #yk_vec.append(yk)

    xk = xkp1

    rhok_inv = np.dot(yk, sk)
    # this was handled in numeric, let it remaines for more safety
    if rhok_inv == 0.:
        rhok = 1000.0
        if disp:
            print("Divide-by-zero encountered: rhok assumed large")
    else:
        rhok = 1. / rhok_inv

    A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
    A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
    Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                       sk[np.newaxis, :])

    Hk_mat.append(Hk)

    end = time.time()
    timeplot.append(end - start)

    err.append(f(xk))
    if callback is not None:
        callback(xk)
    k += 1

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result


def _minimize_onaq(fun, x0, args=(), jac=None, callback=None,
                    gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, err=[], timeplot=[],
                    disp=False, return_all=False, vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, muk=None,
                    dirNorm=True, Hk_mat=None,
                    **unknown_options):
    '''
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    '''
    mu = muk[0]
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    N = len(x0)
    I = np.eye(N, dtype=int)

    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    vk = vk_vec[0]
    k = len(Hk_mat)

    import time
    start = time.time()
    if k == 0:
        print("Parameters: ", len(xk))
        N = len(x0)
        I = np.eye(N, dtype=int)
        Hk = I
    else:
        Hk = Hk_mat[0]


    gfk = myfprime(xk + mu * vk)
    pk = -np.dot(Hk, gfk)

    if dirNorm == True:
        pk = pk / vecnorm(pk, 2)  # direction normalization

    vkp1 = mu * vk + alpha_k[0] * pk
    xkp1 = xk + vkp1
    sk = xkp1 - (xk + mu * vk)
    vk_vec.append(vkp1)
    #sk_vec.append(sk)

    gfkp1 = myfprime(xkp1)
    yk = gfkp1 - gfk + sk
    #yk_vec.append(yk)
    xk = xkp1

    rhok_inv = np.dot(yk, sk)
    # this was handled in numeric, let it remaines for more safety
    if rhok_inv == 0.:
        rhok = 1000.0
        if disp:
            print("Divide-by-zero encountered: rhok assumed large")
    else:
        rhok = 1. / rhok_inv

    A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
    A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
    Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                       sk[np.newaxis, :])

    Hk_mat.append(Hk)

    end = time.time()

    if callback is not None:
        callback(xk)
    k += 1
    timeplot.append(end - start)
    err.append(f(xk))

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result


def _minimize_omoq(fun, x0, args=(), jac=None, callback=None,
                    gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, err=[], timeplot=[],
                    disp=False, return_all=False, vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, muk=None,
                    dirNorm=True, gfk_vec=None, Hk_mat=None,
                    **unknown_options):
    '''
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    '''
    mu = muk[0]
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    N = len(x0)
    I = np.eye(N, dtype=int)

    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    vk = vk_vec[0]
    k = len(Hk_mat)
    import time
    start = time.time()
    if k == 0:
        print("Parameters: ", len(xk))
        grad_val = myfprime(xk + mu * vk)
        gfk_vec.append(grad_val)
        gfk_vec.append(grad_val)
        N = len(x0)
        I = np.eye(N, dtype=int)
        Hk = I
    else:
        Hk = Hk_mat[0]

    # curr_grad = myfprime(xk)
    gfk = (1 + mu) * gfk_vec[1] - mu * gfk_vec[0]
    # gfk = curr_grad + mu * (gfk_vec[1] - gfk_vec[0])

    pk = -np.dot(Hk, gfk)


    if dirNorm == True:
        pk = pk / vecnorm(pk, 2)  # direction normalization

    vkp1 = mu * vk + alpha_k[0] * pk
    xkp1 = xk + vkp1
    sk = xkp1 - (xk + mu * vk)
    vk_vec.append(vkp1)
    #sk_vec.append(sk)

    gfkp1 = myfprime(xkp1)
    gfk_vec.append(gfkp1)
    yk = gfkp1 - gfk + sk
    #yk_vec.append(yk)

    rhok_inv = np.dot(yk, sk)
    # this was handled in numeric, let it remaines for more safety
    if rhok_inv == 0.:
        rhok = 1000.0
        if disp:
            print("Divide-by-zero encountered: rhok assumed large")
    else:
        rhok = 1. / rhok_inv

    A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
    A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
    Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                       sk[np.newaxis, :])

    Hk_mat.append(Hk)

    xk = xkp1

    end = time.time()
    timeplot.append(end - start)

    if callback is not None:
        callback(xk)
    k += 1
    err.append(f(xk))

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result



def _minimize_lbfgs(fun, x0, args=(), jac=None, callback=None, errPlot=[], timePlot=[], evalPlot=[], LS=[], GEV=[], m=10,
                   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, mu=0.8, sk_vec=None, yk_vec=None,
                   disp=False, return_all=False, finite_diff_rel_step=None, etol=None, nevs=None,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    import time
    # start_time = time.time()
    # timePlot.append(0)
    # theta_k = 1
    # LS = []
    # GEV = []
    TLR = []
    GC = []

    old_fval = f(x0)
    #gfk = myfprime(x0)

    #GEV.append(time.time())
    gfk = myfprime(x0)
    #GEV[-1] = time.time() - GEV[-1]

    errPlot.append(old_fval)
    timePlot.append(0)
    evalPlot.append(0)
    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    #vk = np.zeros_like(x0)
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)

    #tot_start_time = time.time()
    tot_start_time = time.time()
    while (gnorm > gtol) and (k < maxiter):
        if etol != None and old_fval < etol : break
        start_time = time.time()#time.time()
        """
        theta_kp1 = ((1e-5 - (theta_k * theta_k)) + np.sqrt(
            ((1e-5 - (theta_k * theta_k)) * (1e-5 - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
        mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), 0)
        theta_k = theta_kp1
        xmuv = xk + mu * vk
        """
        #gfk = myfprime(xmuv)
        TLR.append(time.time())
        pk = -gfk
        a = []
        idx = min(k, m)
        for i in range(min(k, m)):
            a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
            pk = pk - a[i] * yk_vec[idx - 1 - i]
        if k > 0:
            term = 0
            for i in range(min(k, m)):
                term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                           yk_vec[idx - 1 - i]))
            pk = pk * term / idx
        else:
            pk = 1e-10 * pk
        for i in reversed(range(min(k, m))):
            b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
            pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

        pknorm = vecnorm(pk, ord=norm)
        if pknorm > 1000:
            delta = 1e-7
        else:
            delta = 1e-4
        TLR[-1] = time.time() - TLR[-1]
        LS.append(time.time())
        LHS = f(xk + pk)
        old_old_fval = f(xk)
        gfk_times_pk = np.dot(gfk.T, pk)
        RHS = old_old_fval + 1e-3 * gfk_times_pk
        alpha_k = 1
        LS[-1] = time.time() - LS[-1]

        for line1 in range(10):
            if LHS < RHS :
                old_fval = LHS
                break
            LS.append(time.time())
            alpha_k *= 0.5
            LHS = f(xk + alpha_k * pk)
            RHS = old_old_fval + 1e-3 * alpha_k * gfk_times_pk
            LS[-1] = time.time()-LS[-1]
        sk = alpha_k * pk
        #vkp1 = mu * vk + alpha_k * pk
        #vk = mu * vk + sk
        #xkp1 = xk + vkp1
        xk = xk + sk

        # if retall:
        #    allvecs.append(xkp1)
        #sk = xkp1 - (xk + mu * vk)

        #gfkp1 = myfprime(xkp1)
        GEV.append(time.time())
        gfkp1 = myfprime(xk)
        GEV[-1] = time.time() - GEV[-1]

        GC.append(time.time())
        yk = gfkp1 - gfk
        # Global convergence
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        yk = yk + zeta * gnorm * sk
        GC[-1] = time.time() - GC[-1]
        sk_vec.append(sk)
        yk_vec.append(yk)
        gfk = gfkp1
        #xk = xkp1
        #vk = vkp1

        # if callback is not None:
        #    callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        # if (gnorm <= gtol):
        #    break

        # if not np.isfinite(old_fval):
        # We correctly found +-Inf as optimal value, or something went
        # wrong.
        #    warnflag = 2
        #    break
        end_time = time.time()

        timePlot.append(end_time - start_time)
        errPlot.append(old_fval)
        evalPlot.append(sf.nfev + sf.ngev)
    fval = old_fval
    #tot_end_time = time.time()
    tot_end_time = time.time()
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Algorithm: %s" % "lbfgs")
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Linesearch evaluations: %d" % len(LS))
        print("         Total evaluations: %d" % (sf.ngev + sf.nfev))
        print("         Avg Time per Iteration: %f" % np.mean(timePlot))
        print("         Avg Time per FEV: %f" % np.mean(LS))
        print("         Avg Time per GEV: %f" % np.mean(GEV))
        print("         Avg Time per GC: %f" % np.mean(GC))
        print("         Avg Time per TLR: %f" % np.mean(TLR))
        print("         Time Taken: %f" % (tot_end_time-tot_start_time))

    nevs.append(fval)
    nevs.append(k)
    nevs.append(sf.nfev)
    nevs.append(sf.ngev)
    nevs.append(tot_end_time-tot_start_time)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result



def _minimize_lnaq(fun, x0, args=(), jac=None, callback=None, errPlot=[], timePlot=[], evalPlot=[],  LS=[], GEV=[],m=10,
                   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, mu=0.8, sk_vec=None, yk_vec=None,
                   disp=False, return_all=False, finite_diff_rel_step=None, etol=None, nevs=None,mu_clip=0.95,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)
    errPlot.append(old_fval)
    timePlot.append(0)
    evalPlot.append(0)
    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    vk = np.zeros_like(x0)
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    import time
    # start_time = time.time()
    # timePlot.append(0)
    theta_k = 1
    #LS = []
    TLR = []
    #GEV = []
    GC = []
    MC = []
    #tot_start_time = time.time()
    tot_start_time = time.time()
    while (gnorm > gtol) and (k < maxiter):
        if etol != None and old_fval < etol : break
        start_time = time.time()#time.time()
        MC.append(time.time())
        theta_kp1 = ((1e-5 - (theta_k * theta_k)) + np.sqrt(
            ((1e-5 - (theta_k * theta_k)) * (1e-5 - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
        mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), mu_clip)
        theta_k = theta_kp1
        MC[-1] = time.time() - MC[-1]
        #mu = 0
        xmuv = xk + mu * vk

        GEV.append(time.time())
        gfk = myfprime(xmuv)
        GEV[-1] = time.time() - GEV[-1]

        TLR.append(time.time())
        pk = -gfk
        a = []
        idx = min(k, m)
        for i in range(min(k, m)):
            a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
            pk = pk - a[i] * yk_vec[idx - 1 - i]
        if k > 0:
            term = 0
            for i in range(min(k, m)):
                term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                           yk_vec[idx - 1 - i]))
            pk = pk * term / idx
        else:
            pk = 1e-10 * pk
        for i in reversed(range(min(k, m))):
            b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
            pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

        pknorm = vecnorm(pk, ord=norm)
        if pknorm > 1000:
            delta = 1e-7
        else:
            delta = 1e-4
        TLR[-1] = time.time() - TLR[-1]
        LS.append(time.time())
        LHS = f(xmuv + pk)
        old_old_fval = f(xmuv)
        gfk_times_pk = np.dot(gfk.T, pk)
        RHS = old_old_fval + 1e-3 * gfk_times_pk
        alpha_k = 1
        LS[-1] = time.time() - LS[-1]
        #numLS = 1
        for line1 in range(10):
            if LHS < RHS:
                old_fval = LHS
                break
            #numLS += 1
            LS.append(time.time())
            alpha_k *= 0.5
            LHS = f(xmuv + alpha_k * pk)
            RHS = old_old_fval + 1e-3 * alpha_k * gfk_times_pk
            LS[-1] = time.time() - LS[-1]
        #nLS.append(numLS)
        sk = alpha_k * pk
        #vkp1 = mu * vk + alpha_k * pk
        vk = mu * vk + sk
        #xkp1 = xk + vkp1
        xk = xk + vk

        # if retall:
        #    allvecs.append(xkp1)
        #sk = xkp1 - (xk + mu * vk)

        #gfkp1 = myfprime(xkp1)
        GEV.append(time.time())
        gfkp1 = myfprime(xk)
        GEV[-1] = time.time() - GEV[-1]

        GC.append(time.time())
        yk = gfkp1 - gfk
        # Global convergence
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        yk = yk + zeta * gnorm * sk
        GC[-1] = time.time() - GC[-1]
        sk_vec.append(sk)
        yk_vec.append(yk)
        gfk = gfkp1
        #xk = xkp1
        #vk = vkp1

        # if callback is not None:
        #    callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        # if (gnorm <= gtol):
        #    break

        # if not np.isfinite(old_fval):
        # We correctly found +-Inf as optimal value, or something went
        # wrong.
        #    warnflag = 2
        #    break
        end_time = time.time()

        timePlot.append(end_time - start_time)
        errPlot.append(old_fval)
        evalPlot.append(sf.nfev + sf.ngev)
    fval = old_fval
    #tot_end_time = time.time()
    tot_end_time = time.time()
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Algorithm: %s" % "lnaq")
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Linesearch evaluations: %d" % len(LS))
        print("         Total evaluations: %d" % (sf.ngev + sf.nfev))
        print("         Avg Time per Iteration: %f" % np.mean(timePlot))
        print("         Avg Time per FEV: %f" % np.mean(LS))
        print("         Avg Time per GEV: %f" % np.mean(GEV))
        print("         Avg Time per GC: %f" % np.mean(GC))
        print("         Avg Time per TLR: %f" % np.mean(TLR))
        print("         Avg Time per MC: %f" % np.mean(MC))
        print("         Time Taken: %f" % (tot_end_time - tot_start_time))

    nevs.append(fval)
    nevs.append(k)
    nevs.append(sf.nfev)
    nevs.append(sf.ngev)
    nevs.append(tot_end_time-tot_start_time)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result

def _minimize_lmoq(fun, x0, args=(), jac=None, callback=None, errPlot=[], timePlot=[], m=10, evalPlot=[],  LS=[], GEV=[],nevs=None,
                   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, mu=0.8, sk_vec=None, yk_vec=None,
                   disp=False, return_all=False, finite_diff_rel_step=None, etol=None,mu_clip=0.95,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)
    timePlot.append(0)
    evalPlot.append(0)
    errPlot.append(old_fval)
    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    vk = np.zeros_like(x0)
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    import time

    # timePlot.append(0)
    gfkm1 = gfk
    theta_k = 1
    #LS = []
    TLR = []
    #GEV = []
    GC = []
    MC = []
    #tot_start_time = time.time()
    tot_start_time = time.time()
    while (gnorm > gtol) and (k < maxiter) :
        if etol != None and old_fval < etol : break
        start_time = time.time()
        MC.append(time.time())
        theta_kp1 = ((1e-5 - (theta_k * theta_k)) + np.sqrt(
            ((1e-5 - (theta_k * theta_k)) * (1e-5 - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
        mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), mu_clip)
        theta_k = theta_kp1
        MC[-1] = time.time() - MC[-1]
        #mu = 0
        # if k >0:
        xmuv = xk + mu * vk
        #ogfk = myfprime(xmuv)
        agfk = (1 + mu) * gfk - mu * gfkm1
        # else: agfk=gfk
        TLR.append(time.time())
        pk = -agfk
        a = []
        idx = min(k, m)
        for i in range(min(k, m)):
            a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
            pk = pk - a[i] * yk_vec[idx - 1 - i]
        if k > 0:
            term = 0
            for i in range(min(k, m)):
                term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                           yk_vec[idx - 1 - i]))
            pk = pk * term / idx
        else:
            pk = 1e-10 * pk
        for i in reversed(range(min(k, m))):
            b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
            pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

        pknorm = vecnorm(pk, ord=norm)
        if pknorm > 1000:
            delta = 1e-7
        else:
            delta = 1e-4
        TLR[-1] = time.time() - TLR[-1]

        #gfkp1 = myfprime(xmuv)
        LS.append(time.time())
        LHS = f(xmuv + pk)
        old_old_fval = f(xmuv)
        agfk_times_pk = np.dot(agfk.T, pk)
        RHS = old_old_fval + 1e-3 * agfk_times_pk
        alpha_k = 1
        LS[-1] = time.time() - LS[-1]
        for line1 in range(10):
            if LHS < RHS :
                old_fval = LHS
                break
            LS.append(time.time())
            alpha_k *= 0.5
            LHS = f(xmuv + alpha_k * pk)
            RHS = old_old_fval + 1e-3 * alpha_k * agfk_times_pk
            LS[-1] = time.time()-LS[-1]
        # vkp1 = mu * vk + alpha_k * pk
        # xkp1 = xk + vkp1
        sk = alpha_k * pk
        vk = mu * vk + sk
        xk = xk + vk

        # if retall:
        #    allvecs.append(xkp1)
        # sk = xkp1 - (xk + mu * vk)
        gfkm1 = gfk
        # gfkp1 = myfprime(xkp1)
        GEV.append(time.time())
        gfk = myfprime(xk)
        GEV[-1] = time.time() - GEV[-1]

        GC.append(time.time())
        yk = gfk - agfk
        # yk = gfkp1 - agfk
        # Global convergence
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        yk = yk + zeta * gnorm * sk
        GC[-1] = time.time() - GC[-1]
        sk_vec.append(sk)
        yk_vec.append(yk)
        # gfkm1 = gfk
        # gfk = gfkp1
        # xk = xkp1
        # vk = vkp1

        # if callback is not None:
        #    callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        # if (gnorm <= gtol):
        #    break

        # if not np.isfinite(old_fval):
        # We correctly found +-Inf as optimal value, or something went
        # wrong.
        #    warnflag = 2
        #    break

        end_time = time.time()
        timePlot.append(end_time - start_time)
        errPlot.append(old_fval)
        evalPlot.append(sf.nfev + sf.ngev)
    fval = old_fval
    #tot_end_time = time.time()
    tot_end_time = time.time()
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Algorithm: %s" % "lmoq")
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Linesearch evaluations: %d" % len(LS))
        print("         Total evaluations: %d" % (sf.ngev + sf.nfev))
        print("         Avg Time per Iteration: %f" % np.mean(timePlot))
        print("         Avg Time per FEV: %f" % np.mean(LS))
        print("         Avg Time per GEV: %f" % np.mean(GEV))
        print("         Avg Time per GC: %f" % np.mean(GC))
        print("         Avg Time per TLR: %f" % np.mean(TLR))
        print("         Avg Time per MC: %f" % np.mean(MC))
        print("         Time Taken: %f" % (tot_end_time - tot_start_time))

    nevs.append(fval)
    nevs.append(k)
    nevs.append(sf.nfev)
    nevs.append(sf.ngev)
    nevs.append(tot_end_time-tot_start_time)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result



def _minimize_olbfgs(fun, x0, args=(), jac=None, callback=None,
                     gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, err=[], timeplot=[],
                     disp=False, return_all=False, vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, mu=None,
                     dirNorm=True,
                     **unknown_options):
    '''
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    '''

    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    import time
    start = time.time()
    k = len(sk_vec)

    gfk = myfprime(xk)
    pk = -gfk

    # two loop recursivef
    a = []
    idx = min(k, m)
    for i in range(min(k, m)):
        a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
        pk = pk - a[i] * yk_vec[idx - 1 - i]
    if k > 0:
        term = 0
        for i in range(min(k, m)):
            term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                       yk_vec[idx - 1 - i]))
        pk = pk * term / idx
    else:
        pk = 1e-10 * pk
    for i in reversed(range(min(k, m))):
        b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
        pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

    if dirNorm == True:
        pk = pk / vecnorm(pk, 2)  # direction normalization

    sk = alpha_k[0] * pk
    xkp1 = xk + sk

    sk_vec.append(sk)

    gfkp1 = myfprime(xkp1)
    yk = gfkp1 - gfk + sk
    yk_vec.append(yk)

    xk = xkp1

    end = time.time()
    timeplot.append(end - start)

    err.append(f(xk))
    if callback is not None:
        callback(xk)
    k += 1

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result

def _minimize_olbfgs1(fun, x0, args=(), jac=None, callback=None,
                     gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, err=[], timeplot=[],
                     disp=False, return_all=False, vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, mu=None,
                     dirNorm=True,gfk_vec=None,
                     **unknown_options):
    '''
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    '''

    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    import time
    start = time.time()
    k = len(sk_vec)


    if k>0:
        gfk = gfk_vec[-1]
    else:
        gfk = myfprime(xk)

    pk = -gfk

    # two loop recursivef
    a = []
    idx = min(k, m)
    for i in range(min(k, m)):
        a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
        pk = pk - a[i] * yk_vec[idx - 1 - i]
    if k > 0:
        term = 0
        for i in range(min(k, m)):
            term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                       yk_vec[idx - 1 - i]))
        pk = pk * term / idx
    else:
        pk = 1e-10 * pk
    for i in reversed(range(min(k, m))):
        b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
        pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

    if dirNorm == True:
        pk = pk / vecnorm(pk, 2)  # direction normalization

    sk = alpha_k[0] * pk
    xkp1 = xk + sk

    sk_vec.append(sk)

    gfkp1 = myfprime(xkp1)
    yk = gfkp1 - gfk + sk
    yk_vec.append(yk)
    gfk_vec.append(gfkp1)

    xk = xkp1

    end = time.time()
    timeplot.append(end - start)

    err.append(f(xk))
    if callback is not None:
        callback(xk)
    k += 1

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result

def _minimize_aSNAQ(fun, x0, args=(), jac=None, callback=None,
                     gtol=1e-5, norm=Inf, eps=1e-4, maxiter=None,
                     disp=False, return_all=False, wo_bar_vec=None, ws_vec=None, vo_bar_vec=None, vs_vec=None,
                     vk_vec=None, L=5,err=None,
                     mu_val=None, mu_fac=1.01, mu_init=0.1, mu_clip=0.99, clearF=True, reset=False, dirNorm=True,
                     iter=None, alpha_k=[1.0], sk_vec=None, yk_vec=None, F=None, t_vec=None, gamma=1.01, old_fun_val=None,
                     memF=None, memL=None, timeLapse=[],
                     **unknown_options):
    """
    Bk = minibatch
    |Bk| = b batch size
    L = 5 memory size chosen from (2,5,10,20)
    alpha = ?
    k = iteration count
    mL = 10
    mF = 100
    eps =1e-4
    gamma = 1.01
    """
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    wk = x0.reshape(-1, 1)

    t = t_vec[0]
    k = iter[0]
    # eps = 1e-4
    # gamma = 1.01
    N = len(wk)

    if k == 0:
        wo_bar = np.zeros_like(wk)
        vo_bar = np.zeros_like(wk)
        ws = np.zeros_like(wk)
        vs = np.zeros_like(wk)
        vk = np.zeros_like(wk)
        mu = mu_val[0]

    else:
        wo_bar = wo_bar_vec[0]  # np.zeros_like(wk)
        vo_bar = vo_bar_vec[0]  # np.zeros_like(wk)
        ws = ws_vec[0]  # 0
        vs = vs_vec[0]  # 0
        mu = mu_val[0]  # 0
        vk = vk_vec[0]  # 0

    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    gfk = myfprime(wk + mu * vk).reshape(-1, 1)

    if k == 0: F.append(gfk)
    # two loop recursion

    q = gfk
    tau = len(sk_vec)
    a = np.zeros(tau)
    for i in reversed(range(tau)):
        rho = 1 / np.dot(yk_vec[i].T, sk_vec[i])
        a[i] = rho * np.dot(sk_vec[i].T, q)
        q = q - np.dot(a[i], yk_vec[i])
    term = np.sum(np.square(F), 0)
    Hk0 = 1 / np.sqrt(term + eps)
    r = Hk0 * q
    for i in range(tau):
        rho = 1 / np.dot(yk_vec[i].T, sk_vec[i])
        beta = rho * np.dot(yk_vec[i].T, r)
        r = r + sk_vec[i] * (a[i] - beta)
    pk = r
    if vecnorm(pk, 2) == np.inf or vecnorm(pk, 2) == np.nan:
        pk = np.ones_like(wk)

    elif dirNorm:
        pk = pk / vecnorm(pk, 2)  # Exploding gradients (direction normalization)

    if k == 0: F.clear()
    '''
    pk = -gfk
    a = []

    idx = len(sk_vec)
    for i in range(len(sk_vec)):
        a.append(numpy.dot(sk_vec[idx - 1 - i].T, pk) / numpy.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
        pk = pk - a[i] * yk_vec[idx - 1 - i]

    term = np.sum(np.square(F), 0)
    Hk0 = 1 / np.sqrt(term + eps)
    pk = Hk0 * pk
    for i in reversed(range(len(sk_vec))):
        b = numpy.dot(yk_vec[idx - 1 - i].T, pk) / numpy.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
        pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]
    '''

    flag_ret = 1

    vk = mu * vk - alpha_k[0] * pk
    wk = wk + vk

    ws = ws + wk  # +mu*vk
    vs = vs + vk

    gfkp1 = myfprime(wk).reshape(-1, 1)
    F.append(gfkp1)

    if k % L == 0:
        wn_bar = ws / L
        vn_bar = vs / L
        ws = np.zeros_like(wk)
        vs = np.zeros_like(wk)
        if t > 0:
            if f(wn_bar) > gamma * f(wo_bar):
                sk_vec.clear()
                yk_vec.clear()
                mu = np.minimum(mu / mu_fac, mu_clip)
                mu = np.maximum(mu, mu_init)
                if clearF: F.clear()
                #print("Clearing buffers")
                wk = wo_bar
                vk = vo_bar
                flag_ret = 0
            if flag_ret:
                sk = wn_bar - wo_bar
                fisher = np.asarray(F)[:, :, 0].T
                yk = np.dot(fisher, np.dot(fisher.T, sk))
                mu = np.minimum(mu * mu_fac, mu_clip)
                # yk = (np.sum(fisher, 1, keepdims=True) * sk) / shape(fisher)[-1]
                # yk = 0
                # for i in F:
                #    yk += np.dot(i,np.dot(i.T,sk))
                # yk = yk/len(F)
                if np.dot(sk.T, yk) > eps * np.dot(yk.T, yk):
                    sk_vec.append(sk)
                    yk_vec.append(yk)
                    wo_bar = wn_bar
                    vo_bar = vn_bar
        else:
            wo_bar = wn_bar
            vo_bar = vn_bar
        t += 1
        t_vec.append(t)

    if callback is not None:
        callback(wk)
    k += 1
    iter.append(k)
    mu_val.append(mu)
    wo_bar_vec.append(wo_bar)  # np.zeros_like(wk)
    vo_bar_vec.append(vo_bar)  # np.zeros_like(wk)
    ws_vec.append(ws)  # 0
    vs_vec.append(vs)  # 0
    vk_vec.append(vk)  # 0
    #memL.append(len(sk_vec))
    #memF.append(len(F))
    #err.append(f(wk))

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=wk,
                            nit=k)

    return result

def _minimize_olnaq(fun, x0, args=(), jac=None, callback=None,
                    gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, err=[], timeplot=[],
                    disp=False, return_all=False, vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, muk=None,
                    dirNorm=True,
                    **unknown_options):
    '''
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    '''
    mu = muk[0]
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    vk = vk_vec[0]
    k = len(sk_vec)

    import time
    start = time.time()
    if k == 0:
        print("Parameters: ", len(xk))

    gfk = myfprime(xk + mu * vk)
    pk = -gfk
    a = []
    idx = min(k, m)
    for i in range(min(k, m)):
        a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
        pk = pk - a[i] * yk_vec[idx - 1 - i]
    if k > 0:
        term = 0
        for i in range(min(k, m)):
            term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                       yk_vec[idx - 1 - i]))
        pk = pk * term / idx
    else:
        pk = 1e-10 * pk
    for i in reversed(range(min(k, m))):
        b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
        pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

    if dirNorm == True:
        pk = pk / vecnorm(pk, 2)  # direction normalization

    vkp1 = mu * vk + alpha_k[0] * pk
    xkp1 = xk + vkp1
    sk = xkp1 - (xk + mu * vk)
    vk_vec.append(vkp1)
    sk_vec.append(sk)

    gfkp1 = myfprime(xkp1)
    yk = gfkp1 - gfk + sk
    yk_vec.append(yk)
    xk = xkp1
    end = time.time()

    if callback is not None:
        callback(xk)
    k += 1
    timeplot.append(end - start)
    err.append(f(xk))

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result


def _minimize_olmoq(fun, x0, args=(), jac=None, callback=None,
                    gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, err=[], timeplot=[],
                    disp=False, return_all=False, vk_vec=None, sk_vec=None, yk_vec=None, m=8, alpha_k=1.0, muk=None,
                    dirNorm=True, gfk_vec=None,
                    **unknown_options):
    '''
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    '''
    mu = muk[0]
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    vk = vk_vec[0]
    k = len(sk_vec)
    import time
    start = time.time()
    if k == 0:
        print("Parameters: ", len(xk))
        grad_val = myfprime(xk + mu * vk)
        gfk_vec.append(grad_val)
        gfk_vec.append(grad_val)

    # curr_grad = myfprime(xk)
    gfk = (1 + mu) * gfk_vec[1] - mu * gfk_vec[0]
    # gfk = curr_grad + mu * (gfk_vec[1] - gfk_vec[0])

    pk = -gfk
    a = []
    idx = min(k, m)
    for i in range(min(k, m)):
        a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
        pk = pk - a[i] * yk_vec[idx - 1 - i]
    if k > 0:
        term = 0
        for i in range(min(k, m)):
            term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                       yk_vec[idx - 1 - i]))
        pk = pk * term / idx
    else:
        pk = 1e-10 * pk
    for i in reversed(range(min(k, m))):
        b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
        pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

    if dirNorm == True:
        pk = pk / vecnorm(pk, 2)  # direction normalization

    vkp1 = mu * vk + alpha_k[0] * pk
    xkp1 = xk + vkp1
    sk = xkp1 - (xk + mu * vk)
    vk_vec.append(vkp1)
    sk_vec.append(sk)

    gfkp1 = myfprime(xkp1)
    gfk_vec.append(gfkp1)
    yk = gfkp1 - gfk + sk
    yk_vec.append(yk)
    xk = xkp1

    end = time.time()
    timeplot.append(end - start)

    if callback is not None:
        callback(xk)
    k += 1
    err.append(f(xk))

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result

def rootFinder(a,b,c):
  """return the root of (a * x^2) + b*x + c =0"""
  r = b**2 - 4*a*c

  if r > 0:
      num_roots = 2
      x1 = ((-b) + np.sqrt(r))/(2*a+0.0)
      x2 = ((-b) - np.sqrt(r))/(2*a+0.0)
      x = max(x1,x2)
      if x>=0:
        return x
      else:
        print("no positive root!")
  elif r == 0:
      num_roots = 1
      x = (-b) / (2*a+0.0)
      if x>=0:
        return x
      else:
        print("no positive root!")
  else:
      print("No roots")

def CG_Steinhaug_matFree(epsTR, g, deltak, S, Y, nv):
    """
    The following function is used for sloving the trust region subproblem
    by utilizing "CG_Steinhaug" algorithm discussed in
    Nocedal, J., & Wright, S. J. (2006). Nonlinear Equations (pp. 270-302). Springer New York.;
    moreover, for Hessian-free implementation, we used the compact form of Hessian
    approximation discussed in Byrd, Richard H., Jorge Nocedal, and Robert B. Schnabel.
    "Representations of quasi-Newton matrices and their use in limited memory methods."
    Mathematical Programming 63.1-3 (1994): 129-156
    """
    from numpy import linalg as LA

    zOld = np.zeros((nv, 1))
    rOld = g
    dOld = -g
    trsLoop = 1e-12
    if LA.norm(rOld) < epsTR:
        return zOld
    flag = True
    pk = np.zeros((nv, 1))

    # for Hessfree
    L = np.zeros((Y.shape[1], Y.shape[1]))
    for ii in range(Y.shape[1]):
        for jj in range(0, ii):
            L[ii, jj] = S[:, ii].dot(Y[:, jj])

    tmp = np.sum((S * Y), axis=0)

    D = np.diag(tmp)
    M = (D + L + L.T)
    Minv = np.linalg.inv(M)

    while flag:

        ################
        tmp1 = np.matmul(Y.T, dOld)
        tmp2 = np.matmul(Minv, tmp1)
        Bk_d = np.matmul(Y, tmp2)

        ################

        if dOld.T.dot(Bk_d) < trsLoop:
            tau = rootFinder(LA.norm(dOld) ** 2, 2 * zOld.T.dot(dOld), (LA.norm(zOld) ** 2 - deltak ** 2))
            pk = zOld + tau * dOld
            flag = False
            break
        alphaj = rOld.T.dot(rOld) / (dOld.T.dot(Bk_d))
        zNew = zOld + alphaj * dOld

        if LA.norm(zNew) >= deltak:
            tau = rootFinder(LA.norm(dOld) ** 2, 2 * zOld.T.dot(dOld), (LA.norm(zOld) ** 2 - deltak ** 2))
            pk = zOld + tau * dOld
            flag = False
            break
        rNew = rOld + alphaj * Bk_d

        if LA.norm(rNew) < epsTR:
            pk = zNew
            flag = False
            break
        betajplus1 = rNew.T.dot(rNew) / (rOld.T.dot(rOld))
        dNew = -rNew + betajplus1 * dOld

        zOld = zNew
        dOld = dNew
        rOld = rNew
    return pk


def sample_pairs_SY_SLSR1(X, y, num_weights, mmr, radius, eps, dnn, numHessEval, sess):
    """ Function that computes SY pairs for S-LSR1 method"""

    Stemp = radius * np.random.randn(num_weights, mmr)
    Ytemp = np.squeeze(sess.run([dnn.Hvs], feed_dict={dnn.x: X, dnn.y: y, dnn.vecs: Stemp})).T
    numHessEval += 1
    S = np.zeros((num_weights, 0))
    Y = np.zeros((num_weights, 0))

    counterSucc = 0
    for idx in range(mmr):

        L = np.zeros((Y.shape[1], Y.shape[1]))
        for ii in range(Y.shape[1]):
            for jj in range(0, ii):
                L[ii, jj] = S[:, ii].dot(Y[:, jj])

        tmp = np.sum((S * Y), axis=0)
        D = np.diag(tmp)
        M = (D + L + L.T)
        Minv = np.linalg.inv(M)

        tmp1 = np.matmul(Y.T, Stemp[:, idx])
        tmp2 = np.matmul(Minv, tmp1)
        Bksk = np.squeeze(np.matmul(Y, tmp2))
        yk_BkskDotsk = (Ytemp[:, idx] - Bksk).T.dot(Stemp[:, idx])
        if np.abs(np.squeeze(yk_BkskDotsk)) > (
                eps * (LA.norm(Ytemp[:, idx] - Bksk) * LA.norm(Stemp[:, idx]))):
            counterSucc += 1

            S = np.append(S, Stemp[:, idx].reshape(num_weights, 1), axis=1)
            Y = np.append(Y, Ytemp[:, idx].reshape(num_weights, 1), axis=1)

    return S, Y, counterSucc, numHessEval



def _minimize_trlbfgs(fun, x0, args=(), jac=None, callback=None, errPlot=[], timePlot=[], evalPlot=[], LS=[], GEV=[], m=10,
                   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, mu=0.8, sk_vec=None, yk_vec=None,
                   disp=False, return_all=False, finite_diff_rel_step=None, etol=None, nevs=None,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    import time
    # start_time = time.time()
    # timePlot.append(0)
    # theta_k = 1
    # LS = []
    # GEV = []
    TLR = []
    GC = []

    old_fval = f(x0)
    #gfk = myfprime(x0)

    #GEV.append(time.time())
    gfk = myfprime(x0)
    #GEV[-1] = time.time() - GEV[-1]

    errPlot.append(old_fval)
    timePlot.append(0)
    evalPlot.append(0)
    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    #vk = np.zeros_like(x0)
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)

    #tot_start_time = time.time()
    tot_start_time = time.time()
    while (gnorm > gtol) and (k < maxiter):
        if etol != None and old_fval < etol : break
        start_time = time.time()#time.time()
        """
        theta_kp1 = ((1e-5 - (theta_k * theta_k)) + np.sqrt(
            ((1e-5 - (theta_k * theta_k)) * (1e-5 - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
        mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), 0)
        theta_k = theta_kp1
        xmuv = xk + mu * vk
        """
        #gfk = myfprime(xmuv)
        TLR.append(time.time())
        pk = -gfk
        a = []
        idx = min(k, m)
        for i in range(min(k, m)):
            a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
            pk = pk - a[i] * yk_vec[idx - 1 - i]
        if k > 0:
            term = 0
            for i in range(min(k, m)):
                term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                           yk_vec[idx - 1 - i]))
            pk = pk * term / idx
        else:
            pk = 1e-10 * pk
        for i in reversed(range(min(k, m))):
            b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
            pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

        pknorm = vecnorm(pk, ord=norm)
        if pknorm > 1000:
            delta = 1e-7
        else:
            delta = 1e-4
        TLR[-1] = time.time() - TLR[-1]
        LS.append(time.time())
        LHS = f(xk + pk)
        old_old_fval = f(xk)
        gfk_times_pk = np.dot(gfk.T, pk)
        RHS = old_old_fval + 1e-3 * gfk_times_pk
        alpha_k = 1
        LS[-1] = time.time() - LS[-1]

        for line1 in range(10):
            if LHS < RHS :
                old_fval = LHS
                break
            LS.append(time.time())
            alpha_k *= 0.5
            LHS = f(xk + alpha_k * pk)
            RHS = old_old_fval + 1e-3 * alpha_k * gfk_times_pk
            LS[-1] = time.time()-LS[-1]
        sk = alpha_k * pk
        #vkp1 = mu * vk + alpha_k * pk
        #vk = mu * vk + sk
        #xkp1 = xk + vkp1
        xk = xk + sk

        # if retall:
        #    allvecs.append(xkp1)
        #sk = xkp1 - (xk + mu * vk)

        #gfkp1 = myfprime(xkp1)
        GEV.append(time.time())
        gfkp1 = myfprime(xk)
        GEV[-1] = time.time() - GEV[-1]

        GC.append(time.time())
        yk = gfkp1 - gfk
        # Global convergence
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        yk = yk + zeta * gnorm * sk
        GC[-1] = time.time() - GC[-1]
        sk_vec.append(sk)
        yk_vec.append(yk)
        gfk = gfkp1
        #xk = xkp1
        #vk = vkp1

        # if callback is not None:
        #    callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        # if (gnorm <= gtol):
        #    break

        # if not np.isfinite(old_fval):
        # We correctly found +-Inf as optimal value, or something went
        # wrong.
        #    warnflag = 2
        #    break
        end_time = time.time()

        timePlot.append(end_time - start_time)
        errPlot.append(old_fval)
        evalPlot.append(sf.nfev + sf.ngev)
    fval = old_fval
    #tot_end_time = time.time()
    tot_end_time = time.time()
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Algorithm: %s" % "lbfgs")
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Linesearch evaluations: %d" % len(LS))
        print("         Total evaluations: %d" % (sf.ngev + sf.nfev))
        print("         Avg Time per Iteration: %f" % np.mean(timePlot))
        print("         Avg Time per FEV: %f" % np.mean(LS))
        print("         Avg Time per GEV: %f" % np.mean(GEV))
        print("         Avg Time per GC: %f" % np.mean(GC))
        print("         Avg Time per TLR: %f" % np.mean(TLR))
        print("         Time Taken: %f" % (tot_end_time-tot_start_time))

    nevs.append(fval)
    nevs.append(k)
    nevs.append(sf.nfev)
    nevs.append(sf.ngev)
    nevs.append(tot_end_time-tot_start_time)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result



def _minimize_trlnaq(fun, x0, args=(), jac=None, callback=None, errPlot=[], timePlot=[], evalPlot=[],  LS=[], GEV=[],m=10,
                   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, mu=0.8, sk_vec=None, yk_vec=None,
                   disp=False, return_all=False, finite_diff_rel_step=None, etol=None, nevs=None,mu_clip=0.95,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)
    errPlot.append(old_fval)
    timePlot.append(0)
    evalPlot.append(0)
    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    vk = np.zeros_like(x0)
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    import time
    # start_time = time.time()
    # timePlot.append(0)
    theta_k = 1
    #LS = []
    TLR = []
    #GEV = []
    GC = []
    MC = []
    #tot_start_time = time.time()
    tot_start_time = time.time()
    while (gnorm > gtol) and (k < maxiter):
        if etol != None and old_fval < etol : break
        start_time = time.time()#time.time()
        MC.append(time.time())
        theta_kp1 = ((1e-5 - (theta_k * theta_k)) + np.sqrt(
            ((1e-5 - (theta_k * theta_k)) * (1e-5 - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
        mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), mu_clip)
        theta_k = theta_kp1
        MC[-1] = time.time() - MC[-1]
        #mu = 0
        xmuv = xk + mu * vk

        GEV.append(time.time())
        gfk = myfprime(xmuv)
        GEV[-1] = time.time() - GEV[-1]

        TLR.append(time.time())
        pk = -gfk
        a = []
        idx = min(k, m)
        for i in range(min(k, m)):
            a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
            pk = pk - a[i] * yk_vec[idx - 1 - i]
        if k > 0:
            term = 0
            for i in range(min(k, m)):
                term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                           yk_vec[idx - 1 - i]))
            pk = pk * term / idx
        else:
            pk = 1e-10 * pk
        for i in reversed(range(min(k, m))):
            b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
            pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

        pknorm = vecnorm(pk, ord=norm)
        if pknorm > 1000:
            delta = 1e-7
        else:
            delta = 1e-4
        TLR[-1] = time.time() - TLR[-1]
        LS.append(time.time())
        LHS = f(xmuv + pk)
        old_old_fval = f(xmuv)
        gfk_times_pk = np.dot(gfk.T, pk)
        RHS = old_old_fval + 1e-3 * gfk_times_pk
        alpha_k = 1
        LS[-1] = time.time() - LS[-1]
        #numLS = 1
        for line1 in range(10):
            if LHS < RHS:
                old_fval = LHS
                break
            #numLS += 1
            LS.append(time.time())
            alpha_k *= 0.5
            LHS = f(xmuv + alpha_k * pk)
            RHS = old_old_fval + 1e-3 * alpha_k * gfk_times_pk
            LS[-1] = time.time() - LS[-1]
        #nLS.append(numLS)
        sk = alpha_k * pk
        #vkp1 = mu * vk + alpha_k * pk
        vk = mu * vk + sk
        #xkp1 = xk + vkp1
        xk = xk + vk

        # if retall:
        #    allvecs.append(xkp1)
        #sk = xkp1 - (xk + mu * vk)

        #gfkp1 = myfprime(xkp1)
        GEV.append(time.time())
        gfkp1 = myfprime(xk)
        GEV[-1] = time.time() - GEV[-1]

        GC.append(time.time())
        yk = gfkp1 - gfk
        # Global convergence
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        yk = yk + zeta * gnorm * sk
        GC[-1] = time.time() - GC[-1]
        sk_vec.append(sk)
        yk_vec.append(yk)
        gfk = gfkp1
        #xk = xkp1
        #vk = vkp1

        # if callback is not None:
        #    callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        # if (gnorm <= gtol):
        #    break

        # if not np.isfinite(old_fval):
        # We correctly found +-Inf as optimal value, or something went
        # wrong.
        #    warnflag = 2
        #    break
        end_time = time.time()

        timePlot.append(end_time - start_time)
        errPlot.append(old_fval)
        evalPlot.append(sf.nfev + sf.ngev)
    fval = old_fval
    #tot_end_time = time.time()
    tot_end_time = time.time()
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Algorithm: %s" % "lnaq")
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Linesearch evaluations: %d" % len(LS))
        print("         Total evaluations: %d" % (sf.ngev + sf.nfev))
        print("         Avg Time per Iteration: %f" % np.mean(timePlot))
        print("         Avg Time per FEV: %f" % np.mean(LS))
        print("         Avg Time per GEV: %f" % np.mean(GEV))
        print("         Avg Time per GC: %f" % np.mean(GC))
        print("         Avg Time per TLR: %f" % np.mean(TLR))
        print("         Avg Time per MC: %f" % np.mean(MC))
        print("         Time Taken: %f" % (tot_end_time - tot_start_time))

    nevs.append(fval)
    nevs.append(k)
    nevs.append(sf.nfev)
    nevs.append(sf.ngev)
    nevs.append(tot_end_time-tot_start_time)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result

def _minimize_trlmoq(fun, x0, args=(), jac=None, callback=None, errPlot=[], timePlot=[], m=10, evalPlot=[],  LS=[], GEV=[],nevs=None,
                   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, mu=0.8, sk_vec=None, yk_vec=None,
                   disp=False, return_all=False, finite_diff_rel_step=None, etol=None,mu_clip=0.95,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)
    timePlot.append(0)
    evalPlot.append(0)
    errPlot.append(old_fval)
    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    vk = np.zeros_like(x0)
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    import time

    # timePlot.append(0)
    gfkm1 = gfk
    theta_k = 1
    #LS = []
    TLR = []
    #GEV = []
    GC = []
    MC = []
    #tot_start_time = time.time()
    tot_start_time = time.time()
    while (gnorm > gtol) and (k < maxiter) :
        if etol != None and old_fval < etol : break
        start_time = time.time()
        MC.append(time.time())
        theta_kp1 = ((1e-5 - (theta_k * theta_k)) + np.sqrt(
            ((1e-5 - (theta_k * theta_k)) * (1e-5 - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
        mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), mu_clip)
        theta_k = theta_kp1
        MC[-1] = time.time() - MC[-1]
        #mu = 0
        # if k >0:
        xmuv = xk + mu * vk
        #ogfk = myfprime(xmuv)
        agfk = (1 + mu) * gfk - mu * gfkm1
        # else: agfk=gfk
        TLR.append(time.time())
        pk = -agfk
        a = []
        idx = min(k, m)
        for i in range(min(k, m)):
            a.append(np.dot(sk_vec[idx - 1 - i].T, pk) / np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
            pk = pk - a[i] * yk_vec[idx - 1 - i]
        if k > 0:
            term = 0
            for i in range(min(k, m)):
                term = term + (np.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]) / np.dot(yk_vec[idx - 1 - i].T,
                                                                                           yk_vec[idx - 1 - i]))
            pk = pk * term / idx
        else:
            pk = 1e-10 * pk
        for i in reversed(range(min(k, m))):
            b = np.dot(yk_vec[idx - 1 - i].T, pk) / np.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
            pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]

        pknorm = vecnorm(pk, ord=norm)
        if pknorm > 1000:
            delta = 1e-7
        else:
            delta = 1e-4
        TLR[-1] = time.time() - TLR[-1]

        #gfkp1 = myfprime(xmuv)
        LS.append(time.time())
        LHS = f(xmuv + pk)
        old_old_fval = f(xmuv)
        agfk_times_pk = np.dot(agfk.T, pk)
        RHS = old_old_fval + 1e-3 * agfk_times_pk
        alpha_k = 1
        LS[-1] = time.time() - LS[-1]
        for line1 in range(10):
            if LHS < RHS :
                old_fval = LHS
                break
            LS.append(time.time())
            alpha_k *= 0.5
            LHS = f(xmuv + alpha_k * pk)
            RHS = old_old_fval + 1e-3 * alpha_k * agfk_times_pk
            LS[-1] = time.time()-LS[-1]
        # vkp1 = mu * vk + alpha_k * pk
        # xkp1 = xk + vkp1
        sk = alpha_k * pk
        vk = mu * vk + sk
        xk = xk + vk

        # if retall:
        #    allvecs.append(xkp1)
        # sk = xkp1 - (xk + mu * vk)
        gfkm1 = gfk
        # gfkp1 = myfprime(xkp1)
        GEV.append(time.time())
        gfk = myfprime(xk)
        GEV[-1] = time.time() - GEV[-1]

        GC.append(time.time())
        yk = gfk - agfk
        # yk = gfkp1 - agfk
        # Global convergence
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        yk = yk + zeta * gnorm * sk
        GC[-1] = time.time() - GC[-1]
        sk_vec.append(sk)
        yk_vec.append(yk)
        # gfkm1 = gfk
        # gfk = gfkp1
        # xk = xkp1
        # vk = vkp1

        # if callback is not None:
        #    callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        # if (gnorm <= gtol):
        #    break

        # if not np.isfinite(old_fval):
        # We correctly found +-Inf as optimal value, or something went
        # wrong.
        #    warnflag = 2
        #    break

        end_time = time.time()
        timePlot.append(end_time - start_time)
        errPlot.append(old_fval)
        evalPlot.append(sf.nfev + sf.ngev)
    fval = old_fval
    #tot_end_time = time.time()
    tot_end_time = time.time()
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Algorithm: %s" % "lmoq")
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Linesearch evaluations: %d" % len(LS))
        print("         Total evaluations: %d" % (sf.ngev + sf.nfev))
        print("         Avg Time per Iteration: %f" % np.mean(timePlot))
        print("         Avg Time per FEV: %f" % np.mean(LS))
        print("         Avg Time per GEV: %f" % np.mean(GEV))
        print("         Avg Time per GC: %f" % np.mean(GC))
        print("         Avg Time per TLR: %f" % np.mean(TLR))
        print("         Avg Time per MC: %f" % np.mean(MC))
        print("         Time Taken: %f" % (tot_end_time - tot_start_time))

    nevs.append(fval)
    nevs.append(k)
    nevs.append(sf.nfev)
    nevs.append(sf.ngev)
    nevs.append(tot_end_time-tot_start_time)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def _minimize_mosr1(fun, x0, args=(), jac=None, callback=None,timePlot=[],seed=100,
                  gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, errHistory=None, muHist=None,
                  disp=False, return_all=False, finite_diff_rel_step=None,m=10,etol=None,
                  **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    print("Parameters ", len(x0))
    import collections
    from numpy import linalg as LA
    S = collections.deque(maxlen=m)
    Y = collections.deque(maxlen=m)
    s_vec_tmp = collections.deque(maxlen=m)
    y_vec_tmp = collections.deque(maxlen=m)


    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0).reshape(-1,1)
    gfkm1 = gfk

    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    num_weights = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    deltak = 1
    eta = 1e-6
    epsTR = 1e-10

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0.reshape(-1,1)
    vk = np.zeros_like(xk)
    theta_k = 1
    gamma = 1e-5
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    count = 0
    import time
    timePlot.append(0)
    errHistory.append(old_fval)
    while (gnorm > gtol) and (k < maxiter):
        if etol != None and old_fval < etol: break
        #print(old_fval)
        start_time = time.time()
        k += 1

        """if count > 1:
            count = 0
            theta_k = 1"""

        theta_kp1 = ((gamma - (theta_k * theta_k)) + np.sqrt(
            ((gamma - (theta_k * theta_k)) * (gamma - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
        #mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), 0.8)
        mu = (theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1)
        muHist.append(mu)
        #mu = 0#.85#(theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1)
        theta_k = theta_kp1

        #print("k ", k, " fval ", old_fval, " mu ", mu)
        #errHistory.append(old_fval)

        #gfk = myfprime(xk+mu*vk).reshape(-1, 1)
        agfk = (1 + mu) * gfk - mu * gfkm1

        if k==1:
            #comment later
            np.random.seed(seed)
            Stemp = np.random.randn(N, m)

            for index in range(m):
                y_vec_tmp.append(myfprime(xk + 1 * Stemp[:, index].reshape(-1, 1)).reshape(-1, 1))
                s_vec_tmp.append(xk + 1 * Stemp[:, index].reshape(-1, 1))

            Stemp = np.squeeze(np.asarray(s_vec_tmp)).T
            Ytemp = np.squeeze(np.asarray(y_vec_tmp)).T

            S = np.zeros((num_weights, 0))
            Y = np.zeros((num_weights, 0))

            counterSucc = 0
            for idx in range(m):

                L = np.zeros((Y.shape[1], Y.shape[1]))
                for ii in range(Y.shape[1]):
                    for jj in range(0, ii):
                        L[ii, jj] = S[:, ii].dot(Y[:, jj])

                tmp = np.sum((S * Y), axis=0)
                D = np.diag(tmp)
                M = (D + L + L.T)
                Minv = np.linalg.inv(M)

                tmp1 = np.matmul(Y.T, Stemp[:, idx])
                tmp2 = np.matmul(Minv, tmp1)
                Bksk = np.squeeze(np.matmul(Y, tmp2))
                yk_BkskDotsk = (Ytemp[:, idx] - Bksk).T.dot(Stemp[:, idx])
                if np.abs(np.squeeze(yk_BkskDotsk)) > (
                        eps * (LA.norm(Ytemp[:, idx] - Bksk) * LA.norm(Stemp[:, idx]))):
                    counterSucc += 1

                    S = np.append(S, Stemp[:, idx].reshape(num_weights, 1), axis=1)
                    Y = np.append(Y, Ytemp[:, idx].reshape(num_weights, 1), axis=1)

        #S = np.squeeze(np.asarray(s_vec)).T
        #Y = np.squeeze(np.asarray(y_vec)).T

        sk_TR = CG_Steinhaug_matFree(epsTR, agfk, deltak, S, Y, N)
        #sk_TR =  -np.dot(Hk, gfk) * deltak

        new_fval = f(xk+mu*vk+sk_TR)
        ared = old_fval - new_fval  # Compute actual reduction

        Lp = np.zeros((Y.shape[1], Y.shape[1]))
        for ii in range(Y.shape[1]):
            for jj in range(0, ii):
                Lp[ii, jj] = S[:, ii].dot(Y[:, jj])
        tmpp = np.sum((S * Y), axis=0)
        Dp = np.diag(tmpp)
        Mp = (Dp + Lp + Lp.T)
        Minvp = np.linalg.inv(Mp)
        tmpp1 = np.matmul(Y.T, sk_TR)
        tmpp2 = np.matmul(Minvp, tmpp1)
        Bk_skTR = np.matmul(Y, tmpp2)
        #Bk_skTR = np.dot(Hk, sk_TR)
        pred = -(agfk.T.dot(sk_TR) + 0.5 * sk_TR.T.dot(Bk_skTR))  # Compute predicted reduction

        # Update trust region radius
        if ared / pred > 0.75:
            deltak = 2 * deltak
        elif ared / pred >= 0.1 and ared / pred <= 0.75:
            pass  # no need to change deltak
        elif ared / pred < 0.1:
            deltak = deltak * 0.5

        # Take step
        if ared / pred > eta:
            #count = 0

            xkp1 = xk +mu*vk + sk_TR
            vkp1 = mu*vk + sk_TR
            old_fval = f(xkp1)

        else:
            #count += 1
            theta_k = 1
            xkp1 = xk
            vkp1 = vk
            #end_time = time.time()
            #timePlot.append(end_time - start_time)
            #errHistory.append(old_fval)
            #continue



        """
        try:
            alpha_k = 1
            LHS = f(xk + pk)
            old_old_fval = f(xk)
            gfk_times_pk = np.dot(gfk.T, pk)
            RHS = old_old_fval + 1e-3 * gfk_times_pk
            for line1 in range(20):
                if LHS < RHS :
                    old_fval = LHS
                    print("LineSearch satisfied")
                    break
                alpha_k /= 10
                LHS = f(xk + alpha_k * pk)
                RHS = old_old_fval + 1e-3 * alpha_k * gfk_times_pk
            '''alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100, amax=1e100)'''
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break
        """

        #xkp1 = xk + alpha_k * pk

        if retall:
            allvecs.append(xkp1)
        sk = sk_TR#xkp1 - xk
        xk = xkp1
        vk = vkp1
        # if gfkp1 is None:
        gfkm2 = gfkm1
        gfkm1 = gfk
        gfk = myfprime(xkp1).reshape(-1,1)

        yk = gfk - agfk


        # Global Convergence Term
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        #yk = yk + zeta * gnorm * sk
        yk = yk +  0.1 * sk

        #gfk = gfkp1

        #if abs(np.dot(sk.T, (yk - Bk_skTR))) >= 1e-8 * vecnorm(sk)*vecnorm(yk - Bk_skTR):
        yk_BkskDotsk = (yk - Bk_skTR).T.dot(sk)
        if np.abs(np.squeeze(yk_BkskDotsk)) > (eps * (LA.norm(yk - Bksk) * LA.norm(sk))):
            #s_vec.append(sk)
            #y_vec.append(yk)
            S = np.append(S, sk, axis=1)
            Y = np.append(Y, yk, axis=1)
            S = S[:, -10:]
            Y = Y[:,-10:]


        if callback is not None:
            callback(xk)

        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break
        end_time = time.time()
        timePlot.append(end_time-start_time)
        errHistory.append(old_fval)


        """rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remaines for more safety
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        else:
            rhok = 1. / rhok_inv
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])
        """

        """A1 = sk - np.dot(Hk, yk)
        # if A1.all()!=0:
        num = np.dot(A1, A1.T)
        den = np.dot(A1.T, yk)
        if den != 0:
            Hk = Hk + num / den"""


    fval = old_fval
    #errHistory.append(old_fval)

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result

def _minimize_omosr1(fun, x0, args=(), jac=None, callback=None,timePlot=[],seed=100, thetak=None,delta_k=None,
                  gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, errHistory=None,m=10,vk_vec=None,res=[], muHist=[],
                  disp=False, return_all=False, finite_diff_rel_step=None,s_vec=None, y_vec=None, gfk_vec=None,
                  **unknown_options):
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    import time
    from numpy import linalg as LA
    start = time.time()

    k = len(s_vec)
    N = len(x0)
    num_weights = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    eta = 1e-6
    epsTR = 1e-10
    gamma = 1e-5
    xk = x0.reshape(-1, 1)

    if k == 0:
        deltak = 1
        theta_k = 1
        delta_k.append(deltak)
        old_fval = f(xk)
        gfk = myfprime(xk).reshape(-1, 1)
        gfk_vec.append(gfk)
        gfk_vec.append(gfk)
        import time
        timePlot.append(0)
        errHistory.append(old_fval)
        vk = np.zeros_like(xk)

    else:
        deltak = delta_k[0]
        gfk = gfk_vec[-1]
        vk = vk_vec[0]
        theta_k = thetak[-1]
        old_fval = errHistory[-1]

    # Sets the initial step guess to dx ~ 1
    # old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)

    flag = 1
    start_time = time.time()

    theta_kp1 = ((gamma - (theta_k * theta_k)) + np.sqrt(
        ((gamma - (theta_k * theta_k)) * (gamma - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
    mu = (theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1)
    muHist.append(mu)
    theta_k = theta_kp1
    # thetak.append(theta_k)

    #gfk = myfprime(xk + mu * vk).reshape(-1, 1)
    agfk = (1 + mu) * gfk_vec[-1] - mu * gfk_vec[-2]

    if k == 0:
        # comment later
        np.random.seed(seed)
        Stemp = np.random.randn(N, m)
        y_vec_tmp = []
        s_vec_tmp = []

        for index in range(m):
            y_vec_tmp.append(myfprime(xk + 1 * Stemp[:, index].reshape(-1, 1)).reshape(-1, 1))
            s_vec_tmp.append(xk + 1 * Stemp[:, index].reshape(-1, 1))

        Stemp = np.squeeze(np.asarray(s_vec_tmp)).T
        Ytemp = np.squeeze(np.asarray(y_vec_tmp)).T

        S = np.zeros((num_weights, 0))
        Y = np.zeros((num_weights, 0))

        counterSucc = 0
        for idx in range(m):

            L = np.zeros((Y.shape[1], Y.shape[1]))
            for ii in range(Y.shape[1]):
                for jj in range(0, ii):
                    L[ii, jj] = S[:, ii].dot(Y[:, jj])

            tmp = np.sum((S * Y), axis=0)
            D = np.diag(tmp)
            M = (D + L + L.T)
            Minv = np.linalg.inv(M)

            tmp1 = np.matmul(Y.T, Stemp[:, idx])
            tmp2 = np.matmul(Minv, tmp1)
            Bksk = np.squeeze(np.matmul(Y, tmp2))
            yk_BkskDotsk = (Ytemp[:, idx] - Bksk).T.dot(Stemp[:, idx])
            if np.abs(np.squeeze(yk_BkskDotsk)) > (
                    eps * (LA.norm(Ytemp[:, idx] - Bksk) * LA.norm(Stemp[:, idx]))):
                counterSucc += 1

                S = np.append(S, Stemp[:, idx].reshape(num_weights, 1), axis=1)
                Y = np.append(Y, Ytemp[:, idx].reshape(num_weights, 1), axis=1)
                y_vec.append(Ytemp[:, idx].reshape(num_weights, 1))
                s_vec.append(Stemp[:, idx].reshape(num_weights, 1))

    S = np.squeeze(np.asarray(s_vec)).T
    Y = np.squeeze(np.asarray(y_vec)).T

    try:
        sk_TR = CG_Steinhaug_matFree(epsTR, agfk, deltak, S, Y, N)
    except:
        print("reset mu")
        gfk = myfprime(xk).reshape(-1, 1)
        theta_k = 1
        mu = 0
        muHist.pop()
        muHist.append(mu)
        sk_TR = CG_Steinhaug_matFree(epsTR, gfk, deltak, S, Y, N)

    # sk_TR =  -np.dot(Hk, gfk) * deltak

    new_fval = f(xk + mu * vk + sk_TR)
    ared = old_fval - new_fval  # Compute actual reduction

    Lp = np.zeros((Y.shape[1], Y.shape[1]))
    for ii in range(Y.shape[1]):
        for jj in range(0, ii):
            Lp[ii, jj] = S[:, ii].dot(Y[:, jj])
    tmpp = np.sum((S * Y), axis=0)
    Dp = np.diag(tmpp)
    Mp = (Dp + Lp + Lp.T)
    Minvp = np.linalg.inv(Mp)
    tmpp1 = np.matmul(Y.T, sk_TR)
    tmpp2 = np.matmul(Minvp, tmpp1)
    Bk_skTR = np.matmul(Y, tmpp2)
    # Bk_skTR = np.dot(Hk, sk_TR)
    pred = -(agfk.T.dot(sk_TR) + 0.5 * sk_TR.T.dot(Bk_skTR))  # Compute predicted reduction

    # Update trust region radius
    if ared / pred > 0.75:
        deltak = 2 * deltak
    elif ared / pred >= 0.1 and ared / pred <= 0.75:
        # theta_k = thetak[-1]
        pass  # no need to change deltak
    elif ared / pred < 0.1:
        deltak = deltak * 0.5
        # theta_k = theta_k * 0.5

    delta_k.append(deltak)

    # Take step
    if ared / pred > eta:
        # count = 0
        xkp1 = xk + mu * vk + sk_TR
        vkp1 = mu * vk + sk_TR
        old_fval = f(xkp1)

    else:
        # count += 1
        # theta_k = 1
        # theta_k = thetak[-1]
        # thetak.append(theta_k)
        theta_k = 0.5
        if deltak < 1e-5:
            theta_k = 0.1

        xkp1 = xk
        vkp1 = vk
        flag = 0

    thetak.append(theta_k)

    if True:#flag:

        if retall:
            allvecs.append(xkp1)
        sk = sk_TR  # xkp1 - xk
        xk = xkp1
        vk = vkp1

        # if gfkp1 is None:
        gfkp1 = myfprime(xkp1).reshape(-1, 1)

        gnorm = vecnorm(gfkp1, ord=norm)

        yk = gfkp1 - agfk

        # Global Convergence Term
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const

        yk = yk + zeta * gnorm * sk

        gfk = gfkp1
        gfk_vec.append(gfkp1)

        if abs(np.dot(sk.T, (yk - Bk_skTR))) >= 1e-8 * vecnorm(sk)*vecnorm(yk - Bk_skTR):
            s_vec.append(sk)
            y_vec.append(yk)
            vk_vec.append(vk)

    end = time.time()
    timePlot.append(end - start)
    errHistory.append(f(xk))
    if callback is not None:
        callback(xk)

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result


def _minimize_sr1n(fun, x0, args=(), jac=None, callback=None,timePlot=[],seed=100,
                  gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, errHistory=None,m=10, muHist=[],
                  disp=False, return_all=False, finite_diff_rel_step=None,etol=None,
                  **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    import collections
    from numpy import linalg as LA
    s_vec_tmp = collections.deque(maxlen=m)
    y_vec_tmp = collections.deque(maxlen=m)


    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0).reshape(-1,1)


    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    num_weights = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    deltak = 1
    eta = 1e-6
    epsTR = 1e-10

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0.reshape(-1,1)
    vk = np.zeros_like(xk)
    theta_k = 1
    gamma = 1e-5
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    count = 0
    import time
    timePlot.append(0)
    errHistory.append(old_fval)

    while (gnorm > gtol) and (k < maxiter):
        if etol != None and old_fval < etol: break
        #print(old_fval)
        start_time = time.time()
        k += 1

        """if count > 1:
            count = 0
            theta_k = 1"""

        theta_kp1 = ((gamma - (theta_k * theta_k)) + np.sqrt(
            ((gamma - (theta_k * theta_k)) * (gamma - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
        #mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), 0.8)
        mu = (theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1)
        muHist.append(mu)
        # mu = 0.6#(theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1)
        theta_k = theta_kp1

        #print("k ", k, " fval ", old_fval, " mu ", mu)


        gfk = myfprime(xk+mu*vk).reshape(-1, 1)

        if k == 1:
            # comment later
            np.random.seed(seed)
            Stemp = np.random.randn(N, m)

            for index in range(m):
                y_vec_tmp.append(myfprime(xk + 1 * Stemp[:, index].reshape(-1, 1)).reshape(-1, 1))
                s_vec_tmp.append(xk + 1 * Stemp[:, index].reshape(-1, 1))

            Stemp = np.squeeze(np.asarray(s_vec_tmp)).T
            Ytemp = np.squeeze(np.asarray(y_vec_tmp)).T

            S = np.zeros((num_weights, 0))
            Y = np.zeros((num_weights, 0))

            counterSucc = 0
            for idx in range(m):

                L = np.zeros((Y.shape[1], Y.shape[1]))
                for ii in range(Y.shape[1]):
                    for jj in range(0, ii):
                        L[ii, jj] = S[:, ii].dot(Y[:, jj])

                tmp = np.sum((S * Y), axis=0)
                D = np.diag(tmp)
                M = (D + L + L.T)
                Minv = np.linalg.inv(M)

                tmp1 = np.matmul(Y.T, Stemp[:, idx])
                tmp2 = np.matmul(Minv, tmp1)
                Bksk = np.squeeze(np.matmul(Y, tmp2))
                yk_BkskDotsk = (Ytemp[:, idx] - Bksk).T.dot(Stemp[:, idx])
                if np.abs(np.squeeze(yk_BkskDotsk)) > (
                        eps * (LA.norm(Ytemp[:, idx] - Bksk) * LA.norm(Stemp[:, idx]))):
                    counterSucc += 1

                    S = np.append(S, Stemp[:, idx].reshape(num_weights, 1), axis=1)
                    Y = np.append(Y, Ytemp[:, idx].reshape(num_weights, 1), axis=1)

        sk_TR = CG_Steinhaug_matFree(epsTR, gfk, deltak, S, Y, N)
        #sk_TR =  -np.dot(Hk, gfk) * deltak

        new_fval = f(xk+mu*vk+sk_TR)
        ared = old_fval - new_fval  # Compute actual reduction

        Lp = np.zeros((Y.shape[1], Y.shape[1]))
        for ii in range(Y.shape[1]):
            for jj in range(0, ii):
                Lp[ii, jj] = S[:, ii].dot(Y[:, jj])
        tmpp = np.sum((S * Y), axis=0)
        Dp = np.diag(tmpp)
        Mp = (Dp + Lp + Lp.T)
        Minvp = np.linalg.inv(Mp)
        tmpp1 = np.matmul(Y.T, sk_TR)
        tmpp2 = np.matmul(Minvp, tmpp1)
        Bk_skTR = np.matmul(Y, tmpp2)
        #Bk_skTR = np.dot(Hk, sk_TR)
        pred = -(gfk.T.dot(sk_TR) + 0.5 * sk_TR.T.dot(Bk_skTR))  # Compute predicted reduction

        # Update trust region radius
        if ared / pred > 0.75:
            deltak = 2 * deltak
        elif ared / pred >= 0.1 and ared / pred <= 0.75:
            pass  # no need to change deltak
        elif ared / pred < 0.1:
            deltak = deltak * 0.5

        # Take step
        if ared / pred > eta:
            count += 1
            xkp1 = xk +mu*vk + sk_TR
            vkp1 = mu*vk + sk_TR
            old_fval = f(xkp1)

        else:
            #count += 1
            theta_k = 1
            xkp1 = xk
            vkp1 = vk
            end_time = time.time()
            timePlot.append(end_time - start_time)
            errHistory.append(old_fval)
            continue



        """
        try:
            alpha_k = 1
            LHS = f(xk + pk)
            old_old_fval = f(xk)
            gfk_times_pk = np.dot(gfk.T, pk)
            RHS = old_old_fval + 1e-3 * gfk_times_pk
            for line1 in range(20):
                if LHS < RHS :
                    old_fval = LHS
                    print("LineSearch satisfied")
                    break
                alpha_k /= 10
                LHS = f(xk + alpha_k * pk)
                RHS = old_old_fval + 1e-3 * alpha_k * gfk_times_pk
            '''alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100, amax=1e100)'''
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break
        """

        #xkp1 = xk + alpha_k * pk

        if retall:
            allvecs.append(xkp1)
        sk = sk_TR#xkp1 - xk
        xk = xkp1
        vk = vkp1
        # if gfkp1 is None:
        gfkp1 = myfprime(xkp1).reshape(-1,1)

        yk = gfkp1 - gfk


        # Global Convergence Term
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        yk = yk + zeta * gnorm * sk
        #yk = yk + sk

        gfk = gfkp1

        yk_BkskDotsk = (yk - Bk_skTR).T.dot(sk)
        if np.abs(np.squeeze(yk_BkskDotsk)) > (eps * (LA.norm(yk - Bksk) * LA.norm(sk))):
            # s_vec.append(sk)
            # y_vec.append(yk)
            S = np.append(S, sk, axis=1)
            Y = np.append(Y, yk, axis=1)
            S = S[:, -10:]
            Y = Y[:, -10:]

        if callback is not None:
            callback(xk)

        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        end_time = time.time()
        timePlot.append(end_time - start_time)
        errHistory.append(old_fval)

        """rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remaines for more safety
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        else:
            rhok = 1. / rhok_inv
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])
        """

        """A1 = sk - np.dot(Hk, yk)
        # if A1.all()!=0:
        num = np.dot(A1, A1.T)
        den = np.dot(A1.T, yk)
        if den != 0:
            Hk = Hk + num / den"""


    fval = old_fval
    #errHistory.append(old_fval)

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Successful updates  : %d" % count)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def _minimize_osr1n(fun, x0, args=(), jac=None, callback=None,timePlot=[],seed=100,mu_val=[],
                  gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, errHistory=None,m=10, delta_k=None, muHist=[],
                  disp=False, return_all=False, finite_diff_rel_step=None,etol=None,s_vec=None,y_vec=None,vk_vec=None,gfk_vec=None,thetak=None,
                  **unknown_options):
    """
    
    
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    """
    
    
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    import time
    from numpy import linalg as LA
    start = time.time()

    k = len(s_vec)
    N = len(x0)
    num_weights = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    eta = 1e-6
    epsTR = 1e-10
    gamma = 1e-5
    xk = x0.reshape(-1, 1)


    if k == 0:
        deltak = 1
        theta_k = 1
        delta_k.append(deltak)
        old_fval = f(xk)
        gfk = myfprime(xk).reshape(-1, 1)
        import time
        timePlot.append(0)
        errHistory.append(old_fval)
        vk = np.zeros_like(xk)
        
    else:
        deltak = delta_k[0]
        gfk = gfk_vec[0]
        vk = vk_vec[0]
        theta_k = thetak[-1]
        old_fval = errHistory[-1]


    # Sets the initial step guess to dx ~ 1
    # old_old_fval = old_fval + np.linalg.norm(gfk) / 2


    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)

    flag = 1
    start_time = time.time()
    
    theta_kp1 = ((gamma - (theta_k * theta_k)) + np.sqrt(
            ((gamma - (theta_k * theta_k)) * (gamma - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2
    mu = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1),0.9)
    muHist.append(mu)
    theta_k = theta_kp1
    nochangeT = True
    #thetak.append(theta_k)
    
    gfk = myfprime(xk+mu*vk).reshape(-1, 1)

    if k == 0:
        # comment later
        np.random.seed(seed)
        Stemp = np.random.randn(N, m)
        y_vec_tmp = []
        s_vec_tmp = []

        for index in range(m):
            y_vec_tmp.append(myfprime(xk + 1 * Stemp[:, index].reshape(-1, 1)).reshape(-1, 1))
            s_vec_tmp.append(xk + 1 * Stemp[:, index].reshape(-1, 1))

        Stemp = np.squeeze(np.asarray(s_vec_tmp)).T
        Ytemp = np.squeeze(np.asarray(y_vec_tmp)).T

        S = np.zeros((num_weights, 0))
        Y = np.zeros((num_weights, 0))

        counterSucc = 0
        for idx in range(m):

            L = np.zeros((Y.shape[1], Y.shape[1]))
            for ii in range(Y.shape[1]):
                for jj in range(0, ii):
                    L[ii, jj] = S[:, ii].dot(Y[:, jj])

            tmp = np.sum((S * Y), axis=0)
            D = np.diag(tmp)
            M = (D + L + L.T)
            Minv = np.linalg.inv(M)

            tmp1 = np.matmul(Y.T, Stemp[:, idx])
            tmp2 = np.matmul(Minv, tmp1)
            Bksk = np.squeeze(np.matmul(Y, tmp2))
            yk_BkskDotsk = (Ytemp[:, idx] - Bksk).T.dot(Stemp[:, idx])
            if np.abs(np.squeeze(yk_BkskDotsk)) > (
                    eps * (LA.norm(Ytemp[:, idx] - Bksk) * LA.norm(Stemp[:, idx]))):
                counterSucc += 1

                S = np.append(S, Stemp[:, idx].reshape(num_weights, 1), axis=1)
                Y = np.append(Y, Ytemp[:, idx].reshape(num_weights, 1), axis=1)
                y_vec.append(Ytemp[:, idx].reshape(num_weights, 1))
                s_vec.append(Stemp[:, idx].reshape(num_weights, 1))

    S = np.squeeze(np.asarray(s_vec)).T
    Y = np.squeeze(np.asarray(y_vec)).T

    try:
        sk_TR = CG_Steinhaug_matFree(epsTR, gfk, deltak, S, Y, N)
    except:
        print("reset mu")
        gfk = myfprime(xk).reshape(-1, 1)
        theta_k = 1
        mu = 0
        muHist.pop()
        muHist.append(mu)
        sk_TR = CG_Steinhaug_matFree(epsTR, gfk, deltak, S, Y, N)

    # sk_TR =  -np.dot(Hk, gfk) * deltak

    new_fval = f(xk+mu*vk+sk_TR)
    ared = old_fval - new_fval  # Compute actual reduction
    
    Lp = np.zeros((Y.shape[1], Y.shape[1]))
    for ii in range(Y.shape[1]):
        for jj in range(0, ii):
            Lp[ii, jj] = S[:, ii].dot(Y[:, jj])
    tmpp = np.sum((S * Y), axis=0)
    Dp = np.diag(tmpp)
    Mp = (Dp + Lp + Lp.T)
    Minvp = np.linalg.inv(Mp)
    tmpp1 = np.matmul(Y.T, sk_TR)
    tmpp2 = np.matmul(Minvp, tmpp1)
    Bk_skTR = np.matmul(Y, tmpp2)
    #Bk_skTR = np.dot(Hk, sk_TR)
    pred = -(gfk.T.dot(sk_TR) + 0.5 * sk_TR.T.dot(Bk_skTR))  # Compute predicted reduction

    # Update trust region radius
    if ared / pred > 0.75:
        deltak = 2 * deltak
    elif ared / pred >= 0.1 and ared / pred <= 0.75:
        pass  # no need to change deltak
    elif ared / pred < 0.1:
        deltak = deltak * 0.5
        #theta_k = 1
        #nochangeT = False


    delta_k.append(deltak)

    
    # Take step
    if ared / pred > eta:
        #count = 0
        xkp1 = xk +mu*vk + sk_TR
        vkp1 = mu*vk + sk_TR
        old_fval = f(xkp1)

    else:

        #count += 1
        theta_k = 0.5
        if deltak < 1e-5:
            theta_k = 0.1
        #if nochangeT:
        #theta_k = thetak[-1]
        #thetak.append(theta_k)
        #theta_k = theta_k * 0.5

        xkp1 = xk
        vkp1 = vk
        flag = 0

    #if mu>=0.95: theta_k=1

    thetak.append(theta_k)

    if True:#flag:

        if retall:
            allvecs.append(xkp1)
        sk = sk_TR  # xkp1 - xk
        xk = xkp1
        vk = vkp1
        
        # if gfkp1 is None:
        gfkp1 = myfprime(xkp1).reshape(-1, 1)

        gnorm = vecnorm(gfkp1, ord=norm)

        yk = gfkp1 - gfk

        # Global Convergence Term
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const

        myk = yk + zeta * gnorm * sk

        gfk = gfkp1
        gfk_vec.append(gfkp1)


        """
        yk_BkskDotsk = (Ytemp[:,idx]- Bksk ).T.dot(Stemp[:,idx]  )  
        if np.abs(np.squeeze(yk_BkskDotsk)) > (eps *(LA.norm(Ytemp[:,idx]- Bksk )  * LA.norm(Stemp[:,idx]))  ):        
            counterSucc += 1

            S = np.append(S,Stemp[:,idx].reshape(num_weights,1),axis = 1)
            Y = np.append(Y,Ytemp[:,idx].reshape(num_weights,1),axis=1)
        """

        if abs(np.dot(sk.T, (myk - Bk_skTR))) >= 1e-8 * vecnorm(sk)*vecnorm(myk - Bk_skTR):
            #print("global conv")
            s_vec.append(sk)
            y_vec.append(myk)
            vk_vec.append(vk)

        elif abs(np.dot(sk.T, (yk - Bk_skTR))) >= 1e-8 * vecnorm(sk)*vecnorm(yk - Bk_skTR):
            #print("no global conv")
            s_vec.append(sk)
            y_vec.append(yk)
            vk_vec.append(vk)

        else: print("update skipped")



    end = time.time()
    timePlot.append(end - start)
    errHistory.append(f(xk))
    if callback is not None:
        callback(xk)
    

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result


def _minimize_sr1(fun, x0, args=(), jac=None, callback=None, timePlot=[],seed=100,
                  gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, errHistory=None,
                  disp=False, return_all=False, finite_diff_rel_step=None,m=10,etol=None,
                  **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    """
    _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    import collections
    from numpy import linalg as LA
    s_vec_tmp = collections.deque(maxlen=m)
    y_vec_tmp = collections.deque(maxlen=m)


    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0).reshape(-1,1)


    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    N = len(x0)
    num_weights = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    deltak = 1
    eta = 1e-6
    epsTR = 1e-10

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0.reshape(-1,1)
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    import time
    timePlot.append(0)
    errHistory.append(old_fval)

    while (gnorm > gtol) and (k < maxiter):
        if etol != None and old_fval < etol: break

        start_time = time.time()
        k += 1
        #print(old_fval)
        #errHistory.append(old_fval)

        if k == 1:
            # comment later
            np.random.seed(seed)
            Stemp = np.random.randn(N, m)

            for index in range(m):
                y_vec_tmp.append(myfprime(xk + 1 * Stemp[:, index].reshape(-1, 1)).reshape(-1, 1))
                s_vec_tmp.append(xk + 1 * Stemp[:, index].reshape(-1, 1))

            Stemp = np.squeeze(np.asarray(s_vec_tmp)).T
            Ytemp = np.squeeze(np.asarray(y_vec_tmp)).T

            S = np.zeros((num_weights, 0))
            Y = np.zeros((num_weights, 0))

            counterSucc = 0
            for idx in range(m):

                L = np.zeros((Y.shape[1], Y.shape[1]))
                for ii in range(Y.shape[1]):
                    for jj in range(0, ii):
                        L[ii, jj] = S[:, ii].dot(Y[:, jj])

                tmp = np.sum((S * Y), axis=0)
                D = np.diag(tmp)
                M = (D + L + L.T)
                Minv = np.linalg.inv(M)

                tmp1 = np.matmul(Y.T, Stemp[:, idx])
                tmp2 = np.matmul(Minv, tmp1)
                Bksk = np.squeeze(np.matmul(Y, tmp2))
                yk_BkskDotsk = (Ytemp[:, idx] - Bksk).T.dot(Stemp[:, idx])
                if np.abs(np.squeeze(yk_BkskDotsk)) > (
                        eps * (LA.norm(Ytemp[:, idx] - Bksk) * LA.norm(Stemp[:, idx]))):
                    counterSucc += 1

                    S = np.append(S, Stemp[:, idx].reshape(num_weights, 1), axis=1)
                    Y = np.append(Y, Ytemp[:, idx].reshape(num_weights, 1), axis=1)

        #S = np.squeeze(np.asarray(s_vec)).T
        #Y = np.squeeze(np.asarray(y_vec)).T

        sk_TR = CG_Steinhaug_matFree(epsTR, gfk, deltak, S, Y, N)
        #sk_TR =  -np.dot(Hk, gfk) * deltak

        new_fval = f(xk+sk_TR)
        ared = old_fval - new_fval  # Compute actual reduction



        Lp = np.zeros((Y.shape[1], Y.shape[1]))
        for ii in range(Y.shape[1]):
            for jj in range(0, ii):
                Lp[ii, jj] = S[:, ii].dot(Y[:, jj])
        tmpp = np.sum((S * Y), axis=0)
        Dp = np.diag(tmpp)
        Mp = (Dp + Lp + Lp.T)
        Minvp = np.linalg.inv(Mp)
        tmpp1 = np.matmul(Y.T, sk_TR)
        tmpp2 = np.matmul(Minvp, tmpp1)
        Bk_skTR = np.matmul(Y, tmpp2)
        #Bk_skTR = np.dot(Hk, sk_TR)
        pred = -(gfk.T.dot(sk_TR) + 0.5 * sk_TR.T.dot(Bk_skTR))  # Compute predicted reduction

        # Update trust region radius
        if ared / pred > 0.75:
            deltak = 2 * deltak
        elif ared / pred >= 0.1 and ared / pred <= 0.75:
            pass  # no need to change deltak
        elif ared / pred < 0.1:
            deltak = deltak * 0.5

        # Take step
        if ared / pred > eta:
            xkp1 = xk + sk_TR
            old_fval = f(xkp1)
        else:
            xkp1 = xk
            end_time = time.time()
            timePlot.append(end_time-start_time)
            errHistory.append(old_fval)
            continue



        """
        try:
            alpha_k = 1
            LHS = f(xk + pk)
            old_old_fval = f(xk)
            gfk_times_pk = np.dot(gfk.T, pk)
            RHS = old_old_fval + 1e-3 * gfk_times_pk
            for line1 in range(20):
                if LHS < RHS :
                    old_fval = LHS
                    print("LineSearch satisfied")
                    break

                alpha_k /= 10
                LHS = f(xk + alpha_k * pk)
                RHS = old_old_fval + 1e-3 * alpha_k * gfk_times_pk

            '''alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100, amax=1e100)'''
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break
        """

        #xkp1 = xk + alpha_k * pk

        if retall:
            allvecs.append(xkp1)
        sk = sk_TR#xkp1 - xk
        xk = xkp1
        # if gfkp1 is None:
        gfkp1 = myfprime(xkp1).reshape(-1,1)

        yk = gfkp1 - gfk


        # Global Convergence Term
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        #yk = yk + zeta * gnorm * sk
        #yk = yk + sk

        gfk = gfkp1

        yk_BkskDotsk = (yk - Bk_skTR).T.dot(sk)
        if np.abs(np.squeeze(yk_BkskDotsk)) > (eps * (LA.norm(yk - Bksk) * LA.norm(sk))):
            # s_vec.append(sk)
            # y_vec.append(yk)
            S = np.append(S, sk, axis=1)
            Y = np.append(Y, yk, axis=1)
            S = S[:, -10:]
            Y = Y[:, -10:]

        if callback is not None:
            callback(xk)
        
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        end_time = time.time()
        timePlot.append(end_time - start_time)
        errHistory.append(old_fval)

        """rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remaines for more safety
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        else:
            rhok = 1. / rhok_inv

        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])
        """

        """A1 = sk - np.dot(Hk, yk)
        # if A1.all()!=0:
        num = np.dot(A1, A1.T)
        den = np.dot(A1.T, yk)
        if den != 0:
            Hk = Hk + num / den"""


    fval = old_fval
    #errHistory.append(old_fval)

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def _minimize_osr1(fun, x0, args=(), jac=None, callback=None, timePlot=[],seed=100,
                  gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None, errHistory=None,s_vec=None,y_vec=None,gfk_vec=None,
                  disp=False, return_all=False, finite_diff_rel_step=None,m=10,etol=None,delta_k=None,
                  **unknown_options):

    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    xk = asarray(x0).flatten()
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    import time
    from numpy import linalg as LA
    start = time.time()

    k = len(s_vec)
    N = len(x0)
    num_weights = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    eta = 1e-6
    epsTR = 1e-10


    if k == 0:
        deltak = 1
        delta_k.append(deltak)
        old_fval = f(xk)
        gfk = myfprime(xk).reshape(-1, 1)
        import time
        timePlot.append(0)
        errHistory.append(old_fval)
    else:
        deltak = delta_k[0]
        gfk = gfk_vec[0]
        old_fval = errHistory[-1]


    # Sets the initial step guess to dx ~ 1
    # old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0.reshape(-1, 1)
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)

    flag = 1
    start_time = time.time()

    if k == 0:
        # comment later
        np.random.seed(seed)
        Stemp = np.random.randn(N, m)
        y_vec_tmp = []
        s_vec_tmp = []

        for index in range(m):
            y_vec_tmp.append(myfprime(xk + 1 * Stemp[:, index].reshape(-1, 1)).reshape(-1, 1))
            s_vec_tmp.append(xk + 1 * Stemp[:, index].reshape(-1, 1))

        Stemp = np.squeeze(np.asarray(s_vec_tmp)).T
        Ytemp = np.squeeze(np.asarray(y_vec_tmp)).T

        S = np.zeros((num_weights, 0))
        Y = np.zeros((num_weights, 0))

        counterSucc = 0
        for idx in range(m):

            L = np.zeros((Y.shape[1], Y.shape[1]))
            for ii in range(Y.shape[1]):
                for jj in range(0, ii):
                    L[ii, jj] = S[:, ii].dot(Y[:, jj])

            tmp = np.sum((S * Y), axis=0)
            D = np.diag(tmp)
            M = (D + L + L.T)
            Minv = np.linalg.inv(M)

            tmp1 = np.matmul(Y.T, Stemp[:, idx])
            tmp2 = np.matmul(Minv, tmp1)
            Bksk = np.squeeze(np.matmul(Y, tmp2))
            yk_BkskDotsk = (Ytemp[:, idx] - Bksk).T.dot(Stemp[:, idx])
            if np.abs(np.squeeze(yk_BkskDotsk)) > (
                    eps * (LA.norm(Ytemp[:, idx] - Bksk) * LA.norm(Stemp[:, idx]))):
                counterSucc += 1

                S = np.append(S, Stemp[:, idx].reshape(num_weights, 1), axis=1)
                Y = np.append(Y, Ytemp[:, idx].reshape(num_weights, 1), axis=1)
                y_vec.append(Ytemp[:, idx].reshape(num_weights, 1))
                s_vec.append(Stemp[:, idx].reshape(num_weights, 1))

    S = np.squeeze(np.asarray(s_vec)).T
    Y = np.squeeze(np.asarray(y_vec)).T

    sk_TR = CG_Steinhaug_matFree(epsTR, gfk, deltak, S, Y, N)
    # sk_TR =  -np.dot(Hk, gfk) * deltak

    new_fval = f(xk + sk_TR)
    ared = old_fval - new_fval  # Compute actual reduction

    Lp = np.zeros((Y.shape[1], Y.shape[1]))
    for ii in range(Y.shape[1]):
        for jj in range(0, ii):
            Lp[ii, jj] = S[:, ii].dot(Y[:, jj])
    tmpp = np.sum((S * Y), axis=0)
    Dp = np.diag(tmpp)
    Mp = (Dp + Lp + Lp.T)
    Minvp = np.linalg.inv(Mp)
    tmpp1 = np.matmul(Y.T, sk_TR)
    tmpp2 = np.matmul(Minvp, tmpp1)
    Bk_skTR = np.matmul(Y, tmpp2)
    # Bk_skTR = np.dot(Hk, sk_TR)
    pred = -(gfk.T.dot(sk_TR) + 0.5 * sk_TR.T.dot(Bk_skTR))  # Compute predicted reduction

    # Update trust region radius
    if ared / pred > 0.75:
        deltak = 2 * deltak
    elif ared / pred >= 0.1 and ared / pred <= 0.75:
        pass  # no need to change deltak
    elif ared / pred < 0.1:
        deltak = deltak * 0.5

    delta_k.append(deltak)
    # Take step
    if ared / pred > eta:
        xkp1 = xk + sk_TR
        old_fval = f(xkp1)
    else:
        xkp1 = xk
        flag = 0

    if True:#flag:

        if retall:
            allvecs.append(xkp1)
        sk = sk_TR  # xkp1 - xk
        xk = xkp1
        # if gfkp1 is None:
        gfkp1 = myfprime(xkp1).reshape(-1, 1)

        yk = gfkp1 - gfk

        # Global Convergence Term
        p_times_q = np.dot(sk.T, yk)
        if gnorm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_times_q < 0:
            p_times_p = np.dot(sk.T, sk)
            zeta = const - (p_times_q / (p_times_p * gnorm))
        else:
            zeta = const
        myk = yk + zeta * gnorm * sk

        gfk = gfkp1
        gfk_vec.append(gfkp1)


        if abs(np.dot(sk.T, (myk - Bk_skTR))) >= 1e-8 * vecnorm(sk)*vecnorm(myk - Bk_skTR):
            #print("global conv")
            s_vec.append(sk)
            y_vec.append(myk)


        elif abs(np.dot(sk.T, (yk - Bk_skTR))) >= 1e-8 * vecnorm(sk)*vecnorm(yk - Bk_skTR):
            print("no global conv")
            s_vec.append(sk)
            y_vec.append(yk)




    end = time.time()
    timePlot.append(end - start)
    errHistory.append(f(xk))
    if callback is not None:
        callback(xk)
    k += 1

    result = OptimizeResult(fun=0, jac=0, hess_inv=0, nfev=0,
                            njev=0, status=0,
                            success=(0), message=0, x=xkp1,
                            nit=k)

    return result


def fmin_cg(f, x0, fprime=None, args=(), gtol=1e-5, norm=Inf, epsilon=_epsilon,
            maxiter=None, full_output=0, disp=1, retall=0, callback=None):
    """
    Minimize a function using a nonlinear conjugate gradient algorithm.

    Parameters
    ----------
    f : callable, ``f(x, *args)``
        Objective function to be minimized. Here `x` must be a 1-D array of
        the variables that are to be changed in the search for a minimum, and
        `args` are the other (fixed) parameters of `f`.
    x0 : ndarray
        A user-supplied initial estimate of `xopt`, the optimal value of `x`.
        It must be a 1-D array of values.
    fprime : callable, ``fprime(x, *args)``, optional
        A function that returns the gradient of `f` at `x`. Here `x` and `args`
        are as described above for `f`. The returned value must be a 1-D array.
        Defaults to None, in which case the gradient is approximated
        numerically (see `epsilon`, below).
    args : tuple, optional
        Parameter values passed to `f` and `fprime`. Must be supplied whenever
        additional fixed parameters are needed to completely specify the
        functions `f` and `fprime`.
    gtol : float, optional
        Stop when the norm of the gradient is less than `gtol`.
    norm : float, optional
        Order to use for the norm of the gradient
        (``-np.Inf`` is min, ``np.Inf`` is max).
    epsilon : float or ndarray, optional
        Step size(s) to use when `fprime` is approximated numerically. Can be a
        scalar or a 1-D array. Defaults to ``sqrt(eps)``, with eps the
        floating point machine precision.  Usually ``sqrt(eps)`` is about
        1.5e-8.
    maxiter : int, optional
        Maximum number of iterations to perform. Default is ``200 * len(x0)``.
    full_output : bool, optional
        If True, return `fopt`, `func_calls`, `grad_calls`, and `warnflag` in
        addition to `xopt`.  See the Returns section below for additional
        information on optional return values.
    disp : bool, optional
        If True, return a convergence message, followed by `xopt`.
    retall : bool, optional
        If True, add to the returned values the results of each iteration.
    callback : callable, optional
        An optional user-supplied function, called after each iteration.
        Called as ``callback(xk)``, where ``xk`` is the current value of `x0`.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float, optional
        Minimum value found, f(xopt). Only returned if `full_output` is True.
    func_calls : int, optional
        The number of function_calls made. Only returned if `full_output`
        is True.
    grad_calls : int, optional
        The number of gradient calls made. Only returned if `full_output` is
        True.
    warnflag : int, optional
        Integer value with warning status, only returned if `full_output` is
        True.

        0 : Success.

        1 : The maximum number of iterations was exceeded.

        2 : Gradient and/or function calls were not changing. May indicate
            that precision was lost, i.e., the routine did not converge.

        3 : NaN result encountered.

    allvecs : list of ndarray, optional
        List of arrays, containing the results at each iteration.
        Only returned if `retall` is True.

    See Also
    --------
    minimize : common interface to all `scipy.optimize` algorithms for
               unconstrained and constrained minimization of multivariate
               functions. It provides an alternative way to call
               ``fmin_cg``, by specifying ``method='CG'``.

    Notes
    -----
    This conjugate gradient algorithm is based on that of Polak and Ribiere
    [1]_.

    Conjugate gradient methods tend to work better when:

    1. `f` has a unique global minimizing point, and no local minima or
       other stationary points,
    2. `f` is, at least locally, reasonably well approximated by a
       quadratic function of the variables,
    3. `f` is continuous and has a continuous gradient,
    4. `fprime` is not too large, e.g., has a norm less than 1000,
    5. The initial guess, `x0`, is reasonably close to `f` 's global
       minimizing point, `xopt`.

    References
    ----------
    .. [1] Wright & Nocedal, "Numerical Optimization", 1999, pp. 120-122.

    Examples
    --------
    Example 1: seek the minimum value of the expression
    ``a*u**2 + b*u*v + c*v**2 + d*u + e*v + f`` for given values
    of the parameters and an initial guess ``(u, v) = (0, 0)``.

    >>> args = (2, 3, 7, 8, 9, 10)  # parameter values
    >>> def f(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
    >>> def gradf(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     gu = 2*a*u + b*v + d     # u-component of the gradient
    ...     gv = b*u + 2*c*v + e     # v-component of the gradient
    ...     return np.asarray((gu, gv))
    >>> x0 = np.asarray((0, 0))  # Initial guess.
    >>> from scipy import optimize
    >>> res1 = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
    Optimization terminated successfully.
             Current function value: 1.617021
             Iterations: 4
             Function evaluations: 8
             Gradient evaluations: 8
    >>> res1
    array([-1.80851064, -0.25531915])

    Example 2: solve the same problem using the `minimize` function.
    (This `myopts` dictionary shows all of the available options,
    although in practice only non-default values would be needed.
    The returned value will be a dictionary.)

    >>> opts = {'maxiter' : None,    # default value.
    ...         'disp' : True,    # non-default value.
    ...         'gtol' : 1e-5,    # default value.
    ...         'norm' : np.inf,  # default value.
    ...         'eps' : 1.4901161193847656e-08}  # default value.
    >>> res2 = optimize.minimize(f, x0, jac=gradf, args=args,
    ...                          method='CG', options=opts)
    Optimization terminated successfully.
            Current function value: 1.617021
            Iterations: 4
            Function evaluations: 8
            Gradient evaluations: 8
    >>> res2.x  # minimum found
    array([-1.80851064, -0.25531915])

    """
    opts = {'gtol': gtol,
            'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}

    res = _minimize_cg(f, x0, args, fprime, callback=callback, **opts)

    if full_output:
        retlist = res['x'], res['fun'], res['nfev'], res['njev'], res['status']
        if retall:
            retlist += (res['allvecs'],)
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_cg(fun, x0, args=(), jac=None, callback=None,
                 gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
                 disp=False, return_all=False, finite_diff_rel_step=None,
                 **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    _check_unknown_options(unknown_options)

    retall = return_all

    x0 = asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.") from e

    k = 0
    xk = x0
    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    if retall:
        allvecs = [xk]
    warnflag = 0
    pk = -gfk
    gnorm = vecnorm(gfk, ord=norm)

    sigma_3 = 0.01

    while (gnorm > gtol) and (k < maxiter):
        deltak = np.dot(gfk, gfk)

        cached_step = [None]

        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = myfprime(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, np.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = vecnorm(gfkp1, ord=norm)
            return (alpha, xkp1, pkp1, gfkp1, gnorm)

        def descent_condition(alpha, xkp1, fp1, gfkp1):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            #
            # See Gilbert & Nocedal, "Global convergence properties of
            # conjugate gradient methods for optimization",
            # SIAM J. Optimization 2, 21 (1992).
            cached_step[:] = polak_ribiere_powell_step(alpha, gfkp1)
            alpha, xk, pk, gfk, gnorm = cached_step

            # Accept step if it leads to convergence.
            if gnorm <= gtol:
                return True

            # Accept step if sufficient descent condition applies.
            return np.dot(pk, gfk) <= -sigma_3 * np.dot(gfk, gfk)

        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(f, myfprime, xk, pk, gfk, old_fval,
                                     old_old_fval, c2=0.4, amin=1e-100, amax=1e100,
                                     extra_condition=descent_condition)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        # Reuse already computed results if possible
        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)

        if retall:
            allvecs.append(xk)
        if callback is not None:
            callback(xk)
        k += 1

    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def fmin_ncg(f, x0, fprime, fhess_p=None, fhess=None, args=(), avextol=1e-5,
             epsilon=_epsilon, maxiter=None, full_output=0, disp=1, retall=0,
             callback=None):
    """
    Unconstrained minimization of a function using the Newton-CG method.

    Parameters
    ----------
    f : callable ``f(x, *args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable ``f'(x, *args)``
        Gradient of f.
    fhess_p : callable ``fhess_p(x, p, *args)``, optional
        Function which computes the Hessian of f times an
        arbitrary vector, p.
    fhess : callable ``fhess(x, *args)``, optional
        Function to compute the Hessian matrix of f.
    args : tuple, optional
        Extra arguments passed to f, fprime, fhess_p, and fhess
        (the same set of extra arguments is supplied to all of
        these functions).
    epsilon : float or ndarray, optional
        If fhess is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function which is called after
        each iteration. Called as callback(xk), where xk is the
        current parameter vector.
    avextol : float, optional
        Convergence is assumed when the average relative error in
        the minimizer falls below this amount.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True, return the optional outputs.
    disp : bool, optional
        If True, print convergence message.
    retall : bool, optional
        If True, return a list of results at each iteration.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float
        Value of the function at xopt, i.e., ``fopt = f(xopt)``.
    fcalls : int
        Number of function calls made.
    gcalls : int
        Number of gradient calls made.
    hcalls : int
        Number of Hessian calls made.
    warnflag : int
        Warnings generated by the algorithm.
        1 : Maximum number of iterations exceeded.
        2 : Line search failure (precision loss).
        3 : NaN result encountered.
    allvecs : list
        The result at each iteration, if retall is True (see below).

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'Newton-CG' `method` in particular.

    Notes
    -----
    Only one of `fhess_p` or `fhess` need to be given.  If `fhess`
    is provided, then `fhess_p` will be ignored. If neither `fhess`
    nor `fhess_p` is provided, then the hessian product will be
    approximated using finite differences on `fprime`. `fhess_p`
    must compute the hessian times an arbitrary vector. If it is not
    given, finite-differences on `fprime` are used to compute
    it.

    Newton-CG methods are also called truncated Newton methods. This
    function differs from scipy.optimize.fmin_tnc because

    1. scipy.optimize.fmin_ncg is written purely in Python using NumPy
        and scipy while scipy.optimize.fmin_tnc calls a C function.
    2. scipy.optimize.fmin_ncg is only for unconstrained minimization
        while scipy.optimize.fmin_tnc is for unconstrained minimization
        or box constrained minimization. (Box constraints give
        lower and upper bounds for each variable separately.)

    References
    ----------
    Wright & Nocedal, 'Numerical Optimization', 1999, p. 140.

    """
    opts = {'xtol': avextol,
            'eps': epsilon,
            'maxiter': maxiter,
            'disp': disp,
            'return_all': retall}

    res = _minimize_newtoncg(f, x0, args, fprime, fhess, fhess_p,
                             callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['nfev'], res['njev'],
                   res['nhev'], res['status'])
        if retall:
            retlist += (res['allvecs'],)
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=_epsilon, maxiter=None,
                       disp=False, return_all=False,
                       **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Note that the `jac` parameter (Jacobian) is required.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `hessp` is approximated, use this value for the step size.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    _check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    # TODO: allow hess to be approximated by FD?
    # TODO: add hessp (callable or FD) to ScalarFunction?
    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps, hess=fhess)
    f = sf.fun
    fprime = sf.grad

    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % old_fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % sf.nfev)
            print("         Gradient evaluations: %d" % sf.ngev)
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                                njev=sf.ngev, nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result

    hcalls = 0
    if maxiter is None:
        maxiter = len(x0) * 200
    cg_maxiter = 20 * len(x0)

    xtol = len(x0) * avextol
    update = [2 * xtol]
    xk = x0
    if retall:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = f(x0)
    old_old_fval = None
    float64eps = np.finfo(np.float64).eps
    while np.add.reduce(np.abs(update)) > xtol:
        if k >= maxiter:
            msg = "Warning: " + _status_message['maxiter']
            return terminate(1, msg)
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -fprime(xk)
        maggrad = np.add.reduce(np.abs(b))
        eta = np.min([0.5, np.sqrt(maggrad)])
        termcond = eta * maggrad
        xsupi = zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = np.dot(ri, ri)

        if fhess is not None:  # you want to compute hessian once.
            A = sf.hess(xk)
            hcalls = hcalls + 1

        for k2 in range(cg_maxiter):
            if np.add.reduce(np.abs(ri)) <= termcond:
                break
            if fhess is None:
                if fhess_p is None:
                    Ap = approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls = hcalls + 1
            else:
                Ap = np.dot(A, psupi)
            # check curvature
            Ap = asarray(Ap).squeeze()  # get rid of matrices...
            curv = np.dot(psupi, Ap)
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi = xsupi + alphai * psupi
            ri = ri + alphai * Ap
            dri1 = np.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i = i + 1
            dri0 = dri1  # update np.dot(ri,ri) for next time.
        else:
            # curvature keeps increasing, bail out
            msg = ("Warning: CG iterations didn't converge. The Hessian is not "
                   "positive definite.")
            return terminate(3, msg)

        pk = xsupi  # search direction is solution to system.
        gfk = -b  # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval)
        except _LineSearchError:
            # Line search failed to find a better solution.
            msg = "Warning: " + _status_message['pr_loss']
            return terminate(2, msg)

        update = alphak * pk
        xk = xk + update  # upcast if necessary
        if callback is not None:
            callback(xk)
        if retall:
            allvecs.append(xk)
        k += 1
    else:
        if np.isnan(old_fval) or np.isnan(update).any():
            return terminate(3, _status_message['nan'])

        msg = _status_message['success']
        return terminate(0, msg)


def fminbound(func, x1, x2, args=(), xtol=1e-5, maxfun=500,
              full_output=0, disp=1):
    """Bounded minimization for scalar functions.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized (must accept and return scalars).
    x1, x2 : float or array scalar
        The optimization bounds.
    args : tuple, optional
        Extra arguments passed to function.
    xtol : float, optional
        The convergence tolerance.
    maxfun : int, optional
        Maximum number of function evaluations allowed.
    full_output : bool, optional
        If True, return optional outputs.
    disp : int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.


    Returns
    -------
    xopt : ndarray
        Parameters (over given interval) which minimize the
        objective function.
    fval : number
        The function value at the minimum point.
    ierr : int
        An error flag (0 if converged, 1 if maximum number of
        function calls reached).
    numfunc : int
      The number of function calls made.

    See also
    --------
    minimize_scalar: Interface to minimization algorithms for scalar
        univariate functions. See the 'Bounded' `method` in particular.

    Notes
    -----
    Finds a local minimizer of the scalar function `func` in the
    interval x1 < xopt < x2 using Brent's method. (See `brent`
    for auto-bracketing.)

    Examples
    --------
    `fminbound` finds the minimum of the function in the given range.
    The following examples illustrate the same

    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fminbound(f, -1, 2)
    >>> minimum
    0.0
    >>> minimum = optimize.fminbound(f, 1, 2)
    >>> minimum
    1.0000059608609866
    """
    options = {'xatol': xtol,
               'maxiter': maxfun,
               'disp': disp}

    res = _minimize_scalar_bounded(func, (x1, x2), args, **options)
    if full_output:
        return res['x'], res['fun'], res['status'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_bounded(func, bounds, args=(),
                             xatol=1e-5, maxiter=500, disp=0,
                             **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.
    xatol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    _check_unknown_options(unknown_options)
    maxfun = maxiter
    # Test bounds are of correct form
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    x1, x2 = bounds

    if not (is_array_scalar(x1) and is_array_scalar(x2)):
        raise ValueError("Optimization bounds must be scalars"
                         " or array scalars.")
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    flag = 0
    header = ' Func-count     x          f(x)          Procedure'
    step = '       initial'

    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = np.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    if disp > 2:
        print(" ")
        print(header)
        print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

    while (np.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((np.abs(p) < np.abs(0.5 * q * r)) and (p > q * (a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                step = '       parabolic'

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:  # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
            step = '       golden'

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)
        fu = func(x, *args)
        num += 1
        fmin_data = (num, x, fu)
        if disp > 2:
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    fval = fx
    if disp > 0:
        _endprint(x, flag, fval, maxfun, xatol, disp)

    result = OptimizeResult(fun=fval, status=flag, success=(flag == 0),
                            message={0: 'Solution found.',
                                     1: 'Maximum number of function calls '
                                        'reached.',
                                     2: _status_message['nan']}.get(flag, ''),
                            x=xf, nfev=num)

    return result


class Brent:
    # need to rethink design of __init__
    def __init__(self, func, args=(), tol=1.48e-8, maxiter=500,
                 full_output=0):
        self.func = func
        self.args = args
        self.tol = tol
        self.maxiter = maxiter
        self._mintol = 1.0e-11
        self._cg = 0.3819660
        self.xmin = None
        self.fval = None
        self.iter = 0
        self.funcalls = 0

    # need to rethink design of set_bracket (new options, etc.)
    def set_bracket(self, brack=None):
        self.brack = brack

    def get_bracket_info(self):
        # set up
        func = self.func
        args = self.args
        brack = self.brack
        ### BEGIN core bracket_info code ###
        ### carefully DOCUMENT any CHANGES in core ##
        if brack is None:
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
        elif len(brack) == 2:
            xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0],
                                                       xb=brack[1], args=args)
        elif len(brack) == 3:
            xa, xb, xc = brack
            if (xa > xc):  # swap so xa < xc can be assumed
                xc, xa = xa, xc
            if not ((xa < xb) and (xb < xc)):
                raise ValueError("Not a bracketing interval.")
            fa = func(*((xa,) + args))
            fb = func(*((xb,) + args))
            fc = func(*((xc,) + args))
            if not ((fb < fa) and (fb < fc)):
                raise ValueError("Not a bracketing interval.")
            funcalls = 3
        else:
            raise ValueError("Bracketing interval must be "
                             "length 2 or 3 sequence.")
        ### END core bracket_info code ###

        return xa, xb, xc, fa, fb, fc, funcalls

    def optimize(self):
        # set up for optimization
        func = self.func
        xa, xb, xc, fa, fb, fc, funcalls = self.get_bracket_info()
        _mintol = self._mintol
        _cg = self._cg
        #################################
        # BEGIN CORE ALGORITHM
        #################################
        x = w = v = xb
        fw = fv = fx = func(*((x,) + self.args))
        if (xa < xc):
            a = xa
            b = xc
        else:
            a = xc
            b = xa
        deltax = 0.0
        funcalls += 1
        iter = 0
        while (iter < self.maxiter):
            tol1 = self.tol * np.abs(x) + _mintol
            tol2 = 2.0 * tol1
            xmid = 0.5 * (a + b)
            # check for convergence
            if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
                break
            # XXX In the first iteration, rat is only bound in the true case
            # of this conditional. This used to cause an UnboundLocalError
            # (gh-4140). It should be set before the if (but to what?).
            if (np.abs(deltax) <= tol1):
                if (x >= xmid):
                    deltax = a - x  # do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax
            else:  # do a parabolic step
                tmp1 = (x - w) * (fx - fv)
                tmp2 = (x - v) * (fx - fw)
                p = (x - v) * tmp2 - (x - w) * tmp1
                tmp2 = 2.0 * (tmp2 - tmp1)
                if (tmp2 > 0.0):
                    p = -p
                tmp2 = np.abs(tmp2)
                dx_temp = deltax
                deltax = rat
                # check parabolic fit
                if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                        (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                    rat = p * 1.0 / tmp2  # if parabolic step is useful.
                    u = x + rat
                    if ((u - a) < tol2 or (b - u) < tol2):
                        if xmid - x >= 0:
                            rat = tol1
                        else:
                            rat = -tol1
                else:
                    if (x >= xmid):
                        deltax = a - x  # if it's not do a golden section step
                    else:
                        deltax = b - x
                    rat = _cg * deltax

            if (np.abs(rat) < tol1):  # update by at least tol1
                if rat >= 0:
                    u = x + tol1
                else:
                    u = x - tol1
            else:
                u = x + rat
            fu = func(*((u,) + self.args))  # calculate new output value
            funcalls += 1

            if (fu > fx):  # if it's bigger than current
                if (u < x):
                    a = u
                else:
                    b = u
                if (fu <= fw) or (w == x):
                    v = w
                    w = u
                    fv = fw
                    fw = fu
                elif (fu <= fv) or (v == x) or (v == w):
                    v = u
                    fv = fu
            else:
                if (u >= x):
                    a = x
                else:
                    b = x
                v = w
                w = x
                x = u
                fv = fw
                fw = fx
                fx = fu

            iter += 1
        #################################
        # END CORE ALGORITHM
        #################################

        self.xmin = x
        self.fval = fx
        self.iter = iter
        self.funcalls = funcalls

    def get_result(self, full_output=False):
        if full_output:
            return self.xmin, self.fval, self.iter, self.funcalls
        else:
            return self.xmin


def brent(func, args=(), brack=None, tol=1.48e-8, full_output=0, maxiter=500):
    """
    Given a function of one variable and a possible bracket, return
    the local minimum of the function isolated to a fractional precision
    of tol.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function.
    args : tuple, optional
        Additional arguments (if present).
    brack : tuple, optional
        Either a triple (xa,xb,xc) where xa<xb<xc and func(xb) <
        func(xa), func(xc) or a pair (xa,xb) which are used as a
        starting interval for a downhill bracket search (see
        `bracket`). Providing the pair (xa,xb) does not always mean
        the obtained solution will satisfy xa<=x<=xb.
    tol : float, optional
        Stop if between iteration change is less than `tol`.
    full_output : bool, optional
        If True, return all output args (xmin, fval, iter,
        funcalls).
    maxiter : int, optional
        Maximum number of iterations in solution.

    Returns
    -------
    xmin : ndarray
        Optimum point.
    fval : float
        Optimum value.
    iter : int
        Number of iterations.
    funcalls : int
        Number of objective function evaluations made.

    See also
    --------
    minimize_scalar: Interface to minimization algorithms for scalar
        univariate functions. See the 'Brent' `method` in particular.

    Notes
    -----
    Uses inverse parabolic interpolation when possible to speed up
    convergence of golden section method.

    Does not ensure that the minimum lies in the range specified by
    `brack`. See `fminbound`.

    Examples
    --------
    We illustrate the behaviour of the function when `brack` is of
    size 2 and 3 respectively. In the case where `brack` is of the
    form (xa,xb), we can see for the given values, the output need
    not necessarily lie in the range (xa,xb).

    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.brent(f,brack=(1,2))
    >>> minimum
    0.0
    >>> minimum = optimize.brent(f,brack=(-1,0.5,2))
    >>> minimum
    -2.7755575615628914e-17

    """
    options = {'xtol': tol,
               'maxiter': maxiter}
    res = _minimize_scalar_brent(func, brack, args, **options)
    if full_output:
        return res['x'], res['fun'], res['nit'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_brent(func, brack=None, args=(),
                           xtol=1.48e-8, maxiter=500,
                           **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.

    Notes
    -----
    Uses inverse parabolic interpolation when possible to speed up
    convergence of golden section method.

    """
    _check_unknown_options(unknown_options)
    tol = xtol
    if tol < 0:
        raise ValueError('tolerance should be >= 0, got %r' % tol)

    brent = Brent(func=func, args=args, tol=tol,
                  full_output=True, maxiter=maxiter)
    brent.set_bracket(brack)
    brent.optimize()
    x, fval, nit, nfev = brent.get_result(full_output=True)

    success = nit < maxiter and not (np.isnan(x) or np.isnan(fval))

    return OptimizeResult(fun=fval, x=x, nit=nit, nfev=nfev,
                          success=success)


def golden(func, args=(), brack=None, tol=_epsilon,
           full_output=0, maxiter=5000):
    """
    Return the minimum of a function of one variable using golden section
    method.

    Given a function of one variable and a possible bracketing interval,
    return the minimum of the function isolated to a fractional precision of
    tol.

    Parameters
    ----------
    func : callable func(x,*args)
        Objective function to minimize.
    args : tuple, optional
        Additional arguments (if present), passed to func.
    brack : tuple, optional
        Triple (a,b,c), where (a<b<c) and func(b) <
        func(a),func(c). If bracket consists of two numbers (a,
        c), then they are assumed to be a starting interval for a
        downhill bracket search (see `bracket`); it doesn't always
        mean that obtained solution will satisfy a<=x<=c.
    tol : float, optional
        x tolerance stop criterion
    full_output : bool, optional
        If True, return optional outputs.
    maxiter : int
        Maximum number of iterations to perform.

    See also
    --------
    minimize_scalar: Interface to minimization algorithms for scalar
        univariate functions. See the 'Golden' `method` in particular.

    Notes
    -----
    Uses analog of bisection method to decrease the bracketed
    interval.

    Examples
    --------
    We illustrate the behaviour of the function when `brack` is of
    size 2 and 3, respectively. In the case where `brack` is of the
    form (xa,xb), we can see for the given values, the output need
    not necessarily lie in the range ``(xa, xb)``.

    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.golden(f, brack=(1, 2))
    >>> minimum
    1.5717277788484873e-162
    >>> minimum = optimize.golden(f, brack=(-1, 0.5, 2))
    >>> minimum
    -1.5717277788484873e-162

    """
    options = {'xtol': tol, 'maxiter': maxiter}
    res = _minimize_scalar_golden(func, brack, args, **options)
    if full_output:
        return res['x'], res['fun'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_golden(func, brack=None, args=(),
                            xtol=_epsilon, maxiter=5000, **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.

    """
    _check_unknown_options(unknown_options)
    tol = xtol
    if brack is None:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0],
                                                   xb=brack[1], args=args)
    elif len(brack) == 3:
        xa, xb, xc = brack
        if (xa > xc):  # swap so xa < xc can be assumed
            xc, xa = xa, xc
        if not ((xa < xb) and (xb < xc)):
            raise ValueError("Not a bracketing interval.")
        fa = func(*((xa,) + args))
        fb = func(*((xb,) + args))
        fc = func(*((xc,) + args))
        if not ((fb < fa) and (fb < fc)):
            raise ValueError("Not a bracketing interval.")
        funcalls = 3
    else:
        raise ValueError("Bracketing interval must be length 2 or 3 sequence.")

    _gR = 0.61803399  # golden ratio conjugate: 2.0/(1.0+sqrt(5.0))
    _gC = 1.0 - _gR
    x3 = xc
    x0 = xa
    if (np.abs(xc - xb) > np.abs(xb - xa)):
        x1 = xb
        x2 = xb + _gC * (xc - xb)
    else:
        x2 = xb
        x1 = xb - _gC * (xb - xa)
    f1 = func(*((x1,) + args))
    f2 = func(*((x2,) + args))
    funcalls += 2
    nit = 0
    for i in range(maxiter):
        if np.abs(x3 - x0) <= tol * (np.abs(x1) + np.abs(x2)):
            break
        if (f2 < f1):
            x0 = x1
            x1 = x2
            x2 = _gR * x1 + _gC * x3
            f1 = f2
            f2 = func(*((x2,) + args))
        else:
            x3 = x2
            x2 = x1
            x1 = _gR * x2 + _gC * x0
            f2 = f1
            f1 = func(*((x1,) + args))
        funcalls += 1
        nit += 1
    if (f1 < f2):
        xmin = x1
        fval = f1
    else:
        xmin = x2
        fval = f2

    success = nit < maxiter and not (np.isnan(fval) or np.isnan(xmin))

    return OptimizeResult(fun=fval, nfev=funcalls, x=xmin, nit=nit,
                          success=success)


def bracket(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0, maxiter=1000):
    """
    Bracket the minimum of the function.

    Given a function and distinct initial points, search in the
    downhill direction (as defined by the initial points) and return
    new points xa, xb, xc that bracket the minimum of the function
    f(xa) > f(xb) < f(xc). It doesn't always mean that obtained
    solution will satisfy xa<=x<=xb.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to minimize.
    xa, xb : float, optional
        Bracketing interval. Defaults `xa` to 0.0, and `xb` to 1.0.
    args : tuple, optional
        Additional arguments (if present), passed to `func`.
    grow_limit : float, optional
        Maximum grow limit.  Defaults to 110.0
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 1000.

    Returns
    -------
    xa, xb, xc : float
        Bracket.
    fa, fb, fc : float
        Objective function values in bracket.
    funcalls : int
        Number of function evaluations made.

    Examples
    --------
    This function can find a downward convex region of a function:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.optimize import bracket
    >>> def f(x):
    ...     return 10*x**2 + 3*x + 5
    >>> x = np.linspace(-2, 2)
    >>> y = f(x)
    >>> init_xa, init_xb = 0, 1
    >>> xa, xb, xc, fa, fb, fc, funcalls = bracket(f, xa=init_xa, xb=init_xb)
    >>> plt.axvline(x=init_xa, color="k", linestyle="--")
    >>> plt.axvline(x=init_xb, color="k", linestyle="--")
    >>> plt.plot(x, y, "-k")
    >>> plt.plot(xa, fa, "bx")
    >>> plt.plot(xb, fb, "rx")
    >>> plt.plot(xc, fc, "bx")
    >>> plt.show()

    """
    _gold = 1.618034  # golden ratio: (1.0+sqrt(5.0))/2.0
    _verysmall_num = 1e-21
    fa = func(*(xa,) + args)
    fb = func(*(xb,) + args)
    if (fa < fb):  # Switch so fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc = func(*((xc,) + args))
    funcalls = 3
    iter = 0
    while (fc < fb):
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if np.abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        if iter > maxiter:
            raise RuntimeError("Too many iterations.")
        iter += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            if (fw < fc):
                xa = xb
                xb = w
                fa = fb
                fb = fw
                return xa, xb, xc, fa, fb, fc, funcalls
            elif (fw > fb):
                xc = w
                fc = fw
                return xa, xb, xc, fa, fb, fc, funcalls
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim) * (xc - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            if (fw < fc):
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func(*((w,) + args))
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw
    return xa, xb, xc, fa, fb, fc, funcalls


def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def _linesearch_powell(func, p, xi, tol=1e-3,
                       lower_bound=None, upper_bound=None, fval=None):
    """Line-search algorithm using fminbound.

    Find the minimium of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """

    def myfunc(alpha):
        return func(p + alpha * xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
    elif lower_bound is None and upper_bound is None:
        # non-bounded minimization
        alpha_min, fret, _, _ = brent(myfunc, full_output=1, tol=tol)
        xi = alpha_min * xi
        return squeeze(fret), p + xi, xi
    else:
        bound = _line_for_search(p, xi, lower_bound, upper_bound)
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            # equivalent to unbounded
            return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            # we can use a bounded scalar minimization
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
            xi = res.x * xi
            return squeeze(res.fun), p + xi, xi
        else:
            # only bounded on one side. use the tangent function to convert
            # the infinity bound to a finite bound. The new bounded region
            # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            res = _minimize_scalar_bounded(
                lambda x: myfunc(np.tan(x)),
                bound,
                xatol=tol / 100)
            xi = np.tan(res.x) * xi
            return squeeze(res.fun), p + xi, xi


def fmin_powell(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
                maxfun=None, full_output=0, disp=1, retall=0, callback=None,
                direc=None):
    """
    Minimize a function using modified Powell's method.

    This method only uses function values, not derivatives.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func.
    xtol : float, optional
        Line-search error tolerance.
    ftol : float, optional
        Relative error in ``func(xopt)`` acceptable for convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : int, optional
        Maximum number of function evaluations to make.
    full_output : bool, optional
        If True, ``fopt``, ``xi``, ``direc``, ``iter``, ``funcalls``, and
        ``warnflag`` are returned.
    disp : bool, optional
        If True, print convergence messages.
    retall : bool, optional
        If True, return a list of the solution at each iteration.
    callback : callable, optional
        An optional user-supplied function, called after each
        iteration.  Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    direc : ndarray, optional
        Initial fitting step and parameter order set as an (N, N) array, where N
        is the number of fitting parameters in `x0`. Defaults to step size 1.0
        fitting all parameters simultaneously (``np.eye((N, N))``). To
        prevent initial consideration of values in a step or to change initial
        step size, set to 0 or desired step size in the Jth position in the Mth
        block, where J is the position in `x0` and M is the desired evaluation
        step, with steps being evaluated in index order. Step size and ordering
        will change freely as minimization proceeds.

    Returns
    -------
    xopt : ndarray
        Parameter which minimizes `func`.
    fopt : number
        Value of function at minimum: ``fopt = func(xopt)``.
    direc : ndarray
        Current direction set.
    iter : int
        Number of iterations.
    funcalls : int
        Number of function calls made.
    warnflag : int
        Integer warning flag:
            1 : Maximum number of function evaluations.
            2 : Maximum number of iterations.
            3 : NaN result encountered.
            4 : The result is out of the provided bounds.
    allvecs : list
        List of solutions at each iteration.

    See also
    --------
    minimize: Interface to unconstrained minimization algorithms for
        multivariate functions. See the 'Powell' method in particular.

    Notes
    -----
    Uses a modification of Powell's method to find the minimum of
    a function of N variables. Powell's method is a conjugate
    direction method.

    The algorithm has two loops. The outer loop merely iterates over the inner
    loop. The inner loop minimizes over each current direction in the direction
    set. At the end of the inner loop, if certain conditions are met, the
    direction that gave the largest decrease is dropped and replaced with the
    difference between the current estimated x and the estimated x from the
    beginning of the inner-loop.

    The technical conditions for replacing the direction of greatest
    increase amount to checking that

    1. No further gain can be made along the direction of greatest increase
       from that iteration.
    2. The direction of greatest increase accounted for a large sufficient
       fraction of the decrease in the function value from that iteration of
       the inner loop.

    References
    ----------
    Powell M.J.D. (1964) An efficient method for finding the minimum of a
    function of several variables without calculating derivatives,
    Computer Journal, 7 (2):155-162.

    Press W., Teukolsky S.A., Vetterling W.T., and Flannery B.P.:
    Numerical Recipes (any edition), Cambridge University Press

    Examples
    --------
    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fmin_powell(f, -1)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 2
             Function evaluations: 18
    >>> minimum
    array(0.0)

    """
    opts = {'xtol': xtol,
            'ftol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'direc': direc,
            'return_all': retall}

    res = _minimize_powell(func, x0, args, callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['direc'], res['nit'],
                   res['nfev'], res['status'])
        if retall:
            retlist += (res['allvecs'],)
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_powell(func, x0, args=(), callback=None, bounds=None,
                     xtol=1e-4, ftol=1e-4, maxiter=None, maxfev=None,
                     disp=False, direc=None, return_all=False,
                     **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    modified Powell algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    ftol : float
        Relative error in ``fun(xopt)`` acceptable for convergence.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*1000``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    direc : ndarray
        Initial set of direction vectors for the Powell method.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    bounds : `Bounds`
        If bounds are not provided, then an unbounded line search will be used.
        If bounds are provided and the initial guess is within the bounds, then
        every function evaluation throughout the minimization procedure will be
        within the bounds. If bounds are provided, the initial guess is outside
        the bounds, and `direc` is full rank (or left to default), then some
        function evaluations during the first iteration may be outside the
        bounds, but every function evaluation after the first iteration will be
        within the bounds. If `direc` is not full rank, then some parameters may
        not be optimized and the solution is not guaranteed to be within the
        bounds.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    _check_unknown_options(unknown_options)
    maxfun = maxfev
    retall = return_all
    # we need to use a mutable object here that we can update in the
    # wrapper function
    fcalls, func = _wrap_function(func, args)
    x = asarray(x0).flatten()
    if retall:
        allvecs = [x]
    N = len(x)
    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 1000
        maxfun = N * 1000
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 1000
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 1000
        else:
            maxfun = np.inf

    if direc is None:
        direc = eye(N, dtype=float)
    else:
        direc = asarray(direc, dtype=float)
        if np.linalg.matrix_rank(direc) != direc.shape[0]:
            warnings.warn("direc input is not full rank, some parameters may "
                          "not be optimized",
                          OptimizeWarning, 3)

    if bounds is None:
        # don't make these arrays of all +/- inf. because
        # _linesearch_powell will do an unnecessary check of all the elements.
        # just keep them None, _linesearch_powell will not have to check
        # all the elements.
        lower_bound, upper_bound = None, None
    else:
        # bounds is standardized in _minimize.py.
        lower_bound, upper_bound = bounds.lb, bounds.ub
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds",
                          OptimizeWarning, 3)

    fval = squeeze(func(x))
    x1 = x.copy()
    iter = 0
    ilist = list(range(N))
    while True:
        fx = fval
        bigind = 0
        delta = 0.0
        for i in ilist:
            direc1 = direc[i]
            fx2 = fval
            fval, x, direc1 = _linesearch_powell(func, x, direc1,
                                                 tol=xtol * 100,
                                                 lower_bound=lower_bound,
                                                 upper_bound=upper_bound,
                                                 fval=fval)
            if (fx2 - fval) > delta:
                delta = fx2 - fval
                bigind = i
        iter += 1
        if callback is not None:
            callback(x)
        if retall:
            allvecs.append(x)
        bnd = ftol * (np.abs(fx) + np.abs(fval)) + 1e-20
        if 2.0 * (fx - fval) <= bnd:
            break
        if fcalls[0] >= maxfun:
            break
        if iter >= maxiter:
            break
        if np.isnan(fx) and np.isnan(fval):
            # Ended up in a nan-region: bail out
            break

        # Construct the extrapolated point
        direc1 = x - x1
        x2 = 2 * x - x1
        x1 = x.copy()
        fx2 = squeeze(func(x2))

        if (fx > fx2):
            t = 2.0 * (fx + fx2 - 2.0 * fval)
            temp = (fx - fval - delta)
            t *= temp * temp
            temp = fx - fx2
            t -= delta * temp * temp
            if t < 0.0:
                fval, x, direc1 = _linesearch_powell(func, x, direc1,
                                                     tol=xtol * 100,
                                                     lower_bound=lower_bound,
                                                     upper_bound=upper_bound,
                                                     fval=fval)
                if np.any(direc1):
                    direc[bigind] = direc[-1]
                    direc[-1] = direc1

    warnflag = 0
    # out of bounds is more urgent than exceeding function evals or iters,
    # but I don't want to cause inconsistencies by changing the
    # established warning flags for maxfev and maxiter, so the out of bounds
    # warning flag becomes 3, but is checked for first.
    if bounds and (np.any(lower_bound > x) or np.any(x > upper_bound)):
        warnflag = 4
        msg = _status_message['out_of_bounds']
    elif fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print("Warning: " + msg)
    elif iter >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3
        msg = _status_message['nan']
        if disp:
            print("Warning: " + msg)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iter)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, direc=direc, nit=iter, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x)
    if retall:
        result['allvecs'] = allvecs
    return result


def _endprint(x, flag, fval, maxfun, xtol, disp):
    if flag == 0:
        if disp > 1:
            print("\nOptimization terminated successfully;\n"
                  "The returned value satisfies the termination criteria\n"
                  "(using xtol = ", xtol, ")")
    if flag == 1:
        if disp:
            print("\nMaximum number of function evaluations exceeded --- "
                  "increase maxfun argument.\n")
    if flag == 2:
        if disp:
            print("\n{}".format(_status_message['nan']))
    return


def brute(func, ranges, args=(), Ns=20, full_output=0, finish=fmin,
          disp=False, workers=1):
    """Minimize a function over a given range by brute force.

    Uses the "brute force" method, i.e., computes the function's value
    at each point of a multidimensional grid of points, to find the global
    minimum of the function.

    The function is evaluated everywhere in the range with the datatype of the
    first call to the function, as enforced by the ``vectorize`` NumPy
    function. The value and type of the function evaluation returned when
    ``full_output=True`` are affected in addition by the ``finish`` argument
    (see Notes).

    The brute force approach is inefficient because the number of grid points
    increases exponentially - the number of grid points to evaluate is
    ``Ns ** len(x)``. Consequently, even with coarse grid spacing, even
    moderately sized problems can take a long time to run, and/or run into
    memory limitations.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the
        form ``f(x, *args)``, where ``x`` is the argument in
        the form of a 1-D array and ``args`` is a tuple of any
        additional fixed parameters needed to completely specify
        the function.
    ranges : tuple
        Each component of the `ranges` tuple must be either a
        "slice object" or a range tuple of the form ``(low, high)``.
        The program uses these to create the grid of points on which
        the objective function will be computed. See `Note 2` for
        more detail.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify
        the function.
    Ns : int, optional
        Number of grid points along the axes, if not otherwise
        specified. See `Note2`.
    full_output : bool, optional
        If True, return the evaluation grid and the objective function's
        values on it.
    finish : callable, optional
        An optimization function that is called with the result of brute force
        minimization as initial guess. `finish` should take `func` and
        the initial guess as positional arguments, and take `args` as
        keyword arguments. It may additionally take `full_output`
        and/or `disp` as keyword arguments. Use None if no "polishing"
        function is to be used. See Notes for more details.
    disp : bool, optional
        Set to True to print convergence messages from the `finish` callable.
    workers : int or map-like callable, optional
        If `workers` is an int the grid is subdivided into `workers`
        sections and evaluated in parallel (uses
        `multiprocessing.Pool <multiprocessing>`).
        Supply `-1` to use all cores available to the Process.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the grid in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        Requires that `func` be pickleable.

        .. versionadded:: 1.3.0

    Returns
    -------
    x0 : ndarray
        A 1-D array containing the coordinates of a point at which the
        objective function had its minimum value. (See `Note 1` for
        which point is returned.)
    fval : float
        Function value at the point `x0`. (Returned when `full_output` is
        True.)
    grid : tuple
        Representation of the evaluation grid. It has the same
        length as `x0`. (Returned when `full_output` is True.)
    Jout : ndarray
        Function values at each point of the evaluation
        grid, i.e., ``Jout = func(*grid)``. (Returned
        when `full_output` is True.)

    See Also
    --------
    basinhopping, differential_evolution

    Notes
    -----
    *Note 1*: The program finds the gridpoint at which the lowest value
    of the objective function occurs. If `finish` is None, that is the
    point returned. When the global minimum occurs within (or not very far
    outside) the grid's boundaries, and the grid is fine enough, that
    point will be in the neighborhood of the global minimum.

    However, users often employ some other optimization program to
    "polish" the gridpoint values, i.e., to seek a more precise
    (local) minimum near `brute's` best gridpoint.
    The `brute` function's `finish` option provides a convenient way to do
    that. Any polishing program used must take `brute's` output as its
    initial guess as a positional argument, and take `brute's` input values
    for `args` as keyword arguments, otherwise an error will be raised.
    It may additionally take `full_output` and/or `disp` as keyword arguments.

    `brute` assumes that the `finish` function returns either an
    `OptimizeResult` object or a tuple in the form:
    ``(xmin, Jmin, ... , statuscode)``, where ``xmin`` is the minimizing
    value of the argument, ``Jmin`` is the minimum value of the objective
    function, "..." may be some other returned values (which are not used
    by `brute`), and ``statuscode`` is the status code of the `finish` program.

    Note that when `finish` is not None, the values returned are those
    of the `finish` program, *not* the gridpoint ones. Consequently,
    while `brute` confines its search to the input grid points,
    the `finish` program's results usually will not coincide with any
    gridpoint, and may fall outside the grid's boundary. Thus, if a
    minimum only needs to be found over the provided grid points, make
    sure to pass in `finish=None`.

    *Note 2*: The grid of points is a `numpy.mgrid` object.
    For `brute` the `ranges` and `Ns` inputs have the following effect.
    Each component of the `ranges` tuple can be either a slice object or a
    two-tuple giving a range of values, such as (0, 5). If the component is a
    slice object, `brute` uses it directly. If the component is a two-tuple
    range, `brute` internally converts it to a slice object that interpolates
    `Ns` points from its low-value to its high-value, inclusive.

    Examples
    --------
    We illustrate the use of `brute` to seek the global minimum of a function
    of two variables that is given as the sum of a positive-definite
    quadratic and two deep "Gaussian-shaped" craters. Specifically, define
    the objective function `f` as the sum of three other functions,
    ``f = f1 + f2 + f3``. We suppose each of these has a signature
    ``(z, *params)``, where ``z = (x, y)``,  and ``params`` and the functions
    are as defined below.

    >>> params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)
    >>> def f1(z, *params):
    ...     x, y = z
    ...     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    ...     return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)

    >>> def f2(z, *params):
    ...     x, y = z
    ...     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    ...     return (-g*np.exp(-((x-h)**2 + (y-i)**2) / scale))

    >>> def f3(z, *params):
    ...     x, y = z
    ...     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    ...     return (-j*np.exp(-((x-k)**2 + (y-l)**2) / scale))

    >>> def f(z, *params):
    ...     return f1(z, *params) + f2(z, *params) + f3(z, *params)

    Thus, the objective function may have local minima near the minimum
    of each of the three functions of which it is composed. To
    use `fmin` to polish its gridpoint result, we may then continue as
    follows:

    >>> rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
    >>> from scipy import optimize
    >>> resbrute = optimize.brute(f, rranges, args=params, full_output=True,
    ...                           finish=optimize.fmin)
    >>> resbrute[0]  # global minimum
    array([-1.05665192,  1.80834843])
    >>> resbrute[1]  # function value at global minimum
    -3.4085818767

    Note that if `finish` had been set to None, we would have gotten the
    gridpoint [-1.0 1.75] where the rounded function value is -2.892.

    """
    N = len(ranges)
    if N > 40:
        raise ValueError("Brute Force not possible with more "
                         "than 40 variables.")
    lrange = list(ranges)
    for k in range(N):
        if type(lrange[k]) is not type(slice(None)):
            if len(lrange[k]) < 3:
                lrange[k] = tuple(lrange[k]) + (complex(Ns),)
            lrange[k] = slice(*lrange[k])
    if (N == 1):
        lrange = lrange[0]

    grid = np.mgrid[lrange]

    # obtain an array of parameters that is iterable by a map-like callable
    inpt_shape = grid.shape
    if (N > 1):
        grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

    wrapped_func = _Brute_Wrapper(func, args)

    # iterate over input arrays, possibly in parallel
    with MapWrapper(pool=workers) as mapper:
        Jout = np.array(list(mapper(wrapped_func, grid)))
        if (N == 1):
            grid = (grid,)
            Jout = np.squeeze(Jout)
        elif (N > 1):
            Jout = np.reshape(Jout, inpt_shape[1:])
            grid = np.reshape(grid.T, inpt_shape)

    Nshape = shape(Jout)

    indx = argmin(Jout.ravel(), axis=-1)
    Nindx = np.empty(N, int)
    xmin = np.empty(N, float)
    for k in range(N - 1, -1, -1):
        thisN = Nshape[k]
        Nindx[k] = indx % Nshape[k]
        indx = indx // thisN
    for k in range(N):
        xmin[k] = grid[k][tuple(Nindx)]

    Jmin = Jout[tuple(Nindx)]
    if (N == 1):
        grid = grid[0]
        xmin = xmin[0]

    if callable(finish):
        # set up kwargs for `finish` function
        finish_args = _getfullargspec(finish).args
        finish_kwargs = dict()
        if 'full_output' in finish_args:
            finish_kwargs['full_output'] = 1
        if 'disp' in finish_args:
            finish_kwargs['disp'] = disp
        elif 'options' in finish_args:
            # pass 'disp' as `options`
            # (e.g., if `finish` is `minimize`)
            finish_kwargs['options'] = {'disp': disp}

        # run minimizer
        res = finish(func, xmin, args=args, **finish_kwargs)

        if isinstance(res, OptimizeResult):
            xmin = res.x
            Jmin = res.fun
            success = res.success
        else:
            xmin = res[0]
            Jmin = res[1]
            success = res[-1] == 0
        if not success:
            if disp:
                print("Warning: Either final optimization did not succeed "
                      "or `finish` does not return `statuscode` as its last "
                      "argument.")

    if full_output:
        return xmin, Jmin, grid, Jout
    else:
        return xmin


class _Brute_Wrapper:
    """
    Object to wrap user cost function for optimize.brute, allowing picklability
    """

    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        # flatten needed for one dimensional case.
        return self.f(np.asarray(x).flatten(), *self.args)


def show_options(solver=None, method=None, disp=True):
    """
    Show documentation for additional options of optimization solvers.

    These are method-specific options that can be supplied through the
    ``options`` dict.

    Parameters
    ----------
    solver : str
        Type of optimization solver. One of 'minimize', 'minimize_scalar',
        'root', 'root_scalar', 'linprog', or 'quadratic_assignment'.
    method : str, optional
        If not given, shows all methods of the specified solver. Otherwise,
        show only the options for the specified method. Valid values
        corresponds to methods' names of respective solver (e.g., 'BFGS' for
        'minimize').
    disp : bool, optional
        Whether to print the result rather than returning it.

    Returns
    -------
    text
        Either None (for disp=True) or the text string (disp=False)

    Notes
    -----
    The solver-specific methods are:

    `scipy.optimize.minimize`

    - :ref:`Nelder-Mead <optimize.minimize-neldermead>`
    - :ref:`Powell      <optimize.minimize-powell>`
    - :ref:`CG          <optimize.minimize-cg>`
    - :ref:`BFGS        <optimize.minimize-bfgs>`
    - :ref:`Newton-CG   <optimize.minimize-newtoncg>`
    - :ref:`L-BFGS-B    <optimize.minimize-lbfgsb>`
    - :ref:`TNC         <optimize.minimize-tnc>`
    - :ref:`COBYLA      <optimize.minimize-cobyla>`
    - :ref:`SLSQP       <optimize.minimize-slsqp>`
    - :ref:`dogleg      <optimize.minimize-dogleg>`
    - :ref:`trust-ncg   <optimize.minimize-trustncg>`

    `scipy.optimize.root`

    - :ref:`hybr              <optimize.root-hybr>`
    - :ref:`lm                <optimize.root-lm>`
    - :ref:`broyden1          <optimize.root-broyden1>`
    - :ref:`broyden2          <optimize.root-broyden2>`
    - :ref:`anderson          <optimize.root-anderson>`
    - :ref:`linearmixing      <optimize.root-linearmixing>`
    - :ref:`diagbroyden       <optimize.root-diagbroyden>`
    - :ref:`excitingmixing    <optimize.root-excitingmixing>`
    - :ref:`krylov            <optimize.root-krylov>`
    - :ref:`df-sane           <optimize.root-dfsane>`

    `scipy.optimize.minimize_scalar`

    - :ref:`brent       <optimize.minimize_scalar-brent>`
    - :ref:`golden      <optimize.minimize_scalar-golden>`
    - :ref:`bounded     <optimize.minimize_scalar-bounded>`

    `scipy.optimize.root_scalar`

    - :ref:`bisect  <optimize.root_scalar-bisect>`
    - :ref:`brentq  <optimize.root_scalar-brentq>`
    - :ref:`brenth  <optimize.root_scalar-brenth>`
    - :ref:`ridder  <optimize.root_scalar-ridder>`
    - :ref:`toms748 <optimize.root_scalar-toms748>`
    - :ref:`newton  <optimize.root_scalar-newton>`
    - :ref:`secant  <optimize.root_scalar-secant>`
    - :ref:`halley  <optimize.root_scalar-halley>`

    `scipy.optimize.linprog`

    - :ref:`simplex           <optimize.linprog-simplex>`
    - :ref:`interior-point    <optimize.linprog-interior-point>`
    - :ref:`revised simplex   <optimize.linprog-revised_simplex>`
    - :ref:`highs             <optimize.linprog-highs>`
    - :ref:`highs-ds          <optimize.linprog-highs-ds>`
    - :ref:`highs-ipm         <optimize.linprog-highs-ipm>`

    `scipy.optimize.quadratic_assignment`

    - :ref:`faq             <optimize.qap-faq>`
    - :ref:`2opt            <optimize.qap-2opt>`

    Examples
    --------
    We can print documentations of a solver in stdout:

    >>> from scipy.optimize import show_options
    >>> show_options(solver="minimize")
    ...

    Specifying a method is possible:

    >>> show_options(solver="minimize", method="Nelder-Mead")
    ...

    We can also get the documentations as a string:

    >>> show_options(solver="minimize", method="Nelder-Mead", disp=False)
    Minimization of scalar function of one or more variables using the ...

    """
    import textwrap

    doc_routines = {
        'minimize': (
            ('bfgs', 'scipy.optimize.optimize._minimize_bfgs'),
            ('asnaq', 'scipy.optimize.optimize._minimize_aSNAQ'),
            ('olmoq', 'scipy.optimize.optimize._minimize_olmoq'),
            ('olnaq', 'scipy.optimize.optimize._minimize_olnaq'),
            ('olbfgs', 'scipy.optimize.optimize._minimize_olbfgs'),            
            ('olbfgs1', 'scipy.optimize.optimize._minimize_olbfgs1'),
            ('lmoq', 'scipy.optimize.optimize._minimize_lmoq'),
            ('lnaq', 'scipy.optimize.optimize._minimize_lnaq'),
            ('lbfgs', 'scipy.optimize.optimize._minimize_lbfgs'),
            ('omoq', 'scipy.optimize.optimize._minimize_omoq'),
            ('onaq', 'scipy.optimize.optimize._minimize_onaq'),
            ('obfgs', 'scipy.optimize.optimize._minimize_obfgs'),
            ('lsr1', 'scipy.optimize.optimize._minimize_lsr1'),
            ('sr1', 'scipy.optimize.optimize._minimize_sr1'),
            ('sr1n', 'scipy.optimize.optimize._minimize_sr1n'),
            ('mosr1', 'scipy.optimize.optimize._minimize_mosr1'),
            ('osr1', 'scipy.optimize.optimize._minimize_osr1'),
            ('osr1n', 'scipy.optimize.optimize._minimize_osr1n'),
            ('omosr1', 'scipy.optimize.optimize._minimize_omosr1'),
            ('cg', 'scipy.optimize.optimize._minimize_cg'),
            ('cobyla', 'scipy.optimize.cobyla._minimize_cobyla'),
            ('dogleg', 'scipy.optimize._trustregion_dogleg._minimize_dogleg'),
            ('l-bfgs-b', 'scipy.optimize.lbfgsb._minimize_lbfgsb'),
            ('nelder-mead', 'scipy.optimize.optimize._minimize_neldermead'),
            ('newton-cg', 'scipy.optimize.optimize._minimize_newtoncg'),
            ('powell', 'scipy.optimize.optimize._minimize_powell'),
            ('slsqp', 'scipy.optimize.slsqp._minimize_slsqp'),
            ('tnc', 'scipy.optimize.tnc._minimize_tnc'),
            ('trust-ncg',
             'scipy.optimize._trustregion_ncg._minimize_trust_ncg'),
            ('trust-constr',
             'scipy.optimize._trustregion_constr.'
             '_minimize_trustregion_constr'),
            ('trust-exact',
             'scipy.optimize._trustregion_exact._minimize_trustregion_exact'),
            ('trust-krylov',
             'scipy.optimize._trustregion_krylov._minimize_trust_krylov'),
        ),
        'root': (
            ('hybr', 'scipy.optimize.minpack._root_hybr'),
            ('lm', 'scipy.optimize._root._root_leastsq'),
            ('broyden1', 'scipy.optimize._root._root_broyden1_doc'),
            ('broyden2', 'scipy.optimize._root._root_broyden2_doc'),
            ('anderson', 'scipy.optimize._root._root_anderson_doc'),
            ('diagbroyden', 'scipy.optimize._root._root_diagbroyden_doc'),
            ('excitingmixing', 'scipy.optimize._root._root_excitingmixing_doc'),
            ('linearmixing', 'scipy.optimize._root._root_linearmixing_doc'),
            ('krylov', 'scipy.optimize._root._root_krylov_doc'),
            ('df-sane', 'scipy.optimize._spectral._root_df_sane'),
        ),
        'root_scalar': (
            ('bisect', 'scipy.optimize._root_scalar._root_scalar_bisect_doc'),
            ('brentq', 'scipy.optimize._root_scalar._root_scalar_brentq_doc'),
            ('brenth', 'scipy.optimize._root_scalar._root_scalar_brenth_doc'),
            ('ridder', 'scipy.optimize._root_scalar._root_scalar_ridder_doc'),
            ('toms748', 'scipy.optimize._root_scalar._root_scalar_toms748_doc'),
            ('secant', 'scipy.optimize._root_scalar._root_scalar_secant_doc'),
            ('newton', 'scipy.optimize._root_scalar._root_scalar_newton_doc'),
            ('halley', 'scipy.optimize._root_scalar._root_scalar_halley_doc'),
        ),
        'linprog': (
            ('simplex', 'scipy.optimize._linprog._linprog_simplex_doc'),
            ('interior-point', 'scipy.optimize._linprog._linprog_ip_doc'),
            ('revised simplex', 'scipy.optimize._linprog._linprog_rs_doc'),
            ('highs-ipm', 'scipy.optimize._linprog._linprog_highs_ipm_doc'),
            ('highs-ds', 'scipy.optimize._linprog._linprog_highs_ds_doc'),
            ('highs', 'scipy.optimize._linprog._linprog_highs_doc'),
        ),
        'quadratic_assignment': (
            ('faq', 'scipy.optimize._qap._quadratic_assignment_faq'),
            ('2opt', 'scipy.optimize._qap._quadratic_assignment_2opt'),
        ),
        'minimize_scalar': (
            ('brent', 'scipy.optimize.optimize._minimize_scalar_brent'),
            ('bounded', 'scipy.optimize.optimize._minimize_scalar_bounded'),
            ('golden', 'scipy.optimize.optimize._minimize_scalar_golden'),
        ),
    }

    if solver is None:
        text = ["\n\n\n========\n", "minimize\n", "========\n"]
        text.append(show_options('minimize', disp=False))
        text.extend(["\n\n===============\n", "minimize_scalar\n",
                     "===============\n"])
        text.append(show_options('minimize_scalar', disp=False))
        text.extend(["\n\n\n====\n", "root\n",
                     "====\n"])
        text.append(show_options('root', disp=False))
        text.extend(['\n\n\n=======\n', 'linprog\n',
                     '=======\n'])
        text.append(show_options('linprog', disp=False))
        text = "".join(text)
    else:
        solver = solver.lower()
        if solver not in doc_routines:
            raise ValueError('Unknown solver %r' % (solver,))

        if method is None:
            text = []
            for name, _ in doc_routines[solver]:
                text.extend(["\n\n" + name, "\n" + "=" * len(name) + "\n\n"])
                text.append(show_options(solver, name, disp=False))
            text = "".join(text)
        else:
            method = method.lower()
            methods = dict(doc_routines[solver])
            if method not in methods:
                raise ValueError("Unknown method %r" % (method,))
            name = methods[method]

            # Import function object
            parts = name.split('.')
            mod_name = ".".join(parts[:-1])
            __import__(mod_name)
            obj = getattr(sys.modules[mod_name], parts[-1])

            # Get doc
            doc = obj.__doc__
            if doc is not None:
                text = textwrap.dedent(doc).strip()
            else:
                text = ""

    if disp:
        print(text)
        return
    else:
        return text


def main():
    import time

    times = []
    algor = []
    x0 = [0.8, 1.2, 0.7]
    print("Nelder-Mead Simplex")
    print("===================")
    start = time.time()
    x = fmin(rosen, x0)
    print(x)
    times.append(time.time() - start)
    algor.append('Nelder-Mead Simplex\t')

    print()
    print("Powell Direction Set Method")
    print("===========================")
    start = time.time()
    x = fmin_powell(rosen, x0)
    print(x)
    times.append(time.time() - start)
    algor.append('Powell Direction Set Method.')

    print()
    print("Nonlinear CG")
    print("============")
    start = time.time()
    x = fmin_cg(rosen, x0, fprime=rosen_der, maxiter=200)
    print(x)
    times.append(time.time() - start)
    algor.append('Nonlinear CG     \t')

    print()
    print("BFGS Quasi-Newton")
    print("=================")
    start = time.time()
    x = fmin_bfgs(rosen, x0, fprime=rosen_der, maxiter=80)
    print(x)
    times.append(time.time() - start)
    algor.append('BFGS Quasi-Newton\t')

    print()
    print("BFGS approximate gradient")
    print("=========================")
    start = time.time()
    x = fmin_bfgs(rosen, x0, gtol=1e-4, maxiter=100)
    print(x)
    times.append(time.time() - start)
    algor.append('BFGS without gradient\t')

    print()
    print("Newton-CG with Hessian product")
    print("==============================")
    start = time.time()
    x = fmin_ncg(rosen, x0, rosen_der, fhess_p=rosen_hess_prod, maxiter=80)
    print(x)
    times.append(time.time() - start)
    algor.append('Newton-CG with hessian product')

    print()
    print("Newton-CG with full Hessian")
    print("===========================")
    start = time.time()
    x = fmin_ncg(rosen, x0, rosen_der, fhess=rosen_hess, maxiter=80)
    print(x)
    times.append(time.time() - start)
    algor.append('Newton-CG with full Hessian')

    print()
    print("\nMinimizing the Rosenbrock function of order 3\n")
    print(" Algorithm \t\t\t       Seconds")
    print("===========\t\t\t      =========")
    for k in range(len(algor)):
        print(algor[k], "\t -- ", times[k])


if __name__ == "__main__":
    main()
