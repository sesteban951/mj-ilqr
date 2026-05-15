"""
Box-constrained quadratic program solver.

Solves
    min   0.5 * x^T H x + q^T x
    s.t.  lb <= x <= ub                                     (elementwise)

via the projected-Newton active-set algorithm
(Bertsekas 1982; Tassa, Mansard & Todorov 2014, Algorithm I).

Returns the optimum x*, the free-index mask f at x*, and the Cholesky
factor of H_{ff} so that callers can compute the feedback gain
    K_f = - H_{ff}^{-1} Q_{ux,f}
without re-factorizing.
"""

import numpy as np

def boxqp(H, q, lb, ub, x0=None,
          tol=1e-5, max_iter=100,
          armijo_c=0.1, step_dec=0.5, min_step=1e-22):
    """
    Args:
        H:        (n, n) symmetric (assumed PSD on the free subspace)
        q:        (n,)
        lb, ub:   (n,) box bounds, lb <= ub elementwise
        x0:       (n,) optional warm-start; clipped into [lb, ub]    
        tol:      stop when ||g_f||_inf < tol  (g_f = free-subspace gradient)
        max_iter: outer-iteration cap                                
        armijo_c: Armijo sufficient-decrease constant                
        step_dec: backtracking shrink factor, alpha <- step_dec*alpha 
        min_step: minimum step size before declaring failure         

    Returns:
        x:        (n,) optimum (always feasible: lb <= x <= ub)
        free:     (n,) bool mask of un-clamped indices at returned x
        L_ff:     (nf, nf) lower-triangular Cholesky of H[free, free];
                  None if nf == 0 or H[free, free] not PD
        n_iter:   outer iterations used (0..max_iter)
        status:   "ok"          converged (||g_f||_inf < tol or fully clamped)
                  "not_descent" H[free, free] failed Cholesky -> not PD on free subspace
                  "tiny_step"   line search shrunk below min_step without Armijo accept
                  "max_iter"    hit outer-iteration cap
    """
    n  = H.shape[0]
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if x0 is None:
        x = np.clip(np.zeros(n), lb, ub)
    else:
        x = np.clip(np.asarray(x0, dtype=float).copy(), lb, ub)

    # tolerance for "x is at a bound" (relative to box width, with floor)
    bound_tol = 1e-12 * np.maximum(1.0, np.abs(ub - lb))

    L_ff = None
    free = np.ones(n, dtype=bool)

    for it in range(max_iter):
        # gradient
        grad = q + H @ x

        # active-set identification (Tassa eq 15): clamped iff at bound AND gradient pushes outward
        at_lb   = (x - lb <= bound_tol) & (grad > 0.0)
        at_ub   = (ub - x <= bound_tol) & (grad < 0.0)
        clamped = at_lb | at_ub
        free    = ~clamped
        nf      = int(free.sum())  # number of free variables

        # all clamped: stationary on the boundary
        if nf == 0:
            return x, free, None, it, "ok"

        # partition the gradient into free and clamped components
        grad_free = grad[free]
        if np.linalg.norm(grad_free, np.inf) < tol:
            Hff = H[np.ix_(free, free)]
            try:
                L_ff = np.linalg.cholesky(Hff)
            except np.linalg.LinAlgError:
                L_ff = None
            return x, free, L_ff, it, "ok"

        # Newton step in the free subspace:  H_ff dxf = -g_f
        Hff = H[np.ix_(free, free)]
        try:
            L_ff = np.linalg.cholesky(Hff)
        except np.linalg.LinAlgError:
            return x, free, None, it, "not_descent"

        # delta_xf = - H_ff^{-1} g_f  via the Cholesky factorization
        z   = np.linalg.solve(L_ff,  -grad_free)
        dxf = np.linalg.solve(L_ff.T, z)

        # assemble the full step, with zeros in the clamped dimensions, eq (17)
        dx       = np.zeros(n)
        dx[free] = dxf

        # projected backtracking line search with Armijo (Tassa eq 19)
        f0    = 0.5 * x @ H @ x + q @ x
        alpha = 1.0
        accepted = False
        while alpha > min_step:
            # compute change in descent
            x_new = np.clip(x + alpha * dx, lb, ub)
            f_new = 0.5 * x_new @ H @ x_new + q @ x_new
            df    = f0 - f_new
            denom = grad @ (x - x_new)            # > 0 for a true descent projection

            # check if Armijo satisfied
            if denom > 0.0 and df > armijo_c * denom:
                x = x_new
                accepted = True
                break
            alpha *= step_dec

        if not accepted:
            return x, free, L_ff, it, "tiny_step"

    return x, free, L_ff, max_iter, "max_iter"
