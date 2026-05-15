##
#
# iLQR algorithm.
#
##

# abstract classes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import copy

# standard
import numpy as np

# dynamics
from dynamics import MJDynamics_CPU, MJDynamicsConfig

# box-constrained QP for the feedforward step
from boxqp import boxqp


###############################################################
# iLQR BASE CLASS
###############################################################

@dataclass
class iLQRConfig:
    # convergence
    max_iter: int   = 150
    tol:      float = 1e-2

    # Levenberg-Marquardt regularization on Quu
    mu:        float = 1.0
    mu_min:    float = 1e-6
    mu_max:    float = 1e10
    mu_factor: float = 2.0

    # geometric backtracking Armijo line search
    alpha_init: float = 1.0
    alpha_beta: float = 0.5
    alpha_min:  float = 1e-4
    armijo_c:   float = 1e-4

    # linearization method: "sampling" or "mujoco_fd"
    linearize_method: str = "sampling"

    # sampling-based linearization knobs (consumed by MJDynamics.linearize)
    sampling_K:   int   = 256
    sampling_eps: float = 5e-2
    sampling_reg: float = 1e-8
    sampling_rng: np.random.Generator = field(default_factory=np.random.default_rng)

    # mujoco FD linearization knobs (consumed by MJDynamics.linearize)
    fd_eps:      float = 5e-2
    fd_centered: bool  = True


class iLQRBase(ABC):

    def __init__(self, dyn_config: MJDynamicsConfig, ilqr_config: iLQRConfig):

        # iLQR config
        self.config = copy.deepcopy(ilqr_config)
        
        # Dynamics from config
        self.dyn = MJDynamics_CPU(dyn_config)

    
    # linearize the discrete dynamics at every knot of the nominal trajectory
    def linearize_about_trajectory(self, X, U):
        """
        For k = 0..N-1, estimate (Ad_k, Bd_k) such that the perturbation map
            dx_{k+1} ≈ Ad_k dx_k + Bd_k du_k
        via self.dyn.linearize (which dispatches on self.config.linearize_method).

        Args:
            X: (N+1, nx) nominal state trajectory
            U: (N,   nu) nominal control sequence
        Returns:
            Ad_seq: (N, nx, nx)
            Bd_seq: (N, nx, nu)
        """
        N      = U.shape[0]
        nx, nu = self.dyn.nx, self.dyn.nu

        # initialize arrays
        Ad_seq = np.empty((N, nx, nx))
        Bd_seq = np.empty((N, nx, nu))

        # linearize at each knot
        for k in range(N):
            Ad, Bd = self.dyn.linearize(X[k], U[k], self.config)
            Ad_seq[k] = Ad
            Bd_seq[k] = Bd

        return Ad_seq, Bd_seq


    # closed-loop rollout under the iLQR control law for line-search step alpha
    def forward_pass(self, x0, X_nom, U_nom, k_ff, K_fb, alpha):
        """
        Closed-loop rollout:
            dx[k]      = x_new[k] - x_nom[k]
            du[k]      = alpha * k_ff[k] + K_fb[k] @ dx[k]
            u_new[k]   = clamp(u_nom[k] + du[k], u_lb, u_ub)
            x_new[k+1] = f_disc(x_new[k], u_new[k])    # clip=True (physical rollout)

        Args:
            x0:    (nx,)        initial state
            X_nom: (N+1, nx)    nominal state trajectory
            U_nom: (N,   nu)    nominal control sequence
            k_ff:  (N,   nu)    feedforward gains from backward pass
            K_fb:  (N,   nu, nx) feedback gains from backward pass
            alpha: float        line-search step size in [alpha_min, alpha_init]
        Returns:
            X_new: (N+1, nx)
            U_new: (N,   nu)    actually-applied (clamped) controls
            J_new: scalar       total trajectory cost
        """
        N, nu = U_nom.shape
        nx    = X_nom.shape[1]

        # initialize trajectory arrays
        X_new    = np.empty((N + 1, nx)); X_new[0] = x0
        U_new    = np.empty((N, nu))
        
        # rollout under the iLQR inputs
        xk = x0.copy()
        for k in range(N):
            dx = xk - X_nom[k]
            du = alpha * k_ff[k] + K_fb[k] @ dx
            uk = np.clip(U_nom[k] + du, self.dyn.u_lb, self.dyn.u_ub)
            xk = self.dyn.f_disc(xk, uk, clip=True)
            U_new[k] = uk
            X_new[k + 1] = xk

        # evaluate the new trajectory cost
        J_new = self.cost(X_new, U_new)

        return X_new, U_new, J_new

    # Riccati-style backward sweep with LM regularization + box-constrained QP at each step
    def backward_pass(self, X, U, Ad_seq, Bd_seq, mu):
        """
        Riccati-style backward sweep with Levenberg-Marquardt regularization and
        box-constrained control updates (Tassa, Mansard & Todorov 2014).

        For k = N-1..0, build the local Q-function around (X[k], U[k]) using the
        cost derivatives and the perturbation dynamics dx_{k+1} = Ad_k dx_k + Bd_k du_k,
        then compute feedforward / feedback gains
            du_k = k_ff[k] + K_fb[k] dx_k
        by solving the box-constrained QP
            min   0.5 * du^T Quu_reg du + Qu^T du
            s.t.  u_lb - U[k] <= du <= u_ub - U[k].
        The feedback gain rows for clamped controls are nullified (Tassa III-C.2),
        and the free-row gains use the reduced Hessian:
            K_fb[k][f, :] = -Quu_reg[f, f]^{-1} Qux[f, :].
        Also accumulates the first-order directional derivative of the trajectory
        cost along the iLQR search direction (used by the outer Armijo test):
            dV1 = sum_k  k_ff_k^T Q_u_k     (= dJ_model/dalpha at alpha=0,
                                              < 0 for a descent direction)

        Args:
            X, U:             nominal trajectory and controls
            Ad_seq, Bd_seq:   per-step Jacobians from linearize_about_trajectory
            mu:               scalar LM regularization on Quu
        Returns:
            k_ff_seq: (N, nu)
            K_fb_seq: (N, nu, nx)
            dV1:      float   first-order directional derivative along k_ff
            success:  bool    False if any QP did not converge to a stationary point
        """
        N      = U.shape[0]
        nx     = X.shape[1]
        nu     = U.shape[1]
        u_lb   = self.dyn.u_lb
        u_ub   = self.dyn.u_ub

        # bulk-evaluate cost derivatives along the nominal trajectory
        X_stage = X[:-1]                          # (N, nx)
        lx_all  = self.l_x (X_stage, U)           # (N, nx)
        lxx_all = self.l_xx(X_stage, U)           # (N, nx, nx)
        lu_all  = self.l_u (X_stage, U)           # (N, nu)
        luu_all = self.l_uu(X_stage, U)           # (N, nu, nu)
        lux_all = self.l_ux(X_stage, U)           # (N, nu, nx)

        # initialize feedforward / feedback arrays
        k_ff_seq = np.zeros((N, nu))
        K_fb_seq = np.zeros((N, nu, nx))

        # terminal cost-to-go
        Vx  = self.lf_x (X[-1:])[0]               # (nx,)
        Vxx = self.lf_xx(X[-1:])[0]               # (nx, nx)

        # initialize LM regularization matrix
        mu_I = mu * np.eye(nu)
        dV1  = 0.0

        # main backwards pass: optimal feedforward k_ff and feedback K_fb at each step
        warm = None
        for k in range(N - 1, -1, -1):
            Ad,  Bd  = Ad_seq[k],  Bd_seq[k]
            lx,  lu  = lx_all[k],  lu_all[k]
            lxx, luu = lxx_all[k], luu_all[k]
            lux      = lux_all[k]

            # Q-function derivatives
            Qx  = lx  + Ad.T @ Vx
            Qu  = lu  + Bd.T @ Vx
            Qxx = lxx + Ad.T @ Vxx @ Ad
            Qux = lux + Bd.T @ Vxx @ Ad
            Quu = luu + Bd.T @ Vxx @ Bd

            # Tikhonov-regularize Quu for the QP Hessian
            Quu_reg = Quu + mu_I

            # box-constrained QP for the feedforward step:
            #   min  0.5 du^T Quu_reg du + Qu^T du   s.t.   u_lb - U[k] <= du <= u_ub - U[k]
            lb_k = u_lb - U[k]
            ub_k = u_ub - U[k]
            k_ff, free, L_ff, _, status = boxqp(Quu_reg, Qu, lb_k, ub_k, x0=warm)
            warm = k_ff

            # QP failure
            if status != "ok":
                return k_ff_seq, K_fb_seq, 0.0, False

            # feedback: zero rows for clamped controls, reduced-Hessian solve for free rows
            K_fb = np.zeros((nu, nx))
            if L_ff is not None and free.any():
                Qux_f = Qux[free]                                   # (nf, nx)
                z     = np.linalg.solve(L_ff,    Qux_f)
                y     = np.linalg.solve(L_ff.T,  z)                 # Quu_reg[f,f]^{-1} Qux[f,:]
                K_fb[free] = -y

            k_ff_seq[k] = k_ff
            K_fb_seq[k] = K_fb

            # first-order directional derivative of trajectory cost along k_ff
            dV1 += float(k_ff @ Qu)

            # value-function update (general form, doesn't assume optimal gains)
            Vx  = Qx  + K_fb.T @ Quu_reg @ k_ff + K_fb.T @ Qu  + Qux.T @ k_ff
            Vxx = Qxx + K_fb.T @ Quu_reg @ K_fb + K_fb.T @ Qux + Qux.T @ K_fb
            Vxx = 0.5 * (Vxx + Vxx.T)                       # symmetrize

        return k_ff_seq, K_fb_seq, dV1, True


    # outer iLQR loop: linearize -> backward pass -> Armijo line search, with LM
    def solve(self, x0, U_init):
        """
        Run iLQR to convergence (or max_iter) starting from an initial control guess.

        Args:
            x0:     (nx,) initial state
            U_init: (N, nu) initial control sequence (will be clipped to [u_lb, u_ub])
        Returns:
            X:      (N+1, nx) final state trajectory
            U:      (N, nu)   final control sequence
            J_hist: list[float]; J_hist[0] is the initial cost
        """
        cfg = self.config
        dyn = self.dyn

        N      = U_init.shape[0]
        nx, nu = dyn.nx, dyn.nu

        max_iter   = cfg.max_iter
        tol        = cfg.tol
        mu         = cfg.mu
        mu_min     = cfg.mu_min
        mu_max     = cfg.mu_max
        mu_factor  = cfg.mu_factor
        alpha_init = cfg.alpha_init
        alpha_beta = cfg.alpha_beta
        alpha_min  = cfg.alpha_min
        c1         = cfg.armijo_c

        # initial open-loop rollout of U_init (with clipping)
        X = np.empty((N + 1, nx)); 
        U = np.empty((N, nu))
        X[0] = x0
        xk = x0.copy()
        for k in range(N):
            U[k] = np.clip(U_init[k], dyn.u_lb, dyn.u_ub)
            xk = dyn.f_disc(xk, U[k], clip=True)
            X[k + 1] = xk
        J = self.cost(X, U)
        J_hist = [float(J)]
        print(f"[iLQR] iter   0: J={float(J):.4f}")

        # main iLQR loop
        it = 0
        while it < max_iter:
            # linearize + backward pass
            Ad_seq, Bd_seq = self.linearize_about_trajectory(X, U)
            k_ff, K_fb, dV1, ok = self.backward_pass(X, U, Ad_seq, Bd_seq, mu)

            # backward pass failed -> bump mu and retry (no iteration bump)
            if not ok:
                mu = min(mu * mu_factor, mu_max)
                print(f"[iLQR] iter {it:3d}: backward pass failed, mu -> {mu:.2e}")
                if mu >= mu_max:
                    print(f"[iLQR] mu hit mu_max={mu_max:.2e}; stopping.")
                    break
                continue

            # Classic Armijo line search with geometric backtracking:
            #   start at alpha = alpha_init; multiply by alpha_beta on each rejection;
            #   stop when alpha < alpha_min (-> reject this backward pass, bump mu).
            # First-order predicted reduction along the iLQR search direction:
            #   E(a) = -a * dV1,   with dV1 = sum_k k_ff_k^T Q_u_k  (= dJ/da at a=0).
            # Accept iff   (J - J_try) >= c1 * E(a).
            accepted     = False
            alpha_used   = None
            dJ           = 0.0
            z_used       = 0.0
            exp_red_used = 0.0

            # check if the search direction is a descent direction
            if dV1 >= 0.0:
                # not a descent direction; bail out so the outer loop bumps mu
                # (Quu_reg likely lost positive definiteness)
                a = alpha_min - 1.0
            else:
                a = alpha_init

            while a >= alpha_min:
                exp_red = -a * dV1                   # first-order expected reduction (> 0)
                X_try, U_try, J_try = self.forward_pass(x0, X, U, k_ff, K_fb, a)
                dJ_actual = float(J) - float(J_try)
                if dJ_actual >= c1 * exp_red:
                    z_used       = dJ_actual / exp_red
                    exp_red_used = exp_red
                    dJ           = dJ_actual
                    X, U, J      = X_try, U_try, J_try
                    accepted     = True
                    alpha_used   = a
                    break
                a *= alpha_beta

            # good step: decrease mu, count iteration, log, check convergence
            if accepted:
                it += 1
                mu = max(mu / mu_factor, mu_min)
                J_hist.append(float(J))
                print(f"[iLQR] iter {it:3d}: J={float(J):.4f}  dJ={dJ:.3e}  "
                      f"alpha={alpha_used:.4f}  z={z_used:.2f}  "
                      f"E={exp_red_used:.3e}  mu={mu:.2e}")
                if abs(dJ) < tol or (exp_red_used > 0.0 and exp_red_used < tol):
                    print(f"[iLQR] converged: |dJ|={abs(dJ):.2e}, "
                          f"E={exp_red_used:.2e} < tol={tol:.2e}")
                    break
            # bad step: increase mu and retry (no iteration bump)
            else:
                mu = min(mu * mu_factor, mu_max)
                print(f"[iLQR] iter {it:3d}: line search failed "
                      f"(dV1={dV1:.2e}), mu -> {mu:.2e}")
                if mu >= mu_max:
                    print(f"[iLQR] mu hit mu_max={mu_max:.2e}; stopping.")
                    break

        print(f"[iLQR] done in {len(J_hist) - 1} accepted iterations. "
              f"J: {J_hist[0]:.4f} -> {J_hist[-1]:.4f}")
        return X, U, J_hist


    # evaluate total trajectory cost J(X, U)
    @abstractmethod
    def cost(self, X, U) -> float:
        pass

    # stage cost l(x, u)
    @abstractmethod
    def l(self, x, u) -> np.ndarray:
        pass

    # stage cost gradient w.r.t. state, d l / d x
    @abstractmethod
    def l_x(self, x, u) -> np.ndarray:
        pass

    # stage cost Hessian w.r.t. state, d^2 l / d x^2
    @abstractmethod
    def l_xx(self, x, u) -> np.ndarray:
        pass

    # stage cost gradient w.r.t. control, d l / d u
    @abstractmethod
    def l_u(self, x, u) -> np.ndarray:
        pass

    # stage cost Hessian w.r.t. control, d^2 l / d u^2
    @abstractmethod
    def l_uu(self, x, u) -> np.ndarray:
        pass

    # stage cost cross-Hessian, d^2 l / d u d x
    @abstractmethod
    def l_ux(self, x, u) -> np.ndarray:
        pass

    # terminal cost lf(x)
    @abstractmethod
    def lf(self, x) -> np.ndarray:
        pass

    # terminal cost gradient, d lf / d x
    @abstractmethod
    def lf_x(self, x) -> np.ndarray:
        pass

    # terminal cost Hessian, d^2 lf / d x^2
    @abstractmethod
    def lf_xx(self, x) -> np.ndarray:
        pass