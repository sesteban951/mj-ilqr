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

        X_new    = np.empty((N + 1, nx)); X_new[0] = x0
        U_new    = np.empty((N, nu))
        xk = x0.copy()

        for k in range(N):
            dx = xk - X_nom[k]
            du = alpha * k_ff[k] + K_fb[k] @ dx
            uk = np.clip(U_nom[k] + du, self.dyn.u_lb, self.dyn.u_ub)
            U_new[k] = uk
            xk = self.dyn.f_disc(xk, uk, clip=True)
            X_new[k + 1] = xk

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

        k_ff_seq = np.zeros((N, nu))
        K_fb_seq = np.zeros((N, nu, nx))

        # terminal cost-to-go
        Vx  = self.lf_x (X[-1:])[0]               # (nx,)
        Vxx = self.lf_xx(X[-1:])[0]               # (nx, nx)

        mu_I = mu * np.eye(nu)
        dV1  = 0.0

        # main backwards pass: optimal feedforward k_ff and feedback K_fb at each step
        warm = None
        for k in range(N - 1, -1, -1):
            Ad, Bd = Ad_seq[k], Bd_seq[k]
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

            if status != "ok":
                # QP did not reach a stationary point (not_descent / tiny_step / max_iter)
                # -> caller bumps mu and retries; using a sub-optimal k_ff would corrupt
                # dV1 and the value-function update
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