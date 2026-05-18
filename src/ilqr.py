##
#
# iLQR algorithm.
#
# Upgrades over textbook Tassa-2014, ported from mujoco_mpc's iLQG planner:
#   1. quadratic expected improvement  E(α) = -α (dV1 + α dV2)
#   2. Tassa z-ratio adaptive mu schedule (rate itself accelerates)
#   3. value-regularization mode      Vxx_reg = Vxx + mu I
#   4. inline Cholesky-failure retry in backward pass
#   5. parallel multi-α line search via mujoco.rollout.Rollout
#
##

# abstract classes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import copy

# standard
import numpy as np
import mujoco

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

    # Levenberg-Marquardt regularization
    mu:        float = 1.0
    mu_min:    float = 1e-6
    mu_max:    float = 1e10
    mu_factor: float = 2.0
    regularization_type: str = "control"   # "control" (mu on Quu) | "value" (mu on Vxx)

    # backward-pass inline retry cap on Cholesky / boxQP failure
    max_bp_iter: int = 5

    # parallel multi-α line search (replaces geometric Armijo backtracking)
    num_linesearch_candidates: int   = 8
    min_linesearch_step:       float = 1e-3

    # silence per-iteration prints in solve() (useful for MPC inner solves)
    verbose: bool = True

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
        self.config = copy.deepcopy(ilqr_config)
        self.dyn    = MJDynamics_CPU(dyn_config)


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

        Ad_seq = np.empty((N, nx, nx))
        Bd_seq = np.empty((N, nx, nu))
        for k in range(N):
            Ad, Bd = self.dyn.linearize(X[k], U[k], self.config)
            Ad_seq[k] = Ad
            Bd_seq[k] = Bd
        return Ad_seq, Bd_seq


    # single-trajectory closed-loop rollout (used for tests / single-α probes)
    def forward_pass(self, x0, X_nom, U_nom, k_ff, K_fb, alpha):
        """
        Closed-loop rollout:
            dx[k]      = x_new[k] - x_nom[k]
            du[k]      = alpha * k_ff[k] + K_fb[k] @ dx[k]
            u_new[k]   = clamp(u_nom[k] + du[k], u_lb, u_ub)
            x_new[k+1] = f_disc(x_new[k], u_new[k])    # clip=True (physical rollout)
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
            xk = self.dyn.f_disc(xk, uk, clip=True)
            U_new[k] = uk
            X_new[k + 1] = xk

        J_new = self.cost(X_new, U_new)
        return X_new, U_new, J_new


    # parallel multi-α closed-loop rollouts via the batched mujoco rollout pool
    def forward_pass_batched(self, x0, X_nom, U_nom, k_ff, K_fb, alphas):
        """
        Closed-loop rollouts for K candidate step sizes, run in parallel via the
        thread pool that backs self.dyn._rollout. One mj_step per timestep across
        all K candidates at once.

        Mirrors mujoco_mpc/mjpc/planners/ilqg/planner.cc:ActionRollouts (parallel
        ThreadPool dispatch of num_trajectory_ rollouts at log-spaced alphas).

        Args:
            x0:     (nx,) initial state shared by all candidates
            X_nom:  (N+1, nx) nominal state trajectory
            U_nom:  (N,   nu) nominal controls
            k_ff:   (N,   nu) feedforward gains
            K_fb:   (N,   nu, nx) feedback gains
            alphas: (K,)  step sizes; the last one should be 0.0 to keep the
                          nominal trajectory as a candidate (safety net).
        Returns:
            X_all:  (K, N+1, nx)
            U_all:  (K, N,   nu)
            J_all:  (K,)    total cost; np.inf for candidates that diverged
            failed: (K,)    bool, True if non-finite state encountered
        """
        K   = len(alphas)
        alphas = np.asarray(alphas, dtype=np.float64).reshape(K)
        N, nu = U_nom.shape
        nx    = X_nom.shape[1]
        dyn   = self.dyn

        X_all  = np.empty((K, N + 1, nx))
        U_all  = np.empty((K, N, nu))
        failed = np.zeros(K, dtype=bool)
        X_all[:, 0] = x0[None, :]
        x_curr      = X_all[:, 0].copy()                       # (K, nx)

        # FULLPHYSICS state template (only qpos/qvel are perturbed per candidate)
        spec    = dyn._full_state_spec
        nstate  = dyn._nstate
        qs, vs  = dyn._qpos_slice, dyn._qvel_slice
        dyn.set_state(x0)
        ref_state = np.zeros(nstate, dtype=np.float64)
        mujoco.mj_getState(dyn.model, dyn.data, ref_state, spec)
        init_states = np.tile(ref_state, (K, 1))               # (K, nstate)
        control     = np.empty((K, 1, nu), dtype=np.float64)

        for k in range(N):
            # K candidate controls (clipped to box bounds)
            for i in range(K):
                dx = x_curr[i] - X_nom[k]
                du = alphas[i] * k_ff[k] + K_fb[k] @ dx
                u  = np.clip(U_nom[k] + du, dyn.u_lb, dyn.u_ub)
                U_all[i, k]   = u
                control[i, 0] = u

            # pack K initial states for this step
            init_states[:, qs] = x_curr[:, :dyn.nq]
            init_states[:, vs] = x_curr[:, dyn.nq:]

            # batched one-step rollout across the thread pool
            states, _ = dyn._rollout.rollout(
                dyn.model, dyn._rollout_data, init_states, control, nstep=1
            )                                                  # (K, 1, nstate)

            # extract next [qpos; qvel] for each candidate
            nxt = np.concatenate([states[:, 0, qs], states[:, 0, vs]], axis=1)

            # divergence guard: park failed candidates at x0 to avoid NaN propagation
            bad = ~np.all(np.isfinite(nxt), axis=1)
            if bad.any():
                failed |= bad
                nxt[bad] = x0

            X_all[:, k + 1] = nxt
            x_curr          = nxt

        # cost per candidate; failed -> inf
        J_all = np.full(K, np.inf, dtype=np.float64)
        for i in range(K):
            if not failed[i]:
                J_all[i] = self.cost(X_all[i], U_all[i])
        return X_all, U_all, J_all, failed


    # Riccati-style backward sweep with inline retry, value- or control-LM, and dV2
    def backward_pass(self, X, U, Ad_seq, Bd_seq, mu):
        """
        Riccati-style backward sweep with:
          - control- or value-regularization (cfg.regularization_type)
          - inline mu-bump retry on Cholesky/boxQP failure (cap = cfg.max_bp_iter)
          - quadratic expected-improvement tracking dV = (dV1, dV2)
          - box-constrained feedforward via projected-Newton boxQP

        For k = N-1..0, build the local Q-function around (X[k], U[k]) and compute
            du_k = k_ff[k] + K_fb[k] dx_k
        by solving the box-constrained QP
            min   0.5 du^T Quu_reg du + Qu^T du
            s.t.  u_lb - U[k] <= du <= u_ub - U[k].

        Regularization (mujoco_mpc/mjpc/planners/ilqg/backward_pass.cc:116-153):
          "control":  Quu_reg = Quu + mu I,                 Qux_reg = Qux
          "value":    Vxx_reg = Vxx + mu I, then
                      Quu_reg = luu + B^T Vxx_reg B,
                      Qux_reg = lux + B^T Vxx_reg A.

        Args:
            X, U:           nominal trajectory and controls
            Ad_seq, Bd_seq: per-step Jacobians from linearize_about_trajectory
            mu:             scalar LM regularization (may be increased on retry)
        Returns:
            k_ff_seq: (N, nu)
            K_fb_seq: (N, nu, nx)
            dV1:      float    sum_k k_ff_k^T Q_u_k           (linear; < 0 for descent)
            dV2:      float    sum_k 0.5 k_ff_k^T Quu_reg k_ff_k  (quadratic; >= 0)
            mu:       float    possibly-bumped regularization after inline retries
            success:  bool     False if max_bp_iter retries exhausted
        """
        cfg   = self.config
        N     = U.shape[0]
        nx    = X.shape[1]
        nu    = U.shape[1]
        u_lb  = self.dyn.u_lb
        u_ub  = self.dyn.u_ub

        reg_type    = cfg.regularization_type
        max_bp_iter = cfg.max_bp_iter
        mu_max      = cfg.mu_max
        mu_factor   = cfg.mu_factor

        # bulk-evaluate cost derivatives along the nominal trajectory
        X_stage = X[:-1]                          # (N, nx)
        lx_all  = self.l_x (X_stage, U)
        lxx_all = self.l_xx(X_stage, U)
        lu_all  = self.l_u (X_stage, U)
        luu_all = self.l_uu(X_stage, U)
        lux_all = self.l_ux(X_stage, U)

        # terminal cost-to-go (recomputed at every retry — cheap)
        lfx_T  = self.lf_x (X[-1:])[0]
        lfxx_T = self.lf_xx(X[-1:])[0]

        eye_x = np.eye(nx)
        eye_u = np.eye(nu)

        k_ff_seq = np.zeros((N, nu))
        K_fb_seq = np.zeros((N, nu, nx))

        # Inline retry loop: if any QP fails, bump mu and restart from k = N-1.
        # mujoco_mpc/mjpc/planners/ilqg/backward_pass.cc:445-509
        for _ in range(max_bp_iter):
            Vx  = lfx_T.copy()
            Vxx = lfxx_T.copy()
            dV1 = 0.0
            dV2 = 0.0
            warm = None
            success = True

            for k in range(N - 1, -1, -1):
                Ad,  Bd  = Ad_seq[k],  Bd_seq[k]
                lx,  lu  = lx_all[k],  lu_all[k]
                lxx, luu = lxx_all[k], luu_all[k]
                lux      = lux_all[k]

                # Q-function derivatives — UNREGULARIZED throughout.
                # (mjpc uses unreg Quut/Qxut for dV and the V-function update;
                #  reg versions enter only the QP solve direction.)
                Qx  = lx  + Ad.T @ Vx
                Qu  = lu  + Bd.T @ Vx
                Qxx = lxx + Ad.T @ Vxx @ Ad
                Qux = lux + Bd.T @ Vxx @ Ad
                Quu = luu + Bd.T @ Vxx @ Bd

                # regularized versions for the QP solve only
                # (mujoco_mpc/mjpc/planners/ilqg/backward_pass.cc:116-153)
                if reg_type == "value":
                    Vxx_reg = Vxx + mu * eye_x
                    Qux_reg = lux + Bd.T @ Vxx_reg @ Ad
                    Quu_reg = luu + Bd.T @ Vxx_reg @ Bd
                else:  # "control"
                    Qux_reg = Qux
                    Quu_reg = Quu + mu * eye_u

                # box-constrained QP for the feedforward step
                lb_k = u_lb - U[k]
                ub_k = u_ub - U[k]
                k_ff, free, L_ff, _, status = boxqp(Quu_reg, Qu, lb_k, ub_k, x0=warm)
                warm = k_ff

                if status != "ok":
                    success = False
                    break

                # feedback: zero rows for clamped controls, reduced-Hessian solve for free
                K_fb = np.zeros((nu, nx))
                if L_ff is not None and free.any():
                    Qux_f = Qux_reg[free]
                    z_sol = np.linalg.solve(L_ff,    Qux_f)
                    y_sol = np.linalg.solve(L_ff.T,  z_sol)
                    K_fb[free] = -y_sol

                k_ff_seq[k] = k_ff
                K_fb_seq[k] = K_fb

                # expected-improvement accumulators — unregularized Quu, Qu
                # (mujoco_mpc/mjpc/planners/ilqg/backward_pass.cc:223-226)
                dV1 += float(k_ff @ Qu)
                dV2 += 0.5 * float(k_ff @ Quu @ k_ff)

                # value-function update — unregularized Quu, Qux
                # Vx  = Qx + Qux^T k + K^T (Qu + Quu k)
                # Vxx = Qxx + Qux^T K + K^T Qux + K^T Quu K
                Vx  = Qx  + Qux.T @ k_ff + K_fb.T @ (Qu + Quu @ k_ff)
                Vxx = Qxx + Qux.T @ K_fb + K_fb.T @ Qux + K_fb.T @ Quu @ K_fb
                Vxx = 0.5 * (Vxx + Vxx.T)

            if success:
                return k_ff_seq, K_fb_seq, dV1, dV2, mu, True

            # QP failed at some knot — bump mu and retry
            mu = min(mu * mu_factor, mu_max)
            if mu >= mu_max:
                break

        return k_ff_seq, K_fb_seq, 0.0, 0.0, mu, False


    # Tassa adaptive mu / rate update via z-ratio (actual / expected) and step size
    def _update_mu(self, mu, mu_rate, z, s):
        """
        Adaptive regularization update mirroring mjpc UpdateRegularization +
        ScaleRegularization (backward_pass.cc:330-356):

          bad   (z or s NaN / non-positive)  -> rate <- factor^2-direction
          good  (z > 0.5 or s > 0.3)         -> rate <- 1/factor-direction
          poor  (z < 0.1 or s < 0.06)        -> rate <- factor-direction
          else                                -> no change

        rate accelerates with same-direction bumps:
            grow:   rate = max(rate * scale, scale)
            shrink: rate = min(rate * scale, scale)
        Effective update: mu <- clip(mu * rate, mu_min, mu_max).
        """
        cfg    = self.config
        factor = cfg.mu_factor
        mu_min = cfg.mu_min
        mu_max = cfg.mu_max

        # thresholds from mjpc (backward_pass.cc:341-356)
        Z_HI, Z_LO = 0.5,  0.1
        S_HI, S_LO = 0.3,  0.06

        z_bad = (not np.isfinite(z)) or z <= 0.0
        s_bad = (not np.isfinite(s)) or s <= 0.0

        if z_bad or s_bad:
            scale = factor * factor
            mu_rate = max(mu_rate * scale, scale)
        elif z > Z_HI or s > S_HI:
            scale = 1.0 / factor
            mu_rate = min(mu_rate * scale, scale)
        elif z < Z_LO or s < S_LO:
            scale = factor
            mu_rate = max(mu_rate * scale, scale)
        else:
            return mu, mu_rate                                  # no change

        mu = min(max(mu * mu_rate, mu_min), mu_max)
        return mu, mu_rate


    # outer iLQR loop with parallel α line search and Tassa z-ratio mu schedule
    def solve(self, x0, U_init):
        """
        Run iLQR to convergence (or max_iter) starting from an initial control guess.

        Args:
            x0:     (nx,) initial state
            U_init: (N, nu) initial control sequence (clipped to [u_lb, u_ub])
        Returns:
            X:      (N+1, nx) final state trajectory
            U:      (N, nu)   final control sequence
            J_hist: list[float]; J_hist[0] is the initial cost
        """
        cfg = self.config
        dyn = self.dyn

        N      = U_init.shape[0]
        nx, nu = dyn.nx, dyn.nu

        max_iter = cfg.max_iter
        tol      = cfg.tol
        mu       = cfg.mu
        mu_max   = cfg.mu_max
        K_ls     = max(1, cfg.num_linesearch_candidates)
        a_min    = cfg.min_linesearch_step

        # log-spaced alphas, last one pinned to 0 so the nominal is always a candidate
        # (mujoco_mpc/mjpc/planners/ilqg/planner.cc:386-388)
        if K_ls == 1:
            alphas = np.array([1.0])
        else:
            alphas = np.concatenate([
                np.geomspace(1.0, a_min, K_ls - 1),
                np.array([0.0]),
            ])

        # initial open-loop rollout of U_init (with clipping)
        X = np.empty((N + 1, nx)); X[0] = x0
        U = np.empty((N, nu))
        xk = x0.copy()
        for k in range(N):
            U[k] = np.clip(U_init[k], dyn.u_lb, dyn.u_ub)
            xk = dyn.f_disc(xk, U[k], clip=True)
            X[k + 1] = xk
        J = self.cost(X, U)
        J_hist = [float(J)]
        verbose = cfg.verbose
        if verbose:
            print(f"[iLQR] iter   0: J={float(J):.4f}")

        mu_rate = 1.0   # same-direction acceleration factor (mjpc)

        it = 0
        while it < max_iter:
            # linearize + backward pass (with inline mu-bump retry on Cholesky failure)
            Ad_seq, Bd_seq = self.linearize_about_trajectory(X, U)
            k_ff, K_fb, dV1, dV2, mu, ok = self.backward_pass(X, U, Ad_seq, Bd_seq, mu)

            if not ok:
                # backward pass exhausted internal retries; escalate via z-ratio path
                mu, mu_rate = self._update_mu(mu, mu_rate, float("nan"), float("nan"))
                if verbose:
                    print(f"[iLQR] iter {it:3d}: backward pass failed, mu -> {mu:.2e}")
                if mu >= mu_max:
                    if verbose:
                        print(f"[iLQR] mu hit mu_max={mu_max:.2e}; stopping.")
                    break
                continue

            # parallel multi-α line search
            X_all, U_all, J_all, _failed = self.forward_pass_batched(
                x0, X, U, k_ff, K_fb, alphas
            )

            winner   = int(np.argmin(J_all))
            a_star   = float(alphas[winner])
            J_try    = float(J_all[winner])
            dJ       = float(J) - J_try
            # quadratic expected improvement: E(α) = -α (dV1 + α dV2)
            # (mujoco_mpc/mjpc/planners/ilqg/planner.cc:562-567)
            exp_red  = -a_star * (dV1 + a_star * dV2)

            # mjpc-style accept: any real improvement (candidate set includes α=0)
            accepted = (dJ > 0.0) and np.isfinite(J_try)

            if accepted:
                it += 1
                X, U, J = X_all[winner], U_all[winner], J_try
                J_hist.append(float(J))

                # z-ratio mu update (Tassa)
                z = (dJ / exp_red) if exp_red > 0.0 else float("nan")
                mu, mu_rate = self._update_mu(mu, mu_rate, z, a_star)

                if verbose:
                    print(f"[iLQR] iter {it:3d}: J={float(J):.4f}  dJ={dJ:.3e}  "
                          f"alpha={a_star:.4f}  z={z:.2f}  "
                          f"E={exp_red:.3e}  mu={mu:.2e}")

                # convergence on actual or expected reduction
                if abs(dJ) < tol or (exp_red > 0.0 and exp_red < tol):
                    if verbose:
                        print(f"[iLQR] converged: |dJ|={abs(dJ):.2e}, "
                              f"E={exp_red:.2e} < tol={tol:.2e}")
                    break
            else:
                # no candidate improved -> force-grow mu via "bad" path
                mu, mu_rate = self._update_mu(mu, mu_rate, float("nan"), float("nan"))
                if verbose:
                    print(f"[iLQR] iter {it:3d}: line search failed "
                          f"(dV1={dV1:.2e}, dV2={dV2:.2e}), mu -> {mu:.2e}")
                if mu >= mu_max:
                    if verbose:
                        print(f"[iLQR] mu hit mu_max={mu_max:.2e}; stopping.")
                    break

        if verbose:
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
