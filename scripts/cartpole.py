##
#
# Cartpole swing-up via iLQR.
#
# State  x = [cart_pos, pole_angle, cart_vel, pole_vel]   (nx = 4)
# Control u = [cart_force]                                (nu = 1)
# Goal: pole upright (theta = 0), cart at origin, zero velocities.
#
# The stage / terminal cost is quadratic in the lifted output
#     y(x) = [cart_pos, cos(theta), sin(theta), cart_vel, pole_vel]
# which makes the swing-up cost smooth in theta (no 2*pi wrap).
#
##

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# make src/ importable when running this script directly
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "src")))

from dynamics import MJDynamicsConfig
from ilqr     import iLQRBase, iLQRConfig


###############################################################
# CARTPOLE iLQR (concrete subclass)
###############################################################

class CartpoleILQR(iLQRBase):

    def __init__(self, dyn_config, ilqr_config):
        super().__init__(dyn_config, ilqr_config)

        # weights on lifted state y = [p, cos(th), sin(th), pdot, thdot]
        Q_diag = np.array([1.0, 20.0, 20.0, 0.1, 0.1])
        R_diag = np.array([5e-4])
        self.Q  = np.diag(Q_diag)
        self.R  = np.diag(R_diag)
        self.Qf = self.Q * 500.0

        # target: upright cart at origin, at rest. Lifted: y_des = y(x_des).
        self.x_des = np.array([0.0, 0.0, 0.0, 0.0])
        self.y_des = self._lift(self.x_des[None])[0]

    @staticmethod
    def _lift(x):
        """(B, 4) -> (B, 5): [p, cos(th), sin(th), pdot, thdot]."""
        p     = x[..., 0]
        th    = x[..., 1]
        pdot  = x[..., 2]
        thdot = x[..., 3]
        return np.stack((p, np.cos(th), np.sin(th), pdot, thdot), axis=-1)

    @staticmethod
    def _output_jacobian(x):
        """J = dy/dx, shape (B, 5, 4)."""
        B  = x.shape[0]
        th = x[..., 1]
        J  = np.zeros((B, 5, 4), dtype=x.dtype)
        J[:, 0, 0] =  1.0
        J[:, 1, 1] = -np.sin(th)
        J[:, 2, 1] =  np.cos(th)
        J[:, 3, 2] =  1.0
        J[:, 4, 3] =  1.0
        return J

    # ---- stage cost l(x, u) = 0.5 (y - y_des)^T Q (y - y_des) + 0.5 u^T R u ----
    def l(self, x, u):
        e = self._lift(x) - self.y_des
        return (0.5 * np.einsum("bi,ij,bj->b", e, self.Q, e)
              + 0.5 * np.einsum("bi,ij,bj->b", u, self.R, u))

    def l_x(self, x, u):
        e = self._lift(x) - self.y_des
        J = self._output_jacobian(x)
        return np.einsum("bji,jk,bk->bi", J, self.Q, e)

    # Gauss-Newton (PSD): drops the residual*Hessian-of-y term
    def l_xx(self, x, u):
        J = self._output_jacobian(x)
        return np.einsum("bji,jk,bkl->bil", J, self.Q, J)

    def l_u(self, x, u):
        return np.einsum("ij,bj->bi", self.R, u)

    def l_uu(self, x, u):
        B = u.shape[0]
        return np.broadcast_to(self.R, (B, *self.R.shape)).copy()

    def l_ux(self, x, u):
        B = x.shape[0]
        return np.zeros((B, u.shape[-1], x.shape[-1]), dtype=x.dtype)

    # ---- terminal cost lf(x) = 0.5 (y - y_des)^T Qf (y - y_des) ----
    def lf(self, x):
        e = self._lift(x) - self.y_des
        return 0.5 * np.einsum("bi,ij,bj->b", e, self.Qf, e)

    def lf_x(self, x):
        e = self._lift(x) - self.y_des
        J = self._output_jacobian(x)
        return np.einsum("bji,jk,bk->bi", J, self.Qf, e)

    def lf_xx(self, x):
        J = self._output_jacobian(x)
        return np.einsum("bji,jk,bkl->bil", J, self.Qf, J)

    # ---- total trajectory cost ----
    def cost(self, X, U):
        return float(self.l(X[:-1], U).sum() + self.lf(X[-1:]).item())


###############################################################
# MAIN
###############################################################

if __name__ == "__main__":
    repo = os.path.abspath(os.path.join(_here, ".."))

    dyn_cfg = MJDynamicsConfig(
        xml_path = os.path.join(repo, "models", "cartpole", "cartpole.xml"),
        sim_dt   = 0.01,
        u_lb     = np.array([-100.0]),
        u_ub     = np.array([ 100.0]),
    )

    ilqr_cfg = iLQRConfig(
        max_iter         = 200,
        tol              = 1e-2,
        mu               = 1.0,
        mu_min           = 1e-6,
        mu_max           = 1e10,
        mu_factor        = 2.0,
        alpha_init       = 1.0,
        alpha_beta       = 0.5,
        alpha_min        = 1e-4,
        armijo_c         = 1e-4,
        linearize_method = "sampling",
        sampling_K       = 64,
        sampling_eps     = 5e-2,
        sampling_rng     = np.random.default_rng(0),
    )

    ilqr = CartpoleILQR(dyn_cfg, ilqr_cfg)

    # horizon (number of control knots)
    T = 450

    # initial state: cart at origin, pole hanging down (theta = pi), at rest
    x0 = np.array([0.0, np.pi, 0.0, 0.0])

    # initial control guess: sinusoidal energy-pumping pattern (period 0.5s)
    t_init = np.arange(T) * ilqr.dyn.dt
    U_init = 80.0 * np.sin(2.0 * np.pi * 2.0 * t_init)[:, None] \
             * np.ones((1, ilqr.dyn.nu))

    X, U, J_hist = ilqr.solve(x0, U_init)

    # ---- plot ----
    tspan = np.arange(T + 1) * ilqr.dyn.dt
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    axs[0, 0].plot(tspan, X[:, 0], lw=2)
    axs[0, 0].axhline(0.0, ls="--", c="r", alpha=0.6, label="target 0")
    axs[0, 0].set_xlabel("t (s)"); axs[0, 0].set_ylabel("cart pos (m)")
    axs[0, 0].set_title("Cart position"); axs[0, 0].grid(True); axs[0, 0].legend()

    axs[0, 1].plot(tspan, X[:, 1], lw=2)
    axs[0, 1].axhline(0.0, ls="--", c="r", alpha=0.6, label=r"upright ($\theta=0$)")
    axs[0, 1].set_xlabel("t (s)"); axs[0, 1].set_ylabel(r"$\theta$ (rad)")
    axs[0, 1].set_title("Pole angle"); axs[0, 1].grid(True); axs[0, 1].legend()

    axs[1, 0].step(tspan[:-1], U[:, 0], where="post", lw=2)
    axs[1, 0].axhline(ilqr.dyn.u_ub[0], ls="--", c="k", alpha=0.4)
    axs[1, 0].axhline(ilqr.dyn.u_lb[0], ls="--", c="k", alpha=0.4)
    axs[1, 0].set_xlabel("t (s)"); axs[1, 0].set_ylabel("force (N)")
    axs[1, 0].set_title("Control"); axs[1, 0].grid(True)

    axs[1, 1].plot(range(len(J_hist)), J_hist, "-o", lw=2, ms=4)
    axs[1, 1].set_xlabel("iLQR iteration"); axs[1, 1].set_ylabel("Total cost J")
    axs[1, 1].set_title("Cost history"); axs[1, 1].grid(True)
    axs[1, 1].set_yscale("log")
    fig.tight_layout()

    # ---- save ----
    results_dir = os.path.abspath(os.path.join(_here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, "cartpole_ilqr.png"), dpi=150)
    np.savetxt(os.path.join(results_dir, "state.csv"), X,
               delimiter=",",
               header="cart_pos,pole_angle,cart_vel,pole_vel", comments="")
    np.savetxt(os.path.join(results_dir, "time.csv"), tspan,
               delimiter=",", header="t", comments="")
    # record the XML path (relative to repo/models) so playback loads the matching model
    xml_rel = os.path.relpath(dyn_cfg.xml_path, os.path.join(repo, "models"))
    with open(os.path.join(results_dir, "model.txt"), "w") as f:
        f.write(xml_rel)
    print(f"[cartpole] saved plot + state.csv + time.csv + model.txt -> {results_dir}")

    plt.show()
