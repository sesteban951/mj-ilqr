##
#
# Cartpole swing-up via receding-horizon MPC, using iLQR as the inner solver.
#
# Loop:
#   for t = 0, 1, ..., total_steps:
#     solve iLQR for horizon H from current real state, warm-started by U_warm
#     apply U_opt[:replan_every] to the real system (via MJDynamics.f_disc)
#     shift U_warm by replan_every (pad with last control)
#
# The inner iLQR uses the same upgraded solver as scripts/cartpole.py
# (mjpc-style: quadratic dV, z-ratio adaptive mu, value/control regularization,
#  inline Cholesky retry, parallel multi-α line search).
#
##

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# make src/ and scripts/ importable when running this script directly
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "src")))
sys.path.insert(0, _here)

from dynamics import MJDynamicsConfig
from ilqr     import iLQRConfig
from cartpole import CartpoleILQR


###############################################################
# MPC LOOP
###############################################################

def run_mpc(ilqr, x0, total_steps, U_warm, replan_every=1):
    """
    Receding-horizon MPC.

    Args:
        ilqr:         CartpoleILQR (or any iLQRBase subclass) — its config
                      sets the inner horizon, max_iter, etc.
        x0:           (nx,) initial real-world state
        total_steps:  total simulation length (in solver dt units)
        U_warm:       (H, nu) initial warm-start control over the iLQR horizon
        replan_every: how many real steps to apply between replans (1 = always replan)
    Returns:
        X_real:       (total_steps + 1, nx) realized state trajectory
        U_real:       (total_steps, nu)     realized control sequence
        J_real_hist:  list[float]           running real-time cost at each step
        solve_times:  list[float]           wall-clock seconds per replan
    """
    dyn      = ilqr.dyn
    nx, nu   = dyn.nx, dyn.nu
    H        = U_warm.shape[0]

    X_real = np.empty((total_steps + 1, nx)); X_real[0] = x0
    U_real = np.empty((total_steps, nu))

    x_curr      = x0.copy()
    solve_times = []
    J_run       = 0.0
    J_real_hist = [J_run]

    last_J = None

    step = 0
    while step < total_steps:
        # ---- inner iLQR solve over horizon H from x_curr ----
        t0 = time.perf_counter()
        X_opt, U_opt, J_hist = ilqr.solve(x_curr, U_warm)
        solve_times.append(time.perf_counter() - t0)
        last_J = J_hist[-1]

        # ---- apply first `replan_every` controls to the real system ----
        n_apply = min(replan_every, total_steps - step)
        for i in range(n_apply):
            u = np.clip(U_opt[i], dyn.u_lb, dyn.u_ub)
            U_real[step + i] = u
            # per-step running cost (single-knot stage cost)
            J_run += float(ilqr.l(x_curr[None], u[None])[0])
            x_curr = dyn.f_disc(x_curr, u, clip=True)
            X_real[step + 1 + i] = x_curr
            J_real_hist.append(J_run)

        # ---- shift warm-start: drop the applied head, pad tail with last u ----
        U_warm = np.concatenate([
            U_opt[n_apply:],
            np.tile(U_opt[-1:], (n_apply, 1)),
        ], axis=0)

        # log a heartbeat every 10 replans
        if (step // replan_every) % 10 == 0:
            theta = float(x_curr[1])
            theta_err = np.arctan2(np.sin(theta), np.cos(theta))   # wrap to [-π, π]
            print(f"[MPC] t={step*dyn.dt:5.2f}s  pos={x_curr[0]:+.3f}  "
                  f"θ_err={theta_err:+.3f}rad  thdot={x_curr[3]:+.3f}  "
                  f"J_inner={last_J:.2f}  solve={solve_times[-1]*1000:.0f}ms")

        step += n_apply

    # finalize: add terminal cost
    J_run += float(ilqr.lf(x_curr[None])[0])
    J_real_hist[-1] = J_run

    return X_real, U_real, J_real_hist, solve_times


###############################################################
# MAIN
###############################################################

if __name__ == "__main__":
    repo = os.path.abspath(os.path.join(_here, ".."))

    dyn_cfg = MJDynamicsConfig(
        xml_path = os.path.join(repo, "models", "cartpole", "cartpole.xml"),
        # xml_path = os.path.join(repo, "models", "cartpole", "cartpole_walls.xml"),
        # xml_path = os.path.join(repo, "models", "cartpole", "cartpole_walls_soft.xml"),
        sim_dt   = 0.01,
        u_lb     = np.array([-100.0]),
        u_ub     = np.array([ 100.0]),
    )

    # iLQR config for the inner solve. Tuned for *short* MPC inner loops:
    # - max_iter small (warmstart usually converges in 1-5 iters)
    # - tol relaxed (we don't need pixel-perfect convergence each replan)
    ilqr_cfg = iLQRConfig(
        max_iter                  = 5,
        tol                       = 1e-1,
        mu                        = 1.0,
        mu_min                    = 1e-6,
        mu_max                    = 1e10,
        mu_factor                 = 2.0,
        regularization_type       = "control",
        max_bp_iter               = 3,
        num_linesearch_candidates = 8,
        min_linesearch_step       = 1e-3,
        linearize_method          = "sampling",
        sampling_K                = 128,
        sampling_eps              = 5e-2,
        sampling_rng              = np.random.default_rng(0),
        verbose                   = False,   # quiet inner loop
    )

    ilqr = CartpoleILQR(dyn_cfg, ilqr_cfg)

    # MPC horizon (knots) — long enough that the swing-up plan is feasible inside it
    H = 250

    # Total real simulation length (knots). Pole has H*dt = 2.5s to plan;
    total_steps  = 500
    replan_every = 1

    # initial state: cart at origin, pole hanging down at rest
    x0 = np.array([0.0, np.pi, 0.0, 0.0])

    # initial warm-start: same sinusoidal energy-pumping as offline
    t_init = np.arange(H) * ilqr.dyn.dt
    U_warm = 80.0 * np.sin(2.0 * np.pi * 2.0 * t_init)[:, None] \
             * np.ones((1, ilqr.dyn.nu))

    print(f"[MPC] H={H} knots ({H*ilqr.dyn.dt:.2f}s)  "
          f"total={total_steps} knots ({total_steps*ilqr.dyn.dt:.2f}s)  "
          f"replan_every={replan_every}")
    print(f"[MPC] starting MPC...\n")

    t_total0 = time.perf_counter()
    X, U, J_hist, solve_times = run_mpc(ilqr, x0, total_steps, U_warm,
                                        replan_every=replan_every)
    t_total = time.perf_counter() - t_total0

    print()
    print(f"[MPC] total wall time: {t_total:.2f}s  "
          f"avg solve: {np.mean(solve_times)*1000:.1f}ms  "
          f"max solve: {np.max(solve_times)*1000:.0f}ms")
    print(f"[MPC] final state: pos={X[-1,0]:+.4f}  "
          f"θ={np.arctan2(np.sin(X[-1,1]), np.cos(X[-1,1])):+.4f}rad  "
          f"pdot={X[-1,2]:+.4f}  thdot={X[-1,3]:+.4f}")
    print(f"[MPC] total realized cost: J={J_hist[-1]:.2f}")

    # ---- plot ----
    tspan = np.arange(total_steps + 1) * ilqr.dyn.dt
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    axs[0, 0].plot(tspan, X[:, 0], lw=2)
    axs[0, 0].axhline(0.0, ls="--", c="r", alpha=0.6, label="target 0")
    axs[0, 0].set_xlabel("t (s)"); axs[0, 0].set_ylabel("cart pos (m)")
    axs[0, 0].set_title("Cart position (MPC)"); axs[0, 0].grid(True); axs[0, 0].legend()

    theta_wrapped = np.arctan2(np.sin(X[:, 1]), np.cos(X[:, 1]))
    axs[0, 1].plot(tspan, theta_wrapped, lw=2)
    axs[0, 1].axhline(0.0, ls="--", c="r", alpha=0.6, label=r"upright ($\theta=0$)")
    axs[0, 1].set_xlabel("t (s)"); axs[0, 1].set_ylabel(r"$\theta$ wrapped (rad)")
    axs[0, 1].set_title("Pole angle (MPC)"); axs[0, 1].grid(True); axs[0, 1].legend()

    axs[1, 0].step(tspan[:-1], U[:, 0], where="post", lw=2)
    axs[1, 0].axhline(ilqr.dyn.u_ub[0], ls="--", c="k", alpha=0.4)
    axs[1, 0].axhline(ilqr.dyn.u_lb[0], ls="--", c="k", alpha=0.4)
    axs[1, 0].set_xlabel("t (s)"); axs[1, 0].set_ylabel("force (N)")
    axs[1, 0].set_title("Control (MPC)"); axs[1, 0].grid(True)

    # solve-time histogram — useful for tuning toward real-time
    axs[1, 1].hist(np.array(solve_times) * 1000.0, bins=30, edgecolor="k")
    axs[1, 1].set_xlabel("inner solve time (ms)")
    axs[1, 1].set_ylabel("count")
    axs[1, 1].set_title(f"Inner solve times  (avg {np.mean(solve_times)*1000:.1f}ms)")
    axs[1, 1].grid(True)
    fig.tight_layout()

    # ---- save (per-script subdir: results/<script_name>/) ----
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    results_dir = os.path.abspath(os.path.join(_here, "..", "results", script_name))
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, "plot.png"), dpi=150)
    np.savetxt(os.path.join(results_dir, "state.csv"), X,
               delimiter=",",
               header="cart_pos,pole_angle,cart_vel,pole_vel", comments="")
    np.savetxt(os.path.join(results_dir, "time.csv"), tspan,
               delimiter=",", header="t", comments="")
    xml_rel = os.path.relpath(dyn_cfg.xml_path, os.path.join(repo, "models"))
    with open(os.path.join(results_dir, "model.txt"), "w") as f:
        f.write(xml_rel)
    print(f"[MPC] saved plot + state.csv + time.csv + model.txt -> {results_dir}")

    plt.show()
