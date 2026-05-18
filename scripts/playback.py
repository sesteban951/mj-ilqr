##
#
# Playback an iLQR-optimized trajectory in MuJoCo's passive viewer.
#
# Reads:
#   results/state.csv   (N+1, nx=2*nq) rows: [qpos; qvel]
#   results/time.csv    (N+1,) timestamps
#   results/model.txt   path of the XML used, relative to repo/models/
#
##

import os
import time
import numpy as np
import mujoco
import mujoco.viewer


def load_trajectory(results_dir):
    X = np.loadtxt(os.path.join(results_dir, "state.csv"), delimiter=",", skiprows=1)
    t = np.loadtxt(os.path.join(results_dir, "time.csv"),  delimiter=",", skiprows=1)
    return X, t


def main():
    here        = os.path.dirname(os.path.abspath(__file__))
    repo        = os.path.abspath(os.path.join(here, ".."))
    results_dir = os.path.join(repo, "results")

    # load the XML name recorded by cartpole.py (e.g. "cartpole/cartpole.xml")
    with open(os.path.join(results_dir, "model.txt"), "r") as f:
        xml_rel = f.read().strip()
    xml_path = os.path.join(repo, "models", xml_rel)

    X, t = load_trajectory(results_dir)
    N    = len(X)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    nq = model.nq
    assert X.shape[1] == 2 * nq, (
        f"expected state width {2*nq} (=[qpos; qvel]); got {X.shape[1]}"
    )

    # wall-clock-driven playback at ~50 Hz, robust to any (uniform or non-uniform)
    # solver dt: at each display tick we look up the closest sample to the current
    # simulated time.
    target_fps   = 50.0
    frame_period = 1.0 / target_fps
    sim_t0       = float(t[0])
    sim_total    = float(t[-1] - t[0])

    with mujoco.viewer.launch_passive(model, data,
                                      show_left_ui=False,
                                      show_right_ui=False) as viewer:
        while viewer.is_running():
            wall0 = time.perf_counter()
            while viewer.is_running():
                wall_t = time.perf_counter() - wall0
                if wall_t > sim_total:
                    break                                         # finished this playback pass
                sim_t = sim_t0 + wall_t
                k     = min(int(np.searchsorted(t, sim_t)), N - 1)

                # set state from CSV row and refresh derived quantities
                data.qpos[:] = X[k, :nq]
                data.qvel[:] = X[k, nq:]
                mujoco.mj_forward(model, data)

                # top-left "time = X.XX s" overlay (built-in mjr_overlay text).
                # NB: this `font` arg is mjtFont (NORMAL / SHADOW / BIG), not mjtFontScale;
                # the overall scale is fixed by the MjrContext the viewer created internally.
                viewer.set_texts((mujoco.mjtFont.mjFONT_BIG,
                                  mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  f"time = {float(t[k]):.2f} s", None))

                viewer.sync()

                # pace to ~target_fps in wall-clock time
                target_wall = wall0 + wall_t + frame_period
                sleep_for   = target_wall - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)


if __name__ == "__main__":
    main()
