##
#
# Playback an iLQR-optimized trajectory in MuJoCo's passive viewer.
#
# Each producer script (e.g. cartpole.py, cartpole_mpc.py) writes its outputs
# under results/<script_name>/. Pick which subdir to play back with a CLI arg:
#
#   python scripts/playback.py                # most-recently-modified subdir
#   python scripts/playback.py cartpole       # results/cartpole/
#   python scripts/playback.py cartpole_mpc   # results/cartpole_mpc/
#
# Reads from the chosen subdir:
#   state.csv   (N+1, nx=2*nq) rows: [qpos; qvel]
#   time.csv    (N+1,) timestamps
#   model.txt   path of the XML used, relative to repo/models/
#
##

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer


def load_trajectory(results_dir):
    X = np.loadtxt(os.path.join(results_dir, "state.csv"), delimiter=",", skiprows=1)
    t = np.loadtxt(os.path.join(results_dir, "time.csv"),  delimiter=",", skiprows=1)
    return X, t


def _resolve_results_dir(results_root, requested=None):
    """Pick the results subdir to replay.

    - If `requested` is given, use results_root/requested.
    - Else pick the most-recently-modified subdir containing state.csv.
    Errors out with a helpful message if nothing valid is found.
    """
    if requested is not None:
        d = os.path.join(results_root, requested)
        if not os.path.isfile(os.path.join(d, "state.csv")):
            raise SystemExit(
                f"[playback] {d}/state.csv not found.\n"
                f"Available subdirs: {sorted(os.listdir(results_root)) if os.path.isdir(results_root) else '(none)'}"
            )
        return d

    if not os.path.isdir(results_root):
        raise SystemExit(f"[playback] {results_root} does not exist. Run a producer script first.")

    candidates = []
    for name in os.listdir(results_root):
        sub = os.path.join(results_root, name)
        if os.path.isfile(os.path.join(sub, "state.csv")):
            candidates.append((os.path.getmtime(sub), name, sub))

    if not candidates:
        raise SystemExit(
            f"[playback] no results subdir under {results_root} contains state.csv.\n"
            f"Run scripts/cartpole.py or scripts/cartpole_mpc.py first."
        )

    candidates.sort(reverse=True)
    print(f"[playback] no subdir specified; using most recent: '{candidates[0][1]}'")
    return candidates[0][2]


def main():
    here         = os.path.dirname(os.path.abspath(__file__))
    repo         = os.path.abspath(os.path.join(here, ".."))
    results_root = os.path.join(repo, "results")

    requested   = sys.argv[1] if len(sys.argv) > 1 else None
    results_dir = _resolve_results_dir(results_root, requested)

    # load the XML name recorded by the producer (e.g. "cartpole/cartpole.xml")
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
