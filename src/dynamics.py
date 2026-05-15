##
#
# MuJoCo dynamics wrapper.
#
##

# abstract classes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

# for dynamics
import mujoco
from mujoco import rollout
import numpy as np

###############################################################
# CONFIG and BASE CLASS
###############################################################

@dataclass
class MJDynamicsConfig:

    # path to MuJoCo XML model
    xml_path: str

    # simulation timestep
    sim_dt: float

    # input bounds
    u_lb: np.ndarray
    u_ub: np.ndarray


class MJDynamicsBase(ABC):

    def __init__(self, config):
        # copy config to self
        self.config = config

        # load model and create data
        self.init_model()

    # load the MuJoCo model and set up dimensions/buffers
    @abstractmethod
    def init_model(self):
        pass

    # return current state x = [qpos; qvel] as (nx,)
    @abstractmethod
    def get_state(self):
        pass

    # set state x = [qpos; qvel] and refresh derived quantities
    @abstractmethod
    def set_state(self, x):
        pass

    # estimate (Ad, Bd) of the discrete map f_disc at (x, u)
    @abstractmethod
    def linearize(self, x, u, ilqr_params) -> tuple[np.ndarray, np.ndarray]:
        pass


###############################################################
# MUJOCO CPU
###############################################################

class MJDynamics_CPU(MJDynamicsBase):

    def __init__(self, config):
        super().__init__(config)

    # initalize mujoco model
    def init_model(self):
        # load model and create data
        self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.data  = mujoco.MjData(self.model)
        print("Model loaded from:", self.config.xml_path)

        # override timestep
        self.model.opt.timestep = self.config.sim_dt

        # dimensions
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.nx = self.nq + self.nv
        self.dt = round(float(self.model.opt.timestep), 6)

        # control box (enforced here, not by MuJoCo)
        self.u_lb = np.asarray(self.config.u_lb, dtype=np.float64)
        self.u_ub = np.asarray(self.config.u_ub, dtype=np.float64)
        assert self.u_lb.shape == (self.nu,) and self.u_ub.shape == (self.nu,), (
            f"u_lb/u_ub must have shape ({self.nu},); "
            f"got {self.u_lb.shape} and {self.u_ub.shape}"
        )

        # batched rollout pool for sampling-based linearization
        # mjSTATE_FULLPHYSICS layout is [time, qpos, qvel, act, ...]; we only
        # perturb qpos/qvel, so cache their slices once
        self._full_state_spec = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._nstate          = mujoco.mj_stateSize(self.model, self._full_state_spec)
        self._qpos_slice      = slice(1,            1 + self.nq)
        self._qvel_slice      = slice(1 + self.nq,  1 + self.nq + self.nv)

        self._nthread        = max(1, os.cpu_count() or 1)
        self._rollout        = rollout.Rollout(nthread=self._nthread)
        self._rollout_data   = [mujoco.MjData(self.model) for _ in range(self._nthread)]

    # convienience functions to get current state
    def get_state(self):
        return np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))

    # convienience function to set state and refresh derived quantities
    def set_state(self, x):
        self.data.qpos[:] = x[:self.nq]
        self.data.qvel[:] = x[self.nq:]
        mujoco.mj_forward(self.model, self.data)

    # discrete step function
    def f_disc(self, x, u, clip=False):
        """
        One mj_step from (x, u).

        Args:
            x:    (nx,) state [qpos; qvel]
            u:    (nu,) control
            clip: if True, saturate u to [u_lb, u_ub] before stepping; if False,
                  pass u through unmodified (MuJoCo no longer clamps either, so
                  this lets callers sample raw, un-clipped dynamics).
        Returns:
            x_next: (nx,) state at t + dt
        """
        # saturate control to config bounds (optional)
        if clip:
            u_eff = np.clip(u, self.u_lb, self.u_ub)
        else:
            u_eff = np.asarray(u, dtype=np.float64)

        # set state and input
        self.set_state(x)
        self.data.ctrl[:] = u_eff

        # step forward
        mujoco.mj_step(self.model, self.data)

        return self.get_state()

    # dispatch to one of the two linearization methods below
    def linearize(self, x, u, ilqr_params):
        """
        Estimate (Ad, Bd) of the discrete map f_disc at (x, u).

        Dispatches on ilqr_params.linearize_method (iLQRConfig field):
            "mujoco_fd" -> linearize_mujoco_fd       (uses fd_eps, fd_centered)
            "sampling"  -> linearize_sampling_based  (uses sampling_K, sampling_eps,
                                                      sampling_reg, sampling_rng)
        """
        method = ilqr_params.linearize_method
        if method == "mujoco_fd":
            return self.linearize_mujoco_fd(x, u, ilqr_params)
        if method == "sampling":
            return self.linearize_sampling_based(x, u, ilqr_params)
        raise ValueError(
            f"unknown linearize_method '{method}'; "
            f"expected 'mujoco_fd' or 'sampling'"
        )

    # MuJoCo built-in finite-difference linearization of the discrete map
    def linearize_mujoco_fd(self, x, u, ilqr_params):
        """
        Estimate (Ad, Bd) for the discrete-time linearization of f_disc at (x, u)
        using mujoco.mjd_transitionFD.

        Args:
            x:           (nx,) linearization state
            u:           (nu,) linearization control
            ilqr_params: iLQRConfig; reads
                           .fd_eps      finite-difference step
                           .fd_centered centered differences (bool)
        Returns:
            Ad: (nx, nx)
            Bd: (nx, nu)
        """
        eps      = ilqr_params.fd_eps
        centered = ilqr_params.fd_centered

        # set state + control on data and refresh
        self.set_state(x)
        self.data.ctrl[:] = np.asarray(u, dtype=np.float64)

        Ad = np.zeros((self.nx, self.nx), dtype=np.float64)
        Bd = np.zeros((self.nx, self.nu), dtype=np.float64)
        mujoco.mjd_transitionFD(self.model, self.data, eps, centered, Ad, Bd, None, None)
        return Ad, Bd

    # sampling-based linearization of the discrete map
    def linearize_sampling_based(self, x, u, ilqr_params):
        """
        Estimate (Ad, Bd) for the discrete-time linearization of f_disc at (x, u):
            f_disc(x + dx, u + du) - f_disc(x, u) ≈ Ad dx + Bd du.

        Gaussian smoothing on z = [xi; eta] ~ N(0, I_{nx+nu}):
            y_k = [f_disc(x + eps*xi_k, u + eps*eta_k)
                 - f_disc(x - eps*xi_k, u - eps*eta_k)] / (2*eps)
                ≈ Ad xi_k + Bd eta_k = [Ad | Bd] z_k.
        Monte Carlo least squares:
            [Ad_hat | Bd_hat] = (Y^T Z / K)(Z^T Z / K + reg I)^{-1}.

        Args:
            x:           (nx,) linearization state
            u:           (nu,) linearization control
            ilqr_params: iLQRConfig; reads
                           .sampling_K   number of paired samples
                           .sampling_eps perturbation scale
                           .sampling_reg ridge on the gram matrix
                           .sampling_rng np.random.Generator
        Returns:
            Ad: (nx, nx)
            Bd: (nx, nu)
        """
        nx, nu, nq = self.nx, self.nu, self.nq

        # sampling knobs from ilqr_params
        K   = ilqr_params.sampling_K
        eps = ilqr_params.sampling_eps
        reg = ilqr_params.sampling_reg
        rng = ilqr_params.sampling_rng

        # joint perturbation z = [xi; eta] ~ N(0, I_{nx+nu})
        xi  = rng.standard_normal((K, nx))
        eta = rng.standard_normal((K, nu))

        # reference FULLPHYSICS state at the nominal (x, u); tiled into 2K rows
        # for paired +/- central-difference rollouts
        self.set_state(x)
        ref_state = np.zeros(self._nstate, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, ref_state, self._full_state_spec)
        initial_state = np.tile(ref_state, (2 * K, 1))                     # (2K, nstate)

        qs, vs = self._qpos_slice, self._qvel_slice
        initial_state[:K, qs] += eps * xi[:, :nq]
        initial_state[:K, vs] += eps * xi[:, nq:]
        initial_state[K:, qs] -= eps * xi[:, :nq]
        initial_state[K:, vs] -= eps * xi[:, nq:]

        # one mj_step per rollout: control shape (nroll, nstep, nu)
        u_arr   = np.asarray(u, dtype=np.float64)
        control = np.empty((2 * K, 1, nu), dtype=np.float64)
        control[:K, 0, :] = u_arr + eps * eta
        control[K:, 0, :] = u_arr - eps * eta

        # threaded batched rollouts (single step each)
        state, _ = self._rollout.rollout(
            self.model, self._rollout_data, initial_state, control, nstep=1
        )                                                                  # (2K, 1, nstate)

        # next [qpos; qvel] for each sample
        final  = state[:, 0, :]                                            # (2K, nstate)
        next_x = np.concatenate([final[:, qs], final[:, vs]], axis=1)      # (2K, nx)

        # central-difference directional derivative y_k ≈ [Ad | Bd] z_k
        Y = (next_x[:K] - next_x[K:]) / (2.0 * eps)                        # (K, nx)

        # stacked perturbation Z
        Z = np.concatenate((xi, eta), axis=1)             # (K, nx+nu)

        # least-squares estimate: [Ad | Bd] = (Y^T Z)(Z^T Z + reg I)^{-1}
        YTZ = Y.T @ Z / K                                 # (nx, nx+nu)
        ZTZ = Z.T @ Z / K                                 # (nx+nu, nx+nu)
        G = ZTZ + reg * np.eye(nx + nu)
        AB = np.linalg.solve(G.T, YTZ.T).T                # (nx, nx+nu)

        Ad = AB[:, :nx]
        Bd = AB[:, nx:]
        return Ad, Bd
