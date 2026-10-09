# This is for particles class. Has the positions and velocities, moves them with solve_ivp."""

import numpy as np
from scipy.integrate import solve_ivp


class Particles:

    # Class for all the particles that are getting generated.
    """ Parameters
    ----------
    We have pos : array (N, 3)
        start positions
    vel : array (N, 3)
        start velocities
    gamma : float
        drag coefficient
    g : float
        gravity
    m : float
        mass
    """

    def __init__(self, pos, vel, gamma=0.0, g=1.0, m=1.0):

        self.r = np.array(pos, dtype=float)
        self.v = np.array(vel, dtype=float)

        if self.r.ndim != 2 or self.r.shape[1] != 3 or self.v.shape != self.r.shape:
            raise ValueError("pos and vel must have shape (N, 3)")
        if gamma < 0 or m <= 0:
            raise ValueError("wrong gamma or m")
        
        self.gamma = gamma
        self.g = g
        self.m = m

        N = len(self.r)
        self.v0 = self.v.copy()
        self.E0 = self.energy()
        self.n_bounce = np.zeros(N, dtype=int)
        self.stuck = np.zeros(N, dtype=bool)
        self.first_land = np.full((N, 3), np.nan)
        self.n_eval = 0  # number of rhs calls in solve_ivp

    def __len__(self):
        return len(self.r)

    def n_flying(self):
        # Number of particles that are not stuck."""
        return np.sum(~self.stuck)

    def flying(self):
        # Indices of the particles that are not stuck."""
        return np.where(~self.stuck)[0]

    def energy(self):
        # Kinetic + potential energy for every particle."""
        Ek = 0.5 * self.m * np.sum(self.v**2, axis=1)
        Ep = self.m * self.g * self.r[:, 2]
        return Ek + Ep

    def rhs(self, t, y):
        # dy/dt for solve_ivp. y = [all positions, all velocities]."""
        n = len(y) // 2
        v = y[n:]
        a = -self.gamma / self.m * v
        a[2::3] -= self.g  # z components
        return np.concatenate((v, a))

    def move(self, dt, rtol=1e-8, atol=1e-10):
        """Move the flying particles one step dt with solve_ivp.

        Parameters
        ----------
        dt : float
            time step
        rtol, atol : float
            tolerances of solve_ivp
        """
        ind = self.flying()
        if len(ind) == 0:
            return
        y0 = np.concatenate((self.r[ind].ravel(), self.v[ind].ravel()))
        sol = solve_ivp(self.rhs, (0, dt), y0, rtol=rtol, atol=atol, first_step=dt)
        if not sol.success:
            raise RuntimeError(sol.message)
        self.n_eval += sol.nfev

        y = sol.y[:, -1]
        n = len(y) // 2
        self.r[ind] = y[:n].reshape(-1, 3)
        self.v[ind] = y[n:].reshape(-1, 3)