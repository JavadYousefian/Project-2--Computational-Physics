# Particles class. It keeps the positions and velocities of all particles
# and moves them in time with solve_ivp.

# Equation of motion during the flight (gravity + linear air drag):
# m d^2r/dt^2 = -m g z_hat - gamma dr/dt
# We write it as two first order equations:
# dr/dt = v
# dv/dt = -g z_hat - (gamma / m) v
# We use units with m = 1 and g = 1.

import numpy as np
from scipy.integrate import solve_ivp


class Particles:
    # All particles of the simulation. The particles do not interact with each
    # other, so we keep all of them in numpy arrays and move them together.
    
    # pos : array (N, 3), start positions
    # vel : array (N, 3), start velocities
    # gamma : drag coefficient (gamma = 0 means no air drag)
    # g : gravity, in the -z direction
    # m : mass

    def __init__(self, pos, vel, gamma=0.0, g=1.0, m=1.0):
        self.r = np.array(pos, dtype=float)
        self.v = np.array(vel, dtype=float)

        if self.r.ndim != 2 or self.r.shape[1] != 3 or self.v.shape != self.r.shape:
            raise ValueError("pos and vel must have shape (N, 3)")
        # negative drag would give energy to the particle, and the mass must be > 0
        if gamma < 0 or m <= 0:
            raise ValueError("wrong gamma or m")

        self.gamma = gamma
        self.g = g
        self.m = m

        N = len(self.r)
        # start velocities, for the check of the range R = v^2 sin(2 theta) / g
        self.v0 = self.v.copy()
        # start energies, without drag the energy must stay the same
        self.E0 = self.energy()
        # number of bounces k, should follow P(k) = (1 - p_stick)^k p_stick
        self.n_bounce = np.zeros(N, dtype=int)
        # True when the particle is stuck on the floor, then it does not move
        self.stuck = np.zeros(N, dtype=bool)
        # position of the first landing, nan until the particle lands
        self.first_land = np.full((N, 3), np.nan)
        # number of times solve_ivp called rhs (for the benchmark)
        self.n_eval = 0

    def __len__(self):
        return len(self.r)

    def n_flying(self):
        # number of particles that are not stuck yet
        return np.sum(~self.stuck)

    def flying(self):
        # indices of the particles that are not stuck yet
        return np.where(~self.stuck)[0]

    def energy(self):
        # mechanical energy E = 1/2 m v^2 + m g z for every particle
        Ek = 0.5 * self.m * np.sum(self.v**2, axis=1)
        Ep = self.m * self.g * self.r[:, 2]
        return Ek + Ep

    def rhs(self, t, y):
        # right hand side for solve_ivp: dy/dt = (v, a)
        # y has first all positions and then all velocities:
        # y = [x0, y0, z0, x1, y1, z1, ..., vx0, vy0, vz0, vx1, ...]
        # t is not used (the force does not depend on time), but solve_ivp needs it
        n = len(y) // 2
        v = y[n:]
        # acceleration from the drag: a = -(gamma / m) v
        a = -self.gamma / self.m * v
        # and from gravity, only in z (every third number is a z component)
        a[2::3] -= self.g
        return np.concatenate((v, a))

    def move(self, dt, rtol=1e-8, atol=1e-10):
        # move all flying particles one time step dt with solve_ivp (RK45)
        # rtol, atol are the tolerances of solve_ivp
        # the floor is not checked here, the Floor class does that after the step
        ind = self.flying()
        if len(ind) == 0:
            return

        # put the positions and velocities of the flying particles in one array
        y0 = np.concatenate((self.r[ind].ravel(), self.v[ind].ravel()))
        # first_step=dt: solve_ivp tries the whole step at once. Without drag the
        # path is a parabola and RK45 is exact for it, so one step is enough.
        # With drag it makes smaller steps if the error is too big.
        sol = solve_ivp(self.rhs, (0, dt), y0, rtol=rtol, atol=atol, first_step=dt)
        if not sol.success:
            raise RuntimeError(sol.message)
        self.n_eval += sol.nfev

        # values at the end of the step, put them back in r and v
        y = sol.y[:, -1]
        n = len(y) // 2
        self.r[ind] = y[:n].reshape(-1, 3)
        self.v[ind] = y[n:].reshape(-1, 3)