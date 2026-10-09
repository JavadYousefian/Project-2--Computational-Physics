# Particles class. It keeps the positions and velocities of all particles
# and moves them in time with solve_ivp.

# Equation of motion during the flight (gravity + linear air drag):
#  m d^2r/dt^2 = -m g z_hat - gamma dr/dt
# We write it as two first order equations:
# dr/dt = v
# dv/dt = -g z_hat - (gamma / m) v
# We use units with m = 1 and g = 1.

import numpy as np
from scipy.integrate import solve_ivp


def exact_flight(r0, v0, t, g=1.0, gamma=0.0, m=1.0):
    # Exact solution of the equation of motion (the floor is not used here).
    # We use it to test solve_ivp, and as a second way to move the particles.
    
    # Without drag (gamma = 0) it is normal projectile motion:
    # r(t) = r0 + v0 t - 1/2 g t^2 z_hat
    # v(t) = v0 - g t z_hat
    # With drag we use k = gamma / m and the terminal velocity v_t = (0, 0, g / k):
    # v(t) = -v_t + (v0 + v_t) exp(-k t)
    # r(t) = r0 - v_t t + (v0 + v_t) (1 - exp(-k t)) / k
    # After a long time exp(-k t) -> 0, so v -> -v_t: the particle falls with
    # the terminal velocity m g / gamma.
    g_vec = np.array([0.0, 0.0, g])
    if gamma == 0:
        r = r0 + v0 * t - 0.5 * g_vec * t**2
        v = v0 - g_vec * t
        return r, v

    k = gamma / m
    vt = g_vec / k
    v = -vt + (v0 + vt) * np.exp(-k * t)
    # 1 - exp(-k t) = -expm1(-k t), expm1 is more precise when k t is small
    r = r0 - vt * t - (v0 + vt) * np.expm1(-k * t) / k
    return r, v


class Particles:
    # All particles of the simulation. The particles do not interact with each
    # other, so we keep all of them in numpy arrays and move them together.
    
    # pos: array (N, 3), start positions
    # vel: array (N, 3), start velocities
    # gamma: drag coefficient (gamma = 0 means no air drag)
    # g: gravity, in the -z direction
    # m: mass

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

    def move(self, dt, method="solve_ivp", rtol=1e-8, atol=1e-10):
        # move all flying particles one time step dt
        # method = "solve_ivp" (RK45) or "exact" (the formulas in exact_flight)
        # rtol, atol are the tolerances of solve_ivp
        # the floor is not checked here, the Floor class does that after the step
        ind = self.flying()
        if len(ind) == 0:
            return
        r = self.r[ind]
        v = self.v[ind]

        if method == "solve_ivp":
            # put the positions and velocities of the flying particles in one array
            y0 = np.concatenate((r.ravel(), v.ravel()))
            # first_step=dt: solve_ivp tries the whole step at once. Without drag
            # the path is a parabola and RK45 is exact for it, so one step is
            # enough. With drag it makes smaller steps if the error is too big.
            sol = solve_ivp(self.rhs, (0, dt), y0, rtol=rtol, atol=atol, first_step=dt)
            if not sol.success:
                raise RuntimeError(sol.message)
            self.n_eval += sol.nfev
            # values at the end of the step
            y = sol.y[:, -1]
            n = len(y) // 2
            r_new = y[:n].reshape(-1, 3)
            v_new = y[n:].reshape(-1, 3)
        elif method == "exact":
            r_new, v_new = exact_flight(r, v, dt, self.g, self.gamma, self.m)
        else:
            raise ValueError("method must be solve_ivp or exact")

        self.r[ind] = r_new
        self.v[ind] = v_new