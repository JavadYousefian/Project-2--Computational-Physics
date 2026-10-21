"""Particles class and the exact solution of the flight.

During the flight every particle follows Newton's second law with gravity
and linear air drag::

    m d^2r/dt^2 = -m g z_hat - gamma dr/dt

We write it as two first order equations, dr/dt = v and
dv/dt = -g z_hat - (gamma / m) v, and solve them with
``scipy.integrate.solve_ivp``. We use units with m = 1 and g = 1.
"""

import gc

import numpy as np
from scipy.integrate import solve_ivp

from .sampling import cosine_directions, maxwell_speeds


def exact_flight(r0, v0, t, g=1.0, gamma=0.0, m=1.0):
    """Exact solution of the equation of motion, without the floor.

    We use it to test ``solve_ivp`` and as a second way to move the particles.
    Without drag (gamma = 0) it is normal projectile motion. With drag we use
    k = gamma / m and the terminal velocity v_t = (0, 0, g / k)::

        v(t) = -v_t + (v0 + v_t) exp(-k t)
        r(t) = r0 - v_t t + (v0 + v_t) (1 - exp(-k t)) / k

    After a long time exp(-k t) -> 0, so v -> -v_t: the particle falls with the
    terminal velocity m g / gamma.

    Parameters
    ----------
    r0 : numpy.ndarray, shape (N, 3)
        Start positions.
    v0 : numpy.ndarray, shape (N, 3)
        Start velocities.
    t : float
        Time of the flight.
    g : float, optional
        Gravity, by default 1.
    gamma : float, optional
        Drag coefficient, by default 0 (no drag).
    m : float, optional
        Mass, by default 1.

    Returns
    -------
    r : numpy.ndarray, shape (N, 3)
        Positions at the time t.
    v : numpy.ndarray, shape (N, 3)
        Velocities at the time t.
    """
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
    """All particles of the simulation, saved in numpy arrays.

    The particles do not interact with each other, so we keep all of them in
    numpy arrays and move them together.

    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Start positions.
    vel : array_like, shape (N, 3)
        Start velocities.
    gamma : float, optional
        Drag coefficient, by default 0 (no air drag).
    g : float, optional
        Gravity in the -z direction, by default 1.
    m : float, optional
        Mass, by default 1.

    Attributes
    ----------
    r : numpy.ndarray, shape (N, 3)
        Positions. For a stuck particle this is the final position.
    v : numpy.ndarray, shape (N, 3)
        Velocities. For a stuck particle this is the velocity when it hit the
        floor the last time.
    v0 : numpy.ndarray, shape (N, 3)
        Start velocities, for the check of the range.
    E0 : numpy.ndarray, shape (N,)
        Start energies, for the energy check.
    n_bounce : numpy.ndarray, shape (N,)
        Number of bounces of every particle.
    stuck : numpy.ndarray, shape (N,)
        True when the particle is stuck on the floor.
    first_land : numpy.ndarray, shape (N, 3)
        Position of the first landing, nan before the particle lands.
    n_eval : int
        Number of times ``solve_ivp`` called the right-hand side.

    Raises
    ------
    ValueError
        If the shapes are wrong, gamma is negative or m is not positive.
    """

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

    @classmethod
    def from_source(cls, n, sigma=1.0, rng=None, gamma=0.0, g=1.0, m=1.0):
        """Make n particles at the origin (the source) with random velocities.

        The speed comes from the Maxwell-Boltzmann distribution (hot source) and
        the direction from the cosine law (more particles go up than sideways).

        Parameters
        ----------
        n : int
            Number of particles.
        sigma : float, optional
            Width of the Maxwell-Boltzmann distribution, by default 1.
        rng : numpy.random.Generator or int or None, optional
            Random number generator or seed.
        gamma : float, optional
            Drag coefficient, by default 0.
        g : float, optional
            Gravity, by default 1.
        m : float, optional
            Mass, by default 1.

        Returns
        -------
        Particles
            The new particles.
        """
        rng = np.random.default_rng(rng)
        speed = maxwell_speeds(n, sigma, rng)
        direction = cosine_directions(n, rng)
        vel = speed[:, np.newaxis] * direction
        return cls(np.zeros((n, 3)), vel, gamma, g, m)

    def __len__(self):
        """Return the number of particles.

        Returns
        -------
        int
            Number of particles.
        """
        return len(self.r)

    def n_flying(self):
        """Return the number of particles that are not stuck yet.

        Returns
        -------
        int
            Number of flying particles.
        """
        return np.sum(~self.stuck)

    def flying(self):
        """Return the indices of the particles that are not stuck yet.

        Returns
        -------
        numpy.ndarray
            Indices of the flying particles.
        """
        return np.where(~self.stuck)[0]

    def energy(self):
        """Return the mechanical energy E = 1/2 m v^2 + m g z of every particle.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Energy of every particle.
        """
        Ek = 0.5 * self.m * np.sum(self.v**2, axis=1)
        Ep = self.m * self.g * self.r[:, 2]
        return Ek + Ep

    def rhs(self, t, y):
        """Right-hand side dy/dt = (v, a) of the equation of motion for ``solve_ivp``.

        The array y has first all positions and then all velocities,
        y = [x0, y0, z0, x1, y1, z1, ..., vx0, vy0, vz0, vx1, ...].

        Parameters
        ----------
        t : float
            Time. It is not used (the force does not depend on time), but
            ``solve_ivp`` needs it.
        y : numpy.ndarray, shape (6 n,)
            Positions and velocities of the flying particles.

        Returns
        -------
        numpy.ndarray, shape (6 n,)
            Velocities and accelerations.
        """
        n = len(y) // 2
        v = y[n:]
        # acceleration from the drag: a = -(gamma / m) v
        a = -self.gamma / self.m * v
        # and from gravity, only in z (every third number is a z component)
        a[2::3] -= self.g
        return np.concatenate((v, a))

    def move(self, dt, method="solve_ivp", rtol=1e-8, atol=1e-10):
        """Move all flying particles one time step.

        The floor is not checked here, the Floor class does that after the step.

        Parameters
        ----------
        dt : float
            Time step.
        method : {"solve_ivp", "exact"}, optional
            Use ``solve_ivp`` (RK45) or the formulas of ``exact_flight``.
        rtol : float, optional
            Relative tolerance of ``solve_ivp``.
        atol : float, optional
            Absolute tolerance of ``solve_ivp``.

        Raises
        ------
        ValueError
            If the method is not "solve_ivp" or "exact".
        RuntimeError
            If ``solve_ivp`` fails.
        """
        ind = self.flying()
        if len(ind) == 0:
            return
        r = self.r[ind]
        v = self.v[ind]

        if method == "solve_ivp":
            # put the positions and velocities of the flying particles in one array
            y0 = np.concatenate((r.ravel(), v.ravel()))
            # With the benchmark we saw that the memory grows to a few GB for
            # N = 100000. The solver object of scipy has a reference cycle, so
            # only the garbage collector deletes it, and sometimes much too
            # late. So we stop the garbage collector during the call and clean
            # up right after it.
            gc.disable()
            try:
                # first_step=dt: solve_ivp tries the whole step at once. Without
                # drag the path is a parabola and RK45 is exact for it, so one
                # step is enough. With drag it makes smaller steps if needed.
                sol = solve_ivp(
                    self.rhs, (0, dt), y0, rtol=rtol, atol=atol, first_step=dt
                )
            finally:
                gc.enable()
            gc.collect(0)
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
