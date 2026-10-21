import numpy as np

from .sampling import cosine_directions


class Floor:
    """The floor z = 0, where the particles stick or bounce.

    Parameters
    ----------
    p_stick : float
        Probability that a particle sticks when it lands. It must be > 0,
        otherwise the simulation never stops.
    p_specular : float
        Probability that a bounce is specular. Otherwise the bounce is diffuse.

    Raises
    ------
    ValueError
        If p_stick is not in (0, 1] or p_specular is not in [0, 1].
    """

    def __init__(self, p_stick, p_specular):
        if p_stick <= 0 or p_stick > 1:
            raise ValueError("p_stick must be in (0, 1]")
        if p_specular < 0 or p_specular > 1:
            raise ValueError("p_specular must be in [0, 1]")
        self.p_stick = p_stick
        self.p_specular = p_specular

    def find_landed(self, z):
        """Find the particles that landed during the step.

        Parameters
        ----------
        z : numpy.ndarray
            Heights after the time step.

        Returns
        -------
        numpy.ndarray of bool
            True for the particles that are under the floor now (z < 0).
        """
        return z < 0

    def landing_point(self, r_old, v_old, r_new, v_new):
        """Find where and how fast the particles hit the floor.

        The particle crossed z = 0 somewhere inside the step, at the fraction
        f = z_old / (z_old - z_new) of the step (0 <= f <= 1). We use linear
        interpolation with the same f for the position and the velocity. Without
        drag the velocity changes linearly in time, so this gives the velocity at
        that moment. (If we take v_new from the end of the step, the particle is
        already under the floor and too fast, so it gets energy at every bounce.
        Our energy check found this.)

        Parameters
        ----------
        r_old : numpy.ndarray, shape (n, 3)
            Positions before the step.
        v_old : numpy.ndarray, shape (n, 3)
            Velocities before the step.
        r_new : numpy.ndarray, shape (n, 3)
            Positions after the step.
        v_new : numpy.ndarray, shape (n, 3)
            Velocities after the step.

        Returns
        -------
        r_land : numpy.ndarray, shape (n, 3)
            Landing points, exactly on the floor.
        v_land : numpy.ndarray, shape (n, 3)
            Velocities when the particles hit the floor.
        """
        z_old = r_old[:, 2]
        z_new = r_new[:, 2]
        f = z_old / (z_old - z_new)
        f = np.clip(f, 0, 1)[:, np.newaxis]
        r_land = r_old + f * (r_new - r_old)
        r_land[:, 2] = 0.0  # put it exactly on the floor
        v_land = v_old + f * (v_new - v_old)
        return r_land, v_land

    def land(self, p, ind, r_land, v_land, rng):
        """Make the landed particles stick or bounce.

        Every landing is independent: a particle sticks if a uniform random number
        is smaller than p_stick. The others bounce, specular with probability
        p_specular and diffuse otherwise. Both bounces are elastic.

        Parameters
        ----------
        p : Particles
            All particles. They are changed here.
        ind : numpy.ndarray
            Indices of the particles that landed in this step.
        r_land : numpy.ndarray, shape (len(ind), 3)
            Their landing points.
        v_land : numpy.ndarray, shape (len(ind), 3)
            Their velocities when they hit the floor.
        rng : numpy.random.Generator
            Random number generator.
        """
        if len(ind) == 0:
            return

        # save the first landing, for the check with R = v^2 sin(2 theta) / g
        first = np.isnan(p.first_land[ind, 0])
        p.first_land[ind[first]] = r_land[first]
        p.r[ind] = r_land
        p.v[ind] = v_land

        # each landing is independent: stick if a uniform number u < p_stick
        stick = rng.random(len(ind)) < self.p_stick
        p.stuck[ind[stick]] = True

        # the others bounce, specular with probability p_specular
        b = ind[~stick]
        p.n_bounce[b] += 1
        spec = rng.random(len(b)) < self.p_specular
        b_spec = b[spec]
        b_diff = b[~spec]

        # specular bounce: v_z changes sign, v_x and v_y stay the same,
        # like a ball on a hard floor. We use abs() so the particle always goes
        # up after the bounce (a very flat particle can land in its first step
        # while its v_z is still a tiny bit positive)
        p.v[b_spec, 2] = np.abs(p.v[b_spec, 2])

        # diffuse bounce: the particle keeps its speed but gets a new random
        # direction from the cosine law (same as at the source). The bounces
        # are elastic, so in both cases the kinetic energy does not change.
        speed = np.linalg.norm(p.v[b_diff], axis=1)
        p.v[b_diff] = speed[:, np.newaxis] * cosine_directions(len(b_diff), rng)
