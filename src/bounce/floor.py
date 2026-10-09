# Floor class. The floor is the plane z = 0. After every time step it finds
# the particles that went below the floor, puts them back on the floor, and
# then each landed particle sticks or bounces.

import numpy as np


class Floor:
    # p_stick : probability that a particle sticks when it lands.
    # It must be > 0, otherwise the simulation never stops.
    # For now every bounce is specular (like a ball on a hard floor).

    def __init__(self, p_stick):
        if p_stick <= 0 or p_stick > 1:
            raise ValueError("p_stick must be in (0, 1]")
        self.p_stick = p_stick

    def find_landed(self, z):
        # a particle landed during the step if it is below the floor now
        return z < 0

    def landing_point(self, r_old, r_new):
        # The particle crossed z = 0 somewhere inside the step. We find the
        # point with linear interpolation between the old and the new position:
        # z goes from z_old to z_new, so z = 0 at the fraction
        # f = z_old / (z_old - z_new) of the step (0 <= f <= 1).
        z_old = r_old[:, 2]
        z_new = r_new[:, 2]
        f = z_old / (z_old - z_new)
        f = np.clip(f, 0, 1)[:, np.newaxis]
        r_land = r_old + f * (r_new - r_old)
        r_land[:, 2] = 0.0  # put it exactly on the floor
        return r_land

    def land(self, p, ind, r_land, rng):
        # p : the Particles object (we change it here)
        # ind: indices of the particles that landed in this step
        # r_land: their landing points
        # rng: random number generator
        if len(ind) == 0:
            return

        # save the first landing, for the check with R = v^2 sin(2 theta) / g
        first = np.isnan(p.first_land[ind, 0])
        p.first_land[ind[first]] = r_land[first]
        p.r[ind] = r_land

        # each landing is independent: stick if a uniform number u < p_stick
        stick = rng.random(len(ind)) < self.p_stick
        p.stuck[ind[stick]] = True

        # the others bounce. Specular bounce: v_z changes sign, v_x and v_y
        # stay the same, so the speed (and the energy) does not change.
        b = ind[~stick]
        p.n_bounce[b] += 1
        p.v[b, 2] = -p.v[b, 2]