import numpy as np


class Euler:
    def __init__(self):
        pass

    def step(self, ode, t, dt, u_0):
        u_1 = u_0 + dt * ode.rhs(t, u_0)
        return u_1


class Cromer:
    def __init__(self, L):
        self.L = L

    def step(self, ode, t, dt, u_0):
        u_x = u_0 + ode.rhs(t, u_0) * dt
        u_new = np.zeros_like(u_0)

        u_new[1::2] = u_x[1::2]
        u_new[::2] = u_0[::2] % self.L

        u_1 = u_0 + ode.rhs(t, u_new) * dt
        u_1[1::2] = u_new[1::2]

        return u_1


class Integrator:
    def __init__(self, ode, method):
        self.ode = ode
        self.method = method

    def integrate(self, interval, dt, u_0):
        t_0 = interval[0]
        t_end = interval[1]

        times = [t_0]
        states = [u_0]

        t = t_0
        while t < t_end:
            dt_ = min(dt, t_end - t)
            u_1 = self.method.step(self.ode, t, dt_, u_0)
            t = t + dt_
            u_0 = u_1

            times.append(t)
            states.append(u_1)

        return np.array(times), np.array(states)


class CromerPeriodic:
    def __init__(self, callbacks=[]):
        self.callbacks = callbacks

    def step(self, ode, t, dt, u_0):
        u_star = u_0 + dt * ode.rhs(t, u_0)
        for c in self.callbacks:
            u_star = c.apply(u_star)
        u_1 = u_0 + dt * ode.rhs(t, u_star)
        for c in self.callbacks:
            u_1 = c.apply(u_1)
        u_final = np.zeros_like(u_0)
        u_final[: ode.n_bodies * 2] = u_1[: ode.n_bodies * 2]
        u_final[ode.n_bodies * 2 :] = u_star[ode.n_bodies * 2 :]
        return u_final


class PBCCallback:
    def __init__(self, position_indices, L):
        """Accepts a list of which degrees of freedom in u are positions
        (which varies depending on how you organized them) as well as a
        maximum domain size."""
        self.position_indices = position_indices
        self.L = L

    def apply(self, u):
        # Set the positions to position modulo L
        u[self.position_indices] = u[self.position_indices] % self.L
        return u


class LennardJones:
    """This is an example class for an ODE specification"""

    def __init__(
        self, eps=1.0, sig=1.0, n_bodies=2, periodic_bounds=False, L=10.0, c=2.0, drag=False
    ):
        self.eps = eps
        self.sig = sig
        self.n_bodies = n_bodies
        self.periodic_bounds = periodic_bounds
        self.L = L
        self.c = c
        self.drag = drag

    def periodic_distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]

        if dx > (0.5 * self.L):
            dx -= self.L
        elif dx < (-0.5 * self.L):
            dx += self.L
        else:
            pass

        if dy > (0.5 * self.L):
            dy -= self.L
        elif dy < (-0.5 * self.L):
            dy += self.L
        else:
            pass

        return np.array([dx, dy])

    def rhs(self, t, u):
        u = u.reshape((self.n_bodies, 4))
        u_1 = np.zeros_like(u)
        P = 0
        K = 0
        u_1[0, ::2] += u[0, 1::2]

        if self.drag:
            u_1[0, ::2] = -self.c*u[0, ::2]
    
        for i in range(0, self.n_bodies - 1):
            u_1[i + 1, ::2] += u[i + 1, 1::2]
            K += 0.5*(np.linalg.norm(u[i + 1][1::2])**2)
            if self.drag:
                u_1[i + 1, ::2] = -self.c*u[i + 1, ::2]
            for j in range(i + 1, self.n_bodies):
                # find vector r and its mag
                # assume ordinary distance but use periodic distance if that's closer
                r_ij = -self.periodic_distance(u[i, ::2], u[j, ::2])
                r_ji = -1 * r_ij
                mag = np.linalg.norm(r_ij)
                sig_mag = (self.sig / mag)
                u_1[i, 1::2] += -(
                    24
                    * self.eps
                    * r_ij
                    * (2 * (sig_mag ** 12) - (sig_mag ** 6))
                    / (mag**2)
                )
                u_1[j, 1::2] += -(
                    24
                    * self.eps
                    * r_ji
                    * (2 * (sig_mag ** 12) - (sig_mag ** 6))
                    / (mag**2)
                )

        return u_1.flatten()
