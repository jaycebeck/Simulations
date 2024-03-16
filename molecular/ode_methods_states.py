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
        inc, P, ignore_ = ode.rhs(t, u_0)
        u_x = u_0 + inc * dt
        u_new = np.zeros_like(u_0)

        u_new[1::2] = u_x[1::2]
        u_new[::2] = u_0[::2]

        inc, ignore_, K = ode.rhs(t, u_new)

        u_1 = (u_0 + inc * dt) % self.L
        u_1[1::2] = u_new[1::2]

        return u_1, P, K


class Integrator:
    def __init__(self, ode, method):
        self.ode = ode
        self.method = method

    def integrate(self, interval, dt, u_0):
        t_0 = interval[0]
        t_end = interval[1]

        times = [t_0]
        states = [u_0]
        e_potential = [0]
        e_kinetic = [0]

        t = t_0
        while t < t_end:
            dt_ = min(dt, t_end - t)
            u_1, P, K = self.method.step(self.ode, t, dt_, u_0)
            t = t + dt_
            u_0 = u_1

            times.append(t)
            states.append(u_1)
            e_potential.append(P)
            e_kinetic.append(K)

        return (
            np.array(times),
            np.array(states),
            np.array(e_potential),
            np.array(e_kinetic),
        )


class LennardJones:
    """This is an example class for an ODE specification"""

    def __init__(
        self,
        eps=1.0,
        sig=1.0,
        n_bodies=2,
        periodic_bounds=False,
        L=10.0,
        c=2.0,
        drag=False,
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

        K += 0.5 * (np.linalg.norm(u[0][1::2]) ** 2)

        if self.drag:
            u_1[:, 1::2] = -self.c * u[:, 1::2]

        for i in range(0, self.n_bodies - 1):
            u_1[i + 1, ::2] += u[i + 1, 1::2]
            K += 0.5 * (np.linalg.norm(u[i + 1][1::2]) ** 2)
            for j in range(i + 1, self.n_bodies):
                # find vector r and its mag
                # assume ordinary distance but use periodic distance if that's closer
                r_ij = self.periodic_distance(u[i, ::2], u[j, ::2])
                r_ji = -1 * r_ij
                mag = np.linalg.norm(r_ij)
                sig_mag = self.sig / mag
                u_1[i, 1::2] += (
                    24 * self.eps * r_ij * (2 * (sig_mag**12) - (sig_mag**6)) / (mag**2)
                )
                u_1[j, 1::2] += (
                    24 * self.eps * r_ji * (2 * (sig_mag**12) - (sig_mag**6)) / (mag**2)
                )
                P += 4 * self.eps * ((sig_mag**12) - (sig_mag**6))

        return u_1.reshape(4 * self.n_bodies), P, K
