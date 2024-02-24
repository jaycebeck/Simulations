import numpy as np


class Euler:
    def __init__(self):
        pass

    def step(self, ode, t, dt, u_0):
        u_1 = u_0 + dt * ode.rhs(t, u_0)
        return u_1


class Cromer:
    def __init__(self):
        pass

    def step(self, ode, t, dt, u_0):
        u_x = u_0 + ode.rhs(t, u_0) * dt
        u_new = np.zeros_like(u_0)
        u_new[1::2] = u_x[1::2]
        u_new[::2] = u_0[::2]

        u_1 = u_0 + ode.rhs(t, u_new) * dt

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
