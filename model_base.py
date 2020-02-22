"""
Base for spiking models.
"""

import numpy as np

from scipy.integrate import solve_ivp
from scipy.signal import square


def terminal(func):
    """
    Decorate event function if it's terminal.
    For scipy's intergator.
    """
    func.terminal = True
    return func


class BaseSpikingModel:
    """
    Base class for spiking models. All times are represented in ms.
    """

    dt = 0.1
    y0 = []
    plotting_bounds = []
    index_voltage_variable = 0
    spike_condition = None
    required_params = []

    def __init__(self, parameters, T=250):
        assert isinstance(parameters, dict)
        assert all(param in parameters for param in self.required_params)

        # integration params
        self.params = parameters
        self.T_total = T
        self.n_points = int(np.floor(self.T_total / self.dt))
        # input
        self.ext_current = None
        self.num_spikes = 0

    @property
    def indices_aux(self):
        """
        Return list of other indices, i.e. all except the voltage variable.
        """
        return [
            i for i in range(len(self.y0)) if i != self.index_voltage_variable
        ]

    def set_input(self, ext_current):
        """
        Set input to the neuron.
        """
        # if I is scalar, cast to array
        if isinstance(ext_current, float):
            ext_current = np.ones((self.n_points,)) * ext_current
        # input must have correct shape: at least n_points
        assert len(ext_current) >= self.n_points
        self.ext_current = ext_current

    @terminal
    def reset_condition(self, t, y, I):
        """
        Reset condition - spike.
        It is terminal, i.e. terminate and reset intergation after spike.
        """
        return y[self.index_voltage_variable] - self.spike_condition

    def _apply_reset(self, y):
        """
        Apply reset condition on the vector y.
        """
        return y

    def _rhs(self, t, y, I):
        """
        Right hand side of the equation:
        Reimplement for each model.
        """
        raise NotImplementedError

    def integrate(self):
        """
        Run full integration.
        """
        ts = []
        ys = []
        y0 = self.y0
        t = 0
        # start integration
        t_eval = np.linspace(0, self.T_total, self.n_points, endpoint=False)
        while True:
            # solve until spike
            sol = solve_ivp(
                self._rhs,
                t_span=[t, self.T_total],
                t_eval=t_eval,
                y0=y0,
                args=[self.ext_current],
                events=self.reset_condition,
            )
            ts.append(sol.t)
            ys.append(sol.y)
            so_far_length = sum(len(t_temp) for t_temp in ts)
            # if terminated using event, i.e. spike
            if sol.status == 1:
                self.num_spikes += 1
                # restart with new t0 - the last time from previous integration
                t = sol.t[-1]
                t_eval = np.linspace(
                    t,
                    self.T_total,
                    self.n_points - so_far_length,
                    endpoint=False,
                )
                # restart with new initial conditions as per reset
                y0 = sol.y[:, -1].copy()
                y0 = self._apply_reset(y0)
            # if not terminated using event, i.e. end of integration
            else:
                break
        # stitch results together
        t = np.concatenate(ts)
        y = np.concatenate(ys, axis=1)
        assert t.shape[0] == self.n_points, f"{t.shape[0]} vs. {self.n_points}"
        # return as time and y vector
        return t, y


def get_ext_input(I_max, I_period, current_type, t_total, input_length):
    """
    Construct external current of given type.
    """
    if current_type == "constant":
        return I_max
    elif current_type == "sine":
        time = np.linspace(0, t_total, input_length)
        return I_max * np.sin(2 * np.pi * time * (1.0 / I_period))
    elif current_type == "sq. pulse":
        time = np.linspace(0, t_total, input_length)
        return I_max * square(2 * np.pi * time * (1.0 / I_period))
    elif current_type == "ramp":
        time = np.linspace(0, t_total, input_length)
        return ((I_max / I_period) * time) * (time < I_period) + I_max * (
            time > I_period
        )
    elif current_type == "Ornstein-Uhlenbeck":
        time = np.linspace(0, t_total, input_length)
        return simulate_ornstein_uhlenbeck(
            I_max, np.abs(I_max / 5.0), I_period, time
        )
    else:
        raise ValueError("Unknown current type")


def simulate_ornstein_uhlenbeck(mu, sigma, tau, time):
    """
    Simulate Ornstein-Uhlenbeck process.
    """
    dt = 0.1  # ms
    x = np.zeros_like(time)
    for i in range(x.shape[0] - 1):
        x[i + 1] = (
            x[i]
            + dt * (-(x[i] - mu) / tau)
            + sigma * np.sqrt(2.0 / tau) * np.sqrt(dt) * np.random.randn()
        )
    return x
