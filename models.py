"""
Set of spiking neural models.
"""


import ipywidgets as widgets
import numpy as np

from model_base import BaseSpikingModel

IZHIKEVICH_MODEL_SLIDERS = {
    "a": widgets.FloatSlider(
        min=0.02, max=0.1, step=0.008, value=0.02, description="a"
    ),
    "b": widgets.FloatSlider(
        min=0.2, max=0.25, step=0.01, value=0.2, description="b"
    ),
    "c": widgets.IntSlider(
        min=-65, max=-50, step=5, value=-65, description="c"
    ),
    "d": widgets.FloatSlider(
        min=0.05, max=8.0, step=0.1, value=8.0, description="d"
    ),
}

ADEX_MODEL_SLIDERS = {
    "g_L": widgets.FloatSlider(
        min=0.0, max=20.0, step=0.5, value=10.0, description="$g_{L}$"
    ),
    "E_L": widgets.FloatSlider(
        min=-100.0, max=-20.0, step=5.0, value=-65.0, description="$E_{L}$"
    ),
    "Delta_T": widgets.FloatSlider(
        min=0.0, max=10.0, step=0.1, value=1.5, description="$\Delta_{T}$"
    ),
    "V_T": widgets.FloatSlider(
        min=-100.0, max=-20.0, step=5.0, value=-50.0, description="$V_{T}$"
    ),
    "C": widgets.IntSlider(
        min=100, max=500, step=50, value=200, description="C"
    ),
    "a": widgets.FloatSlider(
        min=0.0, max=30.0, step=0.5, value=15, description="a"
    ),
    "tau_w": widgets.FloatSlider(
        min=50.0, max=500.0, step=50.0, value=200.0, description=r"$\tau_{w}$"
    ),
    "v_r": widgets.FloatSlider(
        min=-100.0, max=-30.0, step=5.0, value=-40.0, description="$V_{r}$"
    ),
    "b": widgets.FloatSlider(
        min=0.0, max=80.0, step=5.0, value=40.0, description="b"
    ),
}


class IzhikevichModel(BaseSpikingModel):
    """
    Izhikevich model of spiking neuron.

    Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE
        Transactions on neural networks, 14(6), 1569-1572.
    """

    y0 = [-70.0, -14.0]
    index_voltage_variable = 0
    spike_condition = 30.0  # mV
    required_params = ["a", "b", "c", "d"]

    def _apply_reset(self, y):
        """
        Spike reset for model:
        v <- c
        u <- u + d
        """
        y[0] = self.params["c"]
        y[1] += self.params["d"]
        return y

    def _rhs(self, t, y, I):
        """
        Right hand side of the equation:
        v' = 0.04*v^2 + 5*v + 140 - u + I
        u' = a*(b*v - u)
        """
        v, u = y
        t_idx = int(t / self.dt)
        v_deriv = 0.04 * np.power(v, 2) + 5.0 * v + 140.0 - u + I[t_idx]
        u_deriv = self.params["a"] * (self.params["b"] * v - u)
        return (v_deriv, u_deriv)


class AdExIntegrateAndFire(BaseSpikingModel):
    """
    Adaptive exponential integrate-and-fire model.

    Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire
        model as an effective description of neuronal activity. Journal of
        neurophysiology, 94(5), 3637-3642.
    """

    y0 = [-70.0, 0.0]
    index_voltage_variable = 0
    spike_condition = -40.0  # mV
    required_params = [
        "g_L",
        "E_L",
        "Delta_T",
        "V_T",
        "C",
        "a",
        "tau_w",
        "v_r",
        "b",
    ]

    def _apply_reset(self, y):
        """
        Spike reset for model:
        v <- v_r
        w <- w + b
        """
        y[0] = self.params["v_r"]
        y[1] += self.params["b"]

    def _rhs(self, t, y, I):
        """
        Right hand side of the equation:
        v' = (-gL(V - EL) + gL*delta_T*exp(V - VT / delta_T) - w + I) / C
        w' = (a(V - EL) - w) / tau_w
        """
        v, w = y
        t_idx = int(t / self.dt)
        v_deriv = (
            -self.params["g_L"] * (v - self.params["E_L"])
            + self.params["g_L"]
            * self.params["Delta_T"]
            * np.exp((v - self.params["V_T"]) / self.params["Delta_T"])
            - w
            + I[t_idx]
        ) / self.params["C"]
        w_deriv = (
            self.params["a"] * (v - self.params["E_L"]) - w
        ) / self.params["tau_w"]
        return (v_deriv, w_deriv)
