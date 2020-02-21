"""
Plotting convenience functions.
"""

from math import ceil

import ipywidgets as widgets
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from model_base import get_ext_input

# define basics
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
plt.style.use("seaborn-muted")
INPUT_START = 1000  # dt, i.e. 100ms
LABEL_SIZE = 16


def setup_sliders_layout(model_specific_sliders):
    """
    Set up interactive part of the plot, i.e. sliders and grid layout.

    model_params: list of model parameters names
    """
    assert isinstance(model_specific_sliders, dict)
    num_model_sliders = len(model_specific_sliders)

    # define general sliders
    I_m_slider = widgets.FloatSlider(
        min=-5, max=20, step=0.5, value=10.0, description="I max"
    )
    T_slider = widgets.IntSlider(
        min=500, max=2000, step=5, value=750, description="time"
    )
    I_types = widgets.ToggleButtons(
        options=["constant", "sq. pulse", "sine", "ramp"],
        value="constant",
        description="Current type",
        disabled=False,
        layout=widgets.Layout(height="auto", width="auto"),
    )
    I_period = widgets.FloatSlider(
        min=10, max=1000, step=5, value=200, description="I period"
    )

    # define grid
    grid = widgets.GridspecLayout(ceil(5 + num_model_sliders / 2), 2)
    grid[0, :] = widgets.Button(
        description="Model parameters",
        layout=widgets.Layout(height="auto", width="auto"),
    )
    # assign model sliders
    for idx, (_, slider) in enumerate(model_specific_sliders.items()):
        grid[idx // 2 + 1, idx % 2] = slider

    grid[idx // 2 + 2, :] = widgets.Button(
        description="External current parameters",
        layout=widgets.Layout(height="auto", width="auto"),
    )
    grid[idx // 2 + 3, 0] = I_period
    grid[idx // 2 + 4, 0] = I_m_slider
    grid[idx // 2 + 4, 1] = T_slider
    grid[idx // 2 + 5, :] = I_types
    sliders = {
        **model_specific_sliders,
        "I_max": I_m_slider,
        "I_period": I_period,
        "T": T_slider,
        "current_type": I_types,
    }
    return grid, sliders


def integrate_and_plot(model_cls, **kwargs):
    """
    Integrate the model given its parameters and plot.
    """
    T = kwargs.pop("T")
    I_max = kwargs.pop("I_max")
    I_period = kwargs.pop("I_period")
    current_type = kwargs.pop("current_type")
    model = model_cls(parameters=kwargs, T=T)
    ext_current = np.zeros((model.n_points + 1))
    input_length = ext_current.shape[0] - INPUT_START
    ext_current[INPUT_START:] = get_ext_input(
        I_max, I_period, current_type, model.T_total, input_length
    )
    model.set_input(ext_current)
    t, y = model.integrate()

    # set up figure
    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    # set up axis for timeseries of input current
    ax2 = fig.add_subplot(spec[2, :2])
    ax2.set_ylim([-20, 20])
    ax2.set_ylabel("INPUT CURRENT [AU]", size=LABEL_SIZE)
    ax2.set_xlabel("TIME [ms]", size=LABEL_SIZE)
    ax2.axvline(100.0, 0, 1, linestyle="--", color="grey", linewidth=0.7)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="both", which="major", labelsize=LABEL_SIZE - 2)

    # set up axis for timeseries of state vector
    ax1 = fig.add_subplot(spec[:2, :2], sharex=ax2)
    ax1.set_ylim([-90, 20])
    ax1.set_ylabel("MEMBRANE POTENTIAL [mV]", size=LABEL_SIZE)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.axvline(100.0, 0, 1, linestyle="--", color="grey", linewidth=0.7)
    ax1.tick_params(axis="both", which="major", labelsize=LABEL_SIZE - 2)
    ax12 = ax1.twinx()
    ax12.set_ylim([-20, 10])
    ax12.set_yticklabels([])
    ax12.set_yticks([])
    ax12.spines["right"].set_visible(False)
    ax12.spines["top"].set_visible(False)
    ax12.spines["bottom"].set_visible(False)
    ax12.tick_params(axis="both", which="major", labelsize=LABEL_SIZE - 2)

    # set up axis for scatter u vs v
    ax3 = fig.add_subplot(spec[:2, 2], sharey=ax1)
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.set_xlabel("MEMBRANE RECOVERY", size=LABEL_SIZE)
    scatter_colors = colors[3]
    ax3.set_ylim([-90, 20])
    ax3.set_xlim([-20, 10])
    ax3.tick_params(axis="both", which="major", labelsize=LABEL_SIZE - 2)

    # plot
    ax1.plot(t, y[0, :], color=colors[0], linewidth=2.5)
    ax12.plot(t, y[1:, :].T, color=colors[1])
    ax2.plot(t, model.ext_current[1:], color=colors[2])
    ax3.scatter(y[1, :], y[0, :], s=7, c=scatter_colors)
    plt.suptitle(f"Number of spikes: {model.num_spikes}", size=LABEL_SIZE + 3)
