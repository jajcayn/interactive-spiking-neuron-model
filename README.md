# Interactive spiking neuron models
_Interactive spiking neuron models_

Interactive models of spiking neuron in `jupyter` notebook! You can play with model (intrinsic) and integration parameters (external current and integration length) and see how the dynamics of the neuron evolves. 

## Launch interactive binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jajcayn/interactive-spiking-neuron-model/master)

Nothing required, just click on the button, wait a bit till binder builds all that is necessary, choose a model notebook and then click `Cell` -> `Run All`. The last cell contains the interactive plot.

## Homemade
```bash
git clone https://github.com/jajcayn/interactive-spiking-neuron-model.git
cd interactive-spiking-neuron-model
pip install -r requirements.txt
jupyter notebook
```
and launch `*_model.ipynb`. Then just `Cell` -> `Run All` and you are done.

## Models
* Izhikevich model
* (others soon)

## Implement your own
If you have an idea for outstanding spiking neuron model (or just want to implement an existing one) nothing is easier than do it yourself! Check out `models.py` how e.g. `IzhikevichModel` is implemented. In general, all models should be subclassed from `model_base.BaseSpikingModel` and the following functions and attributes should be reimplemented:
* `y0` for initial conditions
* `index_voltage_variable` for index of membrane potential within whole arbitrarily long state vector 
* `spike_condition` for setting the condition for spike
* `required_params` for list of required parameters of the model
* `_apply_reset(self, y)` is a function for applying after-spike reset acting on state vector `y`
* `_rhs(self, t, y, I)` is a function that computes single dt step of the model dynamics, i.e. the right hand side of model ODE

And you're done! All other stuff, mainly the integration itself is handled by the `BaseSpikingModel`. If you are really eager to see interactive demo of some model not yet implemented here but you're no good with python, just drop me a message or open an issue and I'll implement it.

**Happy modelling!**
