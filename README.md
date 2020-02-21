# Interactive spiking neuron models
_Interactive spiking neuron models_

Interactive models of spiking neuron in `jupyter` notebook! You can play with model (intrinsic) and integration parameters (external current and integration length) and see how the dynamics of the neuron evolves. 

## Launch interactive binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jajcayn/interactive-spiking-neuron-model/master)

Nothing required, just click on the button, wait a bit till binder builds all that is necessary, choose a model notebook and then click `Cell` -> `Run All`. The last cell contains the interactive plot.

## Homemade
```bash
git clone https://github.com/jajcayn/izhikevich-model.git
cd izhikevich-model
pip install -r requirements.txt
jupyter notebook
```
and launch `*_model.ipynb`. Then just `Cell` -> `Run All` and you are done.

## Models
* Izhikevich model
* (others soon)

Happy modelling!
