# Lattice Monte Carlo simulations of Liquid Crystal symmetry models

This package provides methods for running Monte Carlo simulations of lattice
symmetry models that describe some types of liquid crystals. It is a follow-up
on [previous code](https://github.com/lyckantropen/biaxmc2) that was used for
obtaining results for my PhD dissertation, ["Lattice Models of Biaxial,
Tetrahedratic and Chiral
Order"](https://fais.uj.edu.pl/documents/41628/d13cf0c3-7942-425c-9ab4-26fecb7cb518).
The aim is to reproduce all the results using well-tested code written in
modern Python.

## Installation

Requires Python 3.9+:

```bash
pip install -e .                    # Basic installation
pip install -e .[tests]             # With testing dependencies  
pip install -e .[notebook]          # With Jupyter support
pip install -e .[tests,notebook]    # All dependencies
```

## Examples

**`example.py`** / **`example.ipynb`**: Basic single-temperature simulation  
**`example_synchronized_pt.py`**: Parallel tempering with multiple temperatures  
**`example_tensorboard.ipynb`**: Real-time monitoring with TensorBoard

```bash
python example.py                    # Basic simulation
python example_synchronized_pt.py    # Parallel tempering
jupyter notebook example.ipynb       # Interactive version
```

## Data Persistence

Set `working_folder` to automatically save simulation state and results:

```text
working_folder/
├── data/order_parameters.npz       # Order parameter time series
├── states/lattice_state.npz        # Lattice configuration
├── summary.json                    # Human-readable summary
└── logs/simulation.log             # Debug logs
```

### Parallel Tempering Structure

```text
parallel_tempering_output/
├── parameters/T0.30_lam0.35_tau1.00/  # Use for analysis
├── parameters/T0.35_lam0.35_tau1.00/
├── process_000/                       # Individual process data (debug)
└── tensorboard/                       # TensorBoard logs
```

**Analysis**: Use `parameters/` folders (consolidated across exchanges)  
**Debugging**: Use `process_XXX/` folders (individual process data)

### Data Formats

- **Order Parameters**: Time series of `energy`, `q0`, `q2`, `p`, `d322`, `w`
- **Lattice State**: Complete configuration with particles, properties, dimensions  
- **Summary**: Human-readable progress and latest values

### Loading Data

```python
import numpy as np

# Load time series
data = np.load("working_folder/data/order_parameters.npz")
order_params = data['order_parameters']

# Auto-recovery: set auto_recover=True to resume interrupted simulations
```

## References

- [Trojanowski, Karol, Michaƚ Cieśla, and Lech Longa. "Modulated nematic
  structures and chiral symmetry breaking in 2D." Liquid Crystals 44.1 (2017):
  273-283.](https://arxiv.org/pdf/1607.02297.pdf)
- [Trojanowski, Karol, et al. "Tetrahedratic mesophases, chiral order, and
  helical domains induced by quadrupolar and octupolar interactions." Physical
  Review E 86.1 (2012):
  011704.](https://strathprints.strath.ac.uk/41276/1/PhysRevE_86_011704.pdf)
- [Trojanowski, K., et al. "Theory of phase transitions of a biaxial nematogen
  in an external field." Molecular Crystals and Liquid Crystals 540.1 (2011):
  59-68.](https://www.tandfonline.com/doi/abs/10.1080/15421406.2011.568329)
