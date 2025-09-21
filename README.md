# Lattice Monte Carlo simulations of Liquid Crystal symmetry models

This package provides methods for running Monte Carlo simulations of lattice
symmetry models that describe some types of liquid crystals. It is a follow-up
on [previous code](https://github.com/lyckantropen/biaxmc2) that was used for
obtaining results for my PhD dissertation, ["Lattice Models of Biaxial,
Tetrahedratic and Chiral
Order"](https://fais.uj.edu.pl/documents/41628/d13cf0c3-7942-425c-9ab4-26fecb7cb518).
The aim is to reproduce all the results using well-tested code written in
modern Python.

## Installing

Python 3.9+ is required. After cloning the repository, the package can be installed in development mode:

```bash
# Install the package in development mode
pip install -e .

# Or install with optional dependencies for testing
pip install -e .[tests]

# Or install with optional dependencies for notebooks
pip install -e .[notebook]

# Or install with all optional dependencies
pip install -e .[tests,notebook]
```

The project now uses modern Python packaging with `pyproject.toml` for configuration.

## Quick Start & Examples

The package includes several examples to help you get started with different types of simulations:

### Basic Single-Temperature Simulation

**File**: `example.py` or `example.ipynb`

The most straightforward way to start with latticemc. This example demonstrates:

- Setting up a 3D lattice with partially ordered initialization
- Configuring Monte Carlo simulation parameters (temperature, coupling strength)
- Running a single-temperature simulation with data collection
- Tracking order parameters and fluctuations over time

```python
# Quick start - run the basic example
python example.py
```

The Jupyter notebook version (`example.ipynb`) additionally includes:

- Comprehensive visualization of simulation results
- Real-time plotting of energy evolution and order parameters
- Educational explanations of each simulation component
- Suggestions for next steps and parameter studies

### Parallel Tempering Simulation  

**File**: `example_synchronized_pt.py`

For advanced sampling of systems with complex energy landscapes:

- Multiple temperature replicas running simultaneously
- Synchronized replica exchange using barrier synchronization
- Enhanced exploration of configuration space
- Automatic TensorBoard logging for temperature-dependent properties

This method is particularly useful for:

- Systems with first-order phase transitions
- Complex energy landscapes with multiple metastable states
- Studying temperature-dependent behavior efficiently

### TensorBoard Integration

**File**: `example_tensorboard.ipynb`

Demonstrates real-time monitoring and visualization capabilities:

- Live tracking of simulation progress in TensorBoard
- Temperature-dependent order parameter plots
- Replica exchange statistics and acceptance rates
- Energy vs. temperature phase diagrams

Launch TensorBoard with: `tensorboard --logdir runs/`

### Key Simulation Outputs

All examples track important physical quantities:

- **Energy**: Total system energy per particle
- **Order Parameters**:
  - `q0`, `q2`: Orientational order for liquid crystal phases
  - `p`: Nematic order parameter (molecular alignment)
  - `d322`, `w`: Additional structural order parameters
- **Fluctuations**: Variance in order parameters (related to susceptibilities)
- **Acceptance Rates**: Monte Carlo move acceptance statistics

### Running Examples

```bash
# Basic simulation
python example.py

# Parallel tempering
python example_synchronized_pt.py

# Interactive notebooks (requires jupyter)
pip install -e .[notebook]
jupyter notebook example.ipynb
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
