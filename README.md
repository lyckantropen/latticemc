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

## Working Folder & Data Persistence

The latticemc package provides automatic data saving and recovery functionality through working folders. When you specify a `working_folder` parameter, the simulation automatically saves its state and results, enabling:

- **Automatic Recovery**: Resume interrupted simulations from the last saved state
- **Data Persistence**: All simulation results saved in organized folder structure
- **Post-Processing**: Easy access to saved data for analysis and visualization
- **Process Isolation**: In parallel tempering, each temperature replica gets its own subfolder

### Folder Structure

When using working folders, the following directory structure is automatically created:

```text
working_folder/
├── logs/                           # Simulation logs
│   ├── simulation.log              # Single simulation logs (with parameter ID)
│   ├── parallel_tempering_main.log # Main process logs (parallel tempering)
│   └── process_XXX/logs/           # Individual process logs (parallel tempering)
│       └── simulation.log
├── data/                           # Simulation results
│   ├── order_parameters.npz       # Time series of order parameters
│   └── fluctuations.npz           # Time series of fluctuations  
├── states/                         # Simulation state snapshots
│   ├── lattice_state.npz          # Complete lattice configuration
│   └── simulation_state.pkl       # Simulation metadata and progress
├── summary.json                    # Human-readable simulation summary
└── simulation_in_progress.marker   # Progress marker (deleted on completion)
```

#### For Parallel Tempering with SimulationRunner

When using `SimulationRunner` for parallel tempering, the working folder
contains both process-based and parameter-based organization:

```text
parallel_tempering_output/
├── logs/
│   └── parallel_tempering_main.log # Main coordination process logs
├── tensorboard/                    # TensorBoard logs (when enabled)
│   └── ... (event files for real-time monitoring)
├── process_000/                    # Individual process T₁ replica
│   ├── logs/
│   │   └── simulation.log          # Process-specific logs with T, λ, τ values
│   ├── data/
│   │   ├── order_parameters.npz
│   │   └── fluctuations.npz
│   ├── states/
│   │   ├── lattice_state.npz
│   │   └── simulation_state.joblib
│   └── summary.json
├── process_001/                    # Individual process T₂ replica
│   └── ... (same structure)
├── process_XXX/                    # Additional process replicas
│   └── ... (same structure)
└── parameters/                     # Parameter-based organization (NEW)
    ├── T0.30_lam0.35_tau1.00/      # Data organized by parameter values
    │   ├── data/
    │   │   ├── order_parameters.npz # Consolidated data from all processes with these parameters
    │   │   └── fluctuations.npz     # Merged across parallel tempering exchanges
    │   ├── states/
    │   │   └── latest_state.npz     # Latest state for this parameter set
    │   └── summary.json             # Parameter-specific summary
    ├── T0.35_lam0.35_tau1.00/       # Next parameter set
    │   └── ... (same structure)
    └── T1.65_lam0.35_tau1.00/       # Additional parameter sets
        └── ... (same structure)
```

#### Understanding the Dual Organization

The `SimulationRunner` creates two complementary folder structures:

1. **Process-based folders** (`process_XXX/`):
   - Each individual simulation process gets its own folder
   - Contains the raw simulation data as it runs
   - Useful for monitoring individual process progress
   - Required for simulation recovery and checkpointing

2. **Parameter-based folders** (`parameters/T_X_lam_Y_tau_Z/`):
   - Data organized by the actual parameter values (T, λ, τ)
   - Consolidates results across parallel tempering exchanges
   - In parallel tempering, the same parameter set may run on different processes over time
   - Contains the final, merged data for each unique parameter combination
   - **This is typically what you want for analysis and post-processing**

The parameter-based organization is especially important for parallel tempering
because temperature exchanges mean that process 0 might run different
temperatures over time, but you want all data for temperature T=0.85 in one
place regardless of which process computed it.

### Saved Data Format

#### Order Parameters (`order_parameters.npz`)

Numpy archive containing time series of physical observables:

- `energy`: Total system energy per particle
- `q0`: Scalar order parameter (S parameter)
- `q2`: Tensor order parameter  
- `p`: Nematic order parameter
- `d322`: Biaxial order parameter
- `w`: Octupolar order parameter

#### Fluctuations (`fluctuations.npz`)

Numpy archive containing variance measures:

- Same keys as order parameters
- Calculated over sliding window (default 100 steps)
- Related to physical susceptibilities

#### Lattice State (`lattice_state.npz`)

Complete lattice configuration snapshot:

- `particles`: Quaternion orientations for each lattice site
- `properties`: Derived particle properties
- `lattice_X`, `lattice_Y`, `lattice_Z`: Lattice dimensions
- `current_step`: Simulation progress
- `relevant_history_length`: Valid data length

#### Simulation State (`simulation_state.pkl`)

Metadata and simulation progress:

- Current Monte Carlo step
- Simulation parameters (T, λ, τ)
- Fluctuations window size
- Save interval settings
- Relevant history tracking

#### Summary (`summary.json`)

Human-readable overview:

```json
{
  "current_step": 1000,
  "total_cycles": 1000,
  "parameters": {
    "temperature": 0.9,
    "lam": 0.3,
    "tau": 1.0
  },
  "latest_order_parameters": {
    "energy": -7.845,
    "q0": 0.432,
    "p": 0.156
  },
  "latest_fluctuations": {
    "energy": 0.023,
    "q0": 0.0089
  }
}
```

### Log Files

All simulation logs are saved with process identification:

#### Single Simulation Logs

- Filename: `T{temp}_lam{lam}_tau{tau}.log`
- Format: `timestamp - T0.9_lam0.3_tau1.0 - module - level - message`

#### Parallel Tempering Logs

- Main process: `parallel_tempering_main.log`
- Individual processes: `process_XXX/logs/simulation.log`
- Format: `timestamp - P000_T1.0_λ1.0_τ0.1 - module - level - message`

### Data Access Recommendations

**For Single Simulations:**

- Use the standard structure: `data/`, `states/`, `logs/`, `summary.json`

**For Parallel Tempering Analysis:**

- **Use `parameters/T_X_lam_Y_tau_Z/` folders** for temperature-specific analysis
- These contain consolidated data for each parameter set across all exchanges
- Example: All data for T=0.85 will be in `parameters/T0.85_lam0.35_tau1.00/`

**For Debugging Parallel Tempering:**

- Use `process_XXX/` folders to examine individual process behavior
- Check `logs/parallel_tempering_main.log` for coordination and exchange information

**For Data Loading:**

```python
from pathlib import Path
import numpy as np

# Load parameter-specific data (recommended for analysis)
param_folder = Path("working_folder/parameters/T0.85_lam0.35_tau1.00")
order_params = np.load(param_folder / "data" / "order_parameters.npz")

# Load process-specific data (for debugging)
process_folder = Path("working_folder/process_000")  
process_data = np.load(process_folder / "data" / "order_parameters.npz")
```

### Recovery and Continuation

The `simulation_in_progress.marker` file indicates an incomplete simulation. If `auto_recover=True`:

1. **Automatic Detection**: Simulation detects existing marker on startup
2. **State Recovery**: Loads lattice configuration, history, and progress
3. **Seamless Continuation**: Resumes from the exact step where it stopped
4. **Marker Cleanup**: Removes marker file upon successful completion

### Loading Saved Data

All examples demonstrate how to load and analyze saved data:

```python
import numpy as np
import json
from pathlib import Path

# Load order parameters
data = np.load("working_folder/data/order_parameters.npz")
order_params = data['order_parameters']

# Load summary
with open("working_folder/summary.json") as f:
    summary = json.load(f)

# Load lattice state for visualization
state_data = np.load("working_folder/states/lattice_state.npz")
particles = state_data['particles']
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
