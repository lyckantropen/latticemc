* p order parameter calculated as abs will change fluctuation
X replica exchange doesn't affect history returned from SimulationProcess (*)
* local history takes too much memory and is not used; OrderParametersCalculator appends instead of filling
X LatticeState.latticeAverages is not necessary
* Finished SimulationProcess doesn't notify and adjacent temperatures don'know if they can do PT
X split the order parameters from state variables in particle dtype
* send a termination message to SimulationProcess instead of .terminate()