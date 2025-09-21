"""Emergency save functionality for simulation state."""

from .definitions import LatticeState, OrderParametersHistory


def failsafe_save_simulation(e: Exception, state: LatticeState, order_parameters_history: OrderParametersHistory):
    """Save simulation state to disk when an exception occurs."""
    try:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        print(f'Attempting to save the state of the simulation: {state.parameters}')

        from datetime import datetime
        from os import getcwd
        from pathlib import Path
        from uuid import uuid1

        from joblib import dump

        dest_path = Path(getcwd()) / Path(str(uuid1()))

        print(f'Will attempt to save simulation state to {dest_path}')
        dest_path.mkdir()

        ts = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now())

        lattice_shape = getattr(state.lattice.particles, 'shape', 'unknown') if state.lattice.particles is not None else 'None'
        desc_str = (
            f'time={ts}\niterations={state.iterations}\n'
            f'parameters={state.parameters}\n'
            f'latticeSize={lattice_shape}\n'
            f'wiggle_rate={state.wiggle_rate}\n\n'
            f'{tb}\n'
        )
        desc_file = dest_path / 'desc.txt'
        desc_file.write_text(desc_str)

        state_file = dest_path / 'lattice_state.dump'
        order_parameters_history_file = dest_path / 'order_parameters_history.dump'
        dump(state, state_file.as_posix(), compress=('xz', 9))
        dump(order_parameters_history, order_parameters_history_file.as_posix(), compress=('xz', 9))

        print(f'Saved lattice and parameters to {dest_path}. Use joblib.load to restore them.')
    except Exception as e2:
        print(f'Exception: {e2} during the attempt to save the state of the simulation.')
