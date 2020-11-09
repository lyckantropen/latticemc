from .definitions import LatticeState, OrderParametersHistory


def failsafeSaveSimulation(e: Exception, state: LatticeState, orderParametersHistory: OrderParametersHistory):
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

        destPath = Path(getcwd()) / Path(str(uuid1()))

        print(f'Will attempt to save simulation state to {destPath}')
        destPath.mkdir()

        ts = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now())

        descStr = (
            f'time={ts}\niterations={state.iterations}\n'
            f'parameters={state.parameters}\n'
            f'latticeSize={state.lattice.particles.shape}\n'
            f'wiggleRate={state.wiggleRate}\n\n'
            f'{tb}\n'
        )
        descFile = destPath / 'desc.txt'
        descFile.write_text(descStr)

        stateFile = destPath / f'latticeState.dump'
        orderParametersHistoryFile = destPath / f'orderParametersHistory.dump'
        dump(state, stateFile.as_posix(), compress=('xz', 9))
        dump(orderParametersHistory, orderParametersHistoryFile.as_posix(), compress=('xz', 9))

        print(f'Saved lattice and parameters to {destPath}. Use joblib.load to restore them.')
    except Exception as e2:
        print(f'Exception: {e2} during the attempt to save the state of the simulation.')
