from .definitions import LatticeState, OrderParametersHistory


def failsafeSaveSimulation(e: Exception, state: LatticeState, orderParametersHistory: OrderParametersHistory):
    try:
        print(f'Exception: {e}')
        print(f'Attempting to save the state of the simulation')

        from datetime import datetime
        from joblib import dump
        from os import getcwd
        from uuid import uuid1
        from pathlib import Path

        destPath = Path(getcwd()) / Path(str(uuid1()))

        print(f'Will attempt to save simulation state to {destPath}')
        destPath.mkdir()

        ts = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now())

        descStr = (
            f'time={ts}\niterations={state.iterations}\n'
            f'temperature={state.temperature}\ntau={state.tau}\n'
            f'lambda={state.lam}\nlatticeSize={state.lattice.particles.shape}\n'
            f'wiggleRate={state.wiggleRate}\n'
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
