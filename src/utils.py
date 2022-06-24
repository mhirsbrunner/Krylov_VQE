from pathlib import Path
import pickle as pkl

# File structure
project_src = Path(__file__).parent
project_root = project_src.parent
vqe_data_dir = project_root / 'vqe_circuit_data'


def save_vqe_result(params, data_fname):
    with open(vqe_data_dir / (data_fname + '.pickle'), 'wb') as handle:
        return pkl.dump(params, handle)


def load_vqe_result(data_fname):
    with open(vqe_data_dir / (data_fname + '.pickle'), 'rb') as handle:
        return pkl.load(handle)
