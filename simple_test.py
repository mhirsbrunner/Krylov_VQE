from qiskit.opflow import X, Y, Z
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import SPSA
from qiskit.providers.basicaer import StatevectorSimulatorPy
from qiskit.algorithms import VQE, NumPyMinimumEigensolver

import numpy as np
import os
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt

from src import krylov_vqe as kvqe

# File structure
project_root = Path(__file__).parent
project_src = project_root / 'src'
vqe_data_dir = project_root / 'vqe_circuit_data'
mpl_styles_dir = project_root / 'mpl_styles'
fig_dir = project_root / 'figures'

# Set 'run_vqe' to true to try and generate a better VQE result
run_vqe = True
num_runs = 20

# Define some arbitrary test Hamiltonian
num_qubits = 3
hamiltonian = (Z ^ Z ^ Y) - (X ^ Y ^ Z)

# Find the exact solution for the grounstate eigenvalue
exact_solver = NumPyMinimumEigensolver()
exact_groundstate_energy = exact_solver.compute_minimum_eigenvalue(hamiltonian).eigenvalue
print(f'Exact groundstate eigenvalue: {exact_groundstate_energy:.4f}')

# VQE ansatz definition
ansatz = EfficientSU2(num_qubits, reps=2, entanglement='linear')

# Build the VQE algorithm
optimizer = SPSA(maxiter=50)
initial_point = np.random.random(ansatz.num_parameters)

vqe = VQE(ansatz=ansatz,
          optimizer=optimizer,
          initial_point=initial_point,
          # quantum_instance=QasmSimulatorPy())
          quantum_instance=StatevectorSimulatorPy())

# Define data filenames
vqe_result_fname = 'simple_test_vqe_result.pickle'

# Check for existing results
if os.path.isfile(vqe_data_dir / vqe_result_fname):
    with open(vqe_data_dir / vqe_result_fname, 'rb') as handle:
        optimal_vqe_result = pkl.load(handle)
else:
    print("No saved results found, running an initial VQE calculation...")
    optimal_vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)
    with open(vqe_data_dir / vqe_result_fname, 'wb') as handle:
        pkl.dump(optimal_vqe_result, handle)

# Run VQE to find a good optimized circuit
if run_vqe:

    # Iterate VQE calculation to find a good optimized circuit
    print('Starting VQE runs...')
    for ii in range(num_runs):

        # Generate a new initial point for the VQE
        new_initial_point = np.random.random(ansatz.num_parameters)
        vqe.initial_point = new_initial_point

        # Calculate the minimum eigenvalue
        current_vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)

        print(f'Finished VQE run {ii + 1}/{num_runs}.', end="\r")

        current_diff = np.abs(current_vqe_result.eigenvalue.real - exact_groundstate_energy)
        optimal_diff = np.abs(optimal_vqe_result.eigenvalue.real - exact_groundstate_energy)

        if current_diff < optimal_diff:
            optimal_vqe_result = current_vqe_result

            with open(vqe_data_dir / vqe_result_fname, 'wb') as handle:
                pkl.dump(optimal_vqe_result, handle)

    print('\n')

# %%
# Rebuild the best VQE circuit
optimized_circuit = ansatz.assign_parameters(optimal_vqe_result.optimal_point).decompose()

# Apply the Krylov method to the best VQE circuit
depth = optimized_circuit.depth()
num_states = 1
krylov_solver = kvqe.KrylovVQESolver(hamiltonian, optimized_circuit)
krylov_eigenvalue = krylov_solver.solve_krylov_vqe(method='even', num_states=num_states)

# Report results
print(f'Exact Groundstate Eigenvalue: {exact_groundstate_energy}')
print(f'Best VQE Groundstate Eigenvalue: {optimal_vqe_result.eigenvalue.real}')
print(f'Krylov Groundstate Eigenvalue:{krylov_eigenvalue}\n')

optimal_diff = optimal_vqe_result.eigenvalue - exact_groundstate_energy
krylov_diff = krylov_eigenvalue - exact_groundstate_energy
print(
    f'Best VQE Eigenvalue error: {100 * np.abs(optimal_diff / exact_groundstate_energy):.4e}%')
print(
    f'Krylov Eigenvalue error: {100 * np.abs(krylov_diff / exact_groundstate_energy):.4e}%')

# %% Plot Krylov results for every possible combination of statevectors
num_states = np.arange(1, depth + 2)

krylov_eigenvalues = []
krylov_errors = []
krylov_stds = []

for ii in num_states:
    result = krylov_solver.solve_krylov_vqe(method='all', num_states=ii)
    vals = result[0]
    errs = 100 * np.abs((vals - exact_groundstate_energy) / exact_groundstate_energy)
    mean_err = np.mean(errs)
    std = np.std(errs)

    krylov_eigenvalues.append(vals)
    krylov_errors.append(mean_err)
    krylov_stds.append(std)

# %%
plt.style.use(mpl_styles_dir / 'line_plot.mplstyle')
fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(num_states, krylov_errors, yerr=krylov_stds, fmt='o')
ax.set_xlabel('# states')
ax.set_ylabel('error (%)')

plt.tight_layout()
plt.savefig(fig_dir / 'simple_test_statistics.png')
