from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer

from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_nature.algorithms import VQEUCCFactory
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver

from qiskit.providers.basicaer import StatevectorSimulatorPy  # local simulator

import numpy as np
from src import krylov_vqe as kvqe
import matplotlib.pyplot as plt
from pathlib import Path

# File structure
project_root = Path(__file__).parent
project_src = project_root / 'src'
vqe_data_dir = project_root / 'vqe_circuit_data'
mpl_styles_dir = project_root / 'mpl_styles'
fig_dir = project_root / 'figures'

# Set 'run_vqe' to true to try and generate a better VQE result
run_vqe = True
num_runs = 1

# Specify the molecule
# bond_distance = 2.5  # in Angstrom
bond_distance = 1.5949  # in Angstrom

# define molecule
molecule = Molecule(
    geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, bond_distance]]], charge=0, multiplicity=1
)

# specify driver
driver = ElectronicStructureMoleculeDriver(
    molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.AUTO
)
properties = driver.run()

particle_number = properties.get_property(ParticleNumber)

# specify active space transformation
active_space_trafo = ActiveSpaceTransformer(
    num_electrons=particle_number.num_particles, num_molecular_orbitals=3
)

# define electronic structure problem
problem = ElectronicStructureProblem(driver, transformers=[active_space_trafo])

# construct qubit converter (parity mapping + 2-qubit reduction)
qubit_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)

# Build the qubit Hamiltonian
hamiltonian = qubit_converter.convert(problem.second_q_ops()[0], particle_number.num_particles)

# Classical Solution
print('Starting ED calculation with NumPy solver...')
exact_solver = NumPyMinimumEigensolver()
exact_groundstate_solver = GroundStateEigensolver(qubit_converter, exact_solver)

exact_result = exact_groundstate_solver.solve(problem)

nuclear_repulsion_energy = exact_result.nuclear_repulsion_energy
exact_groundstate_energy = np.real(exact_result.eigenenergies[0]) + nuclear_repulsion_energy

print('Done')

# VQE (Doesn't need to run multiple times, the energy only varies on the scale 1e-15
print('Starting VQEUCCFactory solver...')
vqe_solver = VQEUCCFactory(quantum_instance=StatevectorSimulatorPy())
calc = GroundStateEigensolver(qubit_converter, vqe_solver)
vqe_result = calc.solve(problem)
vqe_groundstate_energy = vqe_result.raw_result.eigenvalue.real + nuclear_repulsion_energy

optimized_circuit = calc.solver.ansatz.assign_parameters(vqe_result.raw_result.optimal_point).decompose()

print('Done')

# %% Simple Krylov test using one particular combination of evenly distributed intermediate statevectors
# Krylov algorithm parameters
depth = optimized_circuit.depth()
num_states = depth - 2  # number of gates between statevector extractions

# Run Krylov
print('Starting Krylov VQE algorithm...')
krylov_solver = kvqe.KrylovVQESolver(hamiltonian, optimized_circuit)
result = krylov_solver.solve_krylov_vqe(method='even', num_states=num_states)
krylov_eigenvalue = krylov_solver.minimum_eigenvalue + nuclear_repulsion_energy
print('Done\n')

# Report results
print(f'Exact Groundstate Eigenvalue: {exact_groundstate_energy:>}')
print(f'VQE Groundstate Eigenvalue: {vqe_groundstate_energy:>}')
print(f'Krylov Groundstate Eigenvalue:{krylov_eigenvalue:>}\n')

vqe_error = 100 * np.abs((vqe_groundstate_energy - exact_groundstate_energy) / exact_groundstate_energy)
krylov_error = 100 * np.abs((krylov_eigenvalue - exact_groundstate_energy) / exact_groundstate_energy)
print(f'VQE Eigenvalue error:'
      f' {vqe_error:.2e}%')
print(f'Krylov Eigenvalue error: {krylov_error:.2g}%\n')

# %% Analyze statistic of intermediate state choices for some given num_states
# Krylov algorithm parameters
depth = optimized_circuit.depth()
num_states = depth - 3  # number of gates between statevector extractions

# Run Krylov
print('Starting Krylov VQE algorithm...')
krylov_solver = kvqe.KrylovVQESolver(hamiltonian, optimized_circuit)
result = krylov_solver.solve_krylov_vqe(method='all', num_states=num_states)
print('Done\n')

krylov_eigenvalue = krylov_solver.minimum_eigenvalue + nuclear_repulsion_energy
vals = result[0] + nuclear_repulsion_energy
errs = 100 * np.abs((vals - exact_groundstate_energy) / exact_groundstate_energy)
mean_err = np.mean(errs)
std = np.std(errs)

# Report results
print(f'Krylov Minimum Error:{min(errs):.2g}')
print(f'Krylov Average Error:{mean_err:.2g}')
print(f'Krylov Error Standard Deviation:{std:.2g}\n')

plt.style.use(mpl_styles_dir / 'line_plot.mplstyle')
fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(errs, len(errs) // 10)
ax.set_xlabel('Error (%)')
ax.set_ylabel('Counts')

plt.tight_layout()
plt.savefig(fig_dir / 'LiH_histogram.png')

# %%  Plot the error as a function of the number of Krylov states used
num_shots = 100

x = []
krylov_min_errors = []
krylov_mean_errors = []
krylov_stds = []
for ii in np.arange(1, depth + 2):
    result = krylov_solver.solve_krylov_vqe(method='random', num_states=ii, num_shots=num_shots)

    vals = result[0] + nuclear_repulsion_energy
    errs = 100 * np.abs((vals - exact_groundstate_energy) / exact_groundstate_energy)
    mean_err = np.mean(errs)
    std = np.std(errs)

    x.append(ii)
    krylov_min_errors.append(min(errs))
    krylov_mean_errors.append(mean_err)
    krylov_stds.append(std)

# %%
plt.style.use(mpl_styles_dir / 'line_plot.mplstyle')
fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(x, krylov_mean_errors, yerr=krylov_stds, fmt='k.', label='mean(err)')
ax.scatter(x, krylov_min_errors, c='r', marker='o', label='min(err)')
ax.legend()

ax.set_xlabel('# States')
ax.set_ylabel('Error (%)')

plt.tight_layout()
plt.savefig(fig_dir / 'LiH_error_vs_states.png')
