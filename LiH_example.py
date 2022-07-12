from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer

from qiskit_nature.algorithms import VQEUCCFactory
from qiskit_nature.circuit.library import UCCSD, HartreeFock
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.circuit.library import EfficientSU2, ExcitationPreserving


from qiskit.providers.basicaer import StatevectorSimulatorPy  # local simulator
from qiskit.providers.aer import StatevectorSimulator

import numpy as np
from src import krylov_vqe as kvqe
import matplotlib.pyplot as plt
from pathlib import Path

# %% File structure
project_root = Path(__file__).parent
project_src = project_root / 'src'
vqe_data_dir = project_root / 'vqe_circuit_data'
mpl_styles_dir = project_root / 'mpl_styles'
fig_dir = project_root / 'figures'

# %% Constants
chem_acc = 43.5e-3

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
# driver = ElectronicStructureMoleculeDriver(
#     molecule, basis="321g", driver_type=ElectronicStructureDriverType.AUTO
# )
properties = driver.run()

particle_number = properties.get_property(ParticleNumber)

# specify active space transformation
active_space_trafo = ActiveSpaceTransformer(
    num_electrons=particle_number.num_particles, num_molecular_orbitals=3
)

# define electronic structure problem
problem = ElectronicStructureProblem(driver, transformers=[active_space_trafo])
# problem = ElectronicStructureProblem(driver)

# construct qubit converter (parity mapping + 2-qubit reduction)
qubit_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
# qubit_converter = QubitConverter(JordanWignerMapper())

# Build the qubit Hamiltonian
hamiltonian = qubit_converter.convert(problem.second_q_ops()[0], particle_number.num_particles)

# %% Exact Solution
print('Starting ED calculation with NumPy solver...')
exact_solver = NumPyMinimumEigensolver()
exact_groundstate_solver = GroundStateEigensolver(qubit_converter, exact_solver)

exact_result = exact_groundstate_solver.solve(problem)

nuclear_repulsion_energy = exact_result.nuclear_repulsion_energy
exact_groundstate_energy = np.real(exact_result.eigenenergies[0])

print('Done')

# %%
print('Starting VQE Calculation...')
# Define the desired ansatz
initial_state = HartreeFock(problem.num_spin_orbitals, particle_number.num_particles, qubit_converter)

# ansatz = ExcitationPreserving(hamiltonian.num_qubits, reps=1, initial_state=initial_state)
# ansatz = EfficientSU2(reps=1, initial_state=initial_state)
ansatz = UCCSD(qubit_converter=qubit_converter, num_particles=particle_number.num_particles,
               num_spin_orbitals=problem.num_spin_orbitals, initial_state=initial_state)

backend = StatevectorSimulatorPy()

solver = VQE(ansatz=ansatz, quantum_instance=backend)

calc = GroundStateEigensolver(qubit_converter, solver)

vqe_result = calc.solve(problem)

vqe_groundstate_energy = vqe_result.groundenergy
vqe_abs_error = np.abs(vqe_groundstate_energy - exact_groundstate_energy)

optimized_circuit = ansatz.assign_parameters(vqe_result.raw_result.optimal_point)
optimized_circuit = optimized_circuit.decompose()

print('Done')
print(f'Exact Groundstate Eigenvalue: {exact_groundstate_energy + nuclear_repulsion_energy:>}')
print(f'VQE Absolute Error: {vqe_abs_error:.2e}')

# %% Simple Krylov test using one particular combination of evenly distributed intermediate statevectors
# Krylov algorithm parameters
depth = optimized_circuit.depth()
num_states = 7  # number of gates between statevector extractions

# Run Krylov
print('Starting Krylov VQE algorithm...')
krylov_solver = kvqe.KrylovVQESolver(hamiltonian, optimized_circuit)
result = krylov_solver.solve_krylov_vqe(method='even', num_states=num_states)
krylov_groundstate_energy = krylov_solver.minimum_eigenvalue
krylov_abs_error = np.abs(krylov_groundstate_energy - exact_groundstate_energy)
print('Done\n')

# Report results
print(f'VQE Absolute Error: {vqe_abs_error:.2e}')
print(f'Krylov Absolute Error:{krylov_abs_error:.2e}\n')

# %% Plot a histogram
krylov_solver.plot_histogram(exact_groundstate_energy, 4, 'random', num_shots=200, fig_fname='LiH_histogram')

# %%  Plot the error as a function of the number of Krylov states used
krylov_solver.plot_error_vs_num_states(exact_groundstate_energy, num_shots=100, max_num_states=8, log_thresh=1e-8,
                                       fig_fname='LiH_error_vs_num_states')
