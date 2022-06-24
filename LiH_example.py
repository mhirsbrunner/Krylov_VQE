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

import numpy as np

from qiskit.providers.basicaer import QasmSimulatorPy  # local simulator

from src import krylov_vqe as kvqe

# Specify the molecule
bond_distance = 2.5  # in Angstrom

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
qubit_op = qubit_converter.convert(problem.second_q_ops()[0], particle_number.num_particles)

# Classical Solution
print('Starting ED calculation with NumPy solver...')
np_solver = NumPyMinimumEigensolver()
np_groundstate_solver = GroundStateEigensolver(qubit_converter, np_solver)

np_result = np_groundstate_solver.solve(problem)

nuclear_repulsion_energy = np_result.nuclear_repulsion_energy

target_energy = np.real(np_result.eigenenergies[0]) + nuclear_repulsion_energy
print('Done')
print("Classical Solution Energy:", target_energy)

# VQE
print('Starting VQE calculation with NumPy solver...')
vqe_solver = VQEUCCFactory(quantum_instance=QasmSimulatorPy())
calc = GroundStateEigensolver(qubit_converter, vqe_solver)

res = calc.solve(problem)

print(f'VQEUCC Energy: {res.hartree_fock_energy}')

raw_res = res.raw_result
krylov_vqe_eigenvalue = kvqe.krylov_vqe(qubit_op, raw_res, calc.solver.ansatz)
print(f'Kylov VQE Eigenvalue:{krylov_vqe_eigenvalue + nuclear_repulsion_energy}')
