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

from qiskit.providers.basicaer import QasmSimulatorPy  # local simulator

import numpy as np

import os
import pickle as pkl

from src import krylov_vqe as kvqe

# Set 'run_vqe' to true to try and generate a better VQE result
run_vqe = False
num_runs = 1

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

exact_ground_eval = np.real(np_result.eigenenergies[0]) + nuclear_repulsion_energy
print('Done')
print("Classical Solution Energy:", exact_ground_eval)

# VQE
result_fname = 'vqe_circuit_data/LiH_data.pickle'

if run_vqe:
    print('Starting VQE runs...')
    vqe_solver = VQEUCCFactory(quantum_instance=QasmSimulatorPy())
    calc = GroundStateEigensolver(qubit_converter, vqe_solver)

    # Iterate VQE calculation to find a good optimized circuit
    for ii in range(num_runs):

        # Try to load old results for comparison
        if ii == 0:
            if os.path.isfile(result_fname):
                with open(result_fname, 'rb') as handle:
                    vqe_res, optimized_circuit = pkl.load(handle)

                vqe_ground_eval = vqe_res.hartree_fock_energy
            else:
                print("No previous VQE results found")
                vqe_ground_eval = np.abs(exact_ground_eval) + 100

        # Run a VQE calculations
        res = calc.solve(problem)

        # Compare to old optimized circuit, overwrite old result
        if np.abs(res.hartree_fock_energy - exact_ground_eval) < np.abs(vqe_ground_eval - exact_ground_eval):
            vqe_res = res
            vqe_ground_eval = vqe_res.hartree_fock_energy

            print(f'Finished VQE run {ii + 1}/{num_runs}. Obtained groundstate eval'
                  f' {res.hartree_fock_energy.real:.4f}. New best!', end='\r')

            optimal_params = vqe_res.raw_result.optimal_point
            optimized_circuit = calc.solver.ansatz.assign_parameters(optimal_params)

            with open(result_fname, 'wb') as handle:
                pkl.dump([vqe_res, optimized_circuit], handle)

        else:
            print(f'Finished VQE run {ii + 1}/{num_runs}. Obtained groundstate eval'
                  f' {res.hartree_fock_energy.real:.4f} ', end='\r')

    print('\n')

with open(result_fname, 'rb') as handle:
    res, optimized_circuit = pkl.load(handle)

print(f'VQEUCC Energy: {res.hartree_fock_energy}')

krylov_eigenvalue = kvqe.krylov_vqe(qubit_op, optimized_circuit, depth_period=2, decompositions=2)\
                    + nuclear_repulsion_energy

# Report results
print(f'Exact Groundstate Eigenvalue: {exact_ground_eval}')
print(f'Best VQE Groundstate Eigenvalue: {res.hartree_fock_energy}')
print(f'Krylov Groundstate Eigenvalue:{krylov_eigenvalue}\n')

print(f'Best VQE Eigenvalue error:'
      f' {100 * np.abs((res.hartree_fock_energy - exact_ground_eval) / exact_ground_eval):.4f}%')
print(f'Krylov Eigenvalue error: {100 * np.abs((krylov_eigenvalue - exact_ground_eval) / exact_ground_eval):.4f}%')
