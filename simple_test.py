from qiskit.opflow import X, Y, Z
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import SPSA
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.algorithms import VQE, NumPyMinimumEigensolver

import numpy as np

from src import krylov_vqe as kvqe
from src import utils


# A simple function for running a VQE calculation for some Hamiltonian and some ansatz circuit
def vqe_calculator(hamiltonian, vqe_ansatz):
    optimizer = SPSA(maxiter=50)

    initial_point = np.random.random(vqe_ansatz.num_parameters)

    intermediate_info = {
        'nfev': [],
        'parameters': [],
        'energy': [],
        'stddev': []
    }

    def callback(nfev, parameters, energy, stddev):
        intermediate_info['nfev'].append(nfev)
        intermediate_info['parameters'].append(parameters)
        intermediate_info['energy'].append(energy)
        intermediate_info['stddev'].append(stddev)

    vqe = VQE(ansatz=vqe_ansatz,
              optimizer=optimizer,
              initial_point=initial_point,
              quantum_instance=QasmSimulatorPy(),
              callback=callback)

    result = vqe.compute_minimum_eigenvalue(hamiltonian)

    return result


# Set 'run_vqe' to true to try and generate a better VQE result
run_vqe = False
num_runs = 15

# Define some arbitrary test Hamiltonian
num_qubits = 3
ham = (Z ^ Z ^ Y) - (X ^ Y ^ Z)

# Find the exact solution for the grounstate eigenvalue
exact_solver = NumPyMinimumEigensolver()
exact_ground_eval = exact_solver.compute_minimum_eigenvalue(ham).eigenvalue

# VQE ansatz definition
ansatz = EfficientSU2(num_qubits, reps=2, entanglement='linear', insert_barriers=True)

# Run VQE to find a good optimized circuit
if run_vqe:
    print(f'Exact groundstate eigenvalue: {exact_ground_eval:.4f}')
    print('Starting VQE runs...')

    # Iterate VQE calculation to find a good optimized circuit
    for ii in range(num_runs):

        # Try to load old results for comparison
        if ii == 0:
            try:
                vqe_res = utils.load_vqe_result('simple_test_vqe_result')
                vqe_ground_eval = vqe_res.eigenvalue.real
            except FileNotFoundError:
                print("No previous VQE results found")
                vqe_ground_eval = np.abs(exact_ground_eval) + 100

        # Run a VQE calculations
        res = vqe_calculator(ham, ansatz)

        # Compare to old optimized circuit, overwrite old result
        if np.abs(res.eigenvalue - exact_ground_eval) < np.abs(vqe_ground_eval - exact_ground_eval):
            vqe_res = res
            vqe_ground_eval = vqe_res.eigenvalue.real

            print(f'Finished VQE run {ii + 1}/{num_runs}. Obtained groundstate eval'
                  f' {res.eigenvalue.real:.4f}. New best!', end='\r')

            utils.save_vqe_result(vqe_res, 'simple_test_vqe_result')
        else:
            print(f'Finished VQE run {ii + 1}/{num_runs}. Obtained groundstate eval'
                  f' {res.eigenvalue.real:.4f} ', end='\r')

    print('\n')

# Load best VQE result
res = utils.load_vqe_result('simple_test_vqe_result')

# Rebuild the best VQE circuit
optimized_circuit = ansatz.assign_parameters(res.optimal_point)

# Apply the Krylov method to the best VQE circuit
krylov_eigenvalue = kvqe.krylov_vqe(ham, optimized_circuit, depth_period=1)

# Report results
print(f'Exact Groundstate Eigenvalue: {exact_ground_eval}')
print(f'Best VQE Groundstate Eigenvalue: {res.eigenvalue.real}')
print(f'Krylov Groundstate Eigenvalue:{krylov_eigenvalue}\n')

print(f'Best VQE Eigenvalue error: {100 * np.abs((res.eigenvalue.real - exact_ground_eval) / exact_ground_eval):.4f}%')
print(f'Krylov Eigenvalue error: {100 * np.abs((krylov_eigenvalue - exact_ground_eval) / exact_ground_eval):.4f}')
