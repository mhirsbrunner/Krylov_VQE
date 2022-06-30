from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

import numpy as np
import scipy.linalg as slg
import random


def intermediate_vqe_states(optimized_circuit: QuantumCircuit, num_states=2, decompositions=1, method='even'):
    """
    Extracts the intermediate statevectors from an optimzed VQE circuit by rebuilding the circuit step by step and
    saving the Statevector at desired circuit depth increments. Importantly, this assumes the input state is all zeros
    :param optimized_circuit: The optimized circuit produced by VQE
    :param num_states: The number of intermediate states desired
    :param decompositions: The number of times to try decomposing the circuit into smaller blocks
    :param method: The desired method for distributing the statevector samples among the gates
    :return: A list of state vectors at each "barrier" instruction of the circuit
    """
    if num_states == 1:
        return [Statevector.from_instruction(optimized_circuit), ]
    elif num_states < 1:
        raise ValueError("Parameter 'num_states' must be a positive integer")
    else:

        new_circuit = QuantumCircuit(optimized_circuit.num_qubits, optimized_circuit.num_clbits)

        for ii in range(decompositions):
            optimized_circuit = optimized_circuit.decompose()

        depth = optimized_circuit.depth()

        if num_states > depth + 1:
            raise ValueError(f'The number of desired intermediate {num_states=} cannot be larger than the one plus the '
                             f'depth of the optimized VQE circuit={optimized_circuit.depth()}.')

        state_vectors = []

        # Distribute the intermediate states evenly, always including the last state
        if method.lower() == 'even':
            target_depths = [int(x) for x in np.floor(np.linspace(0, depth, num_states + 1))[1:-1]]
        elif method.lower() == 'random':
            target_depths = random.sample(list(range(0, depth)), num_states - 1)
        else:
            raise ValueError("Parameter 'method' must be either 'even' or 'random'.")

        for instruction in optimized_circuit.data:
            prev_circuit = new_circuit.copy()
            new_circuit.append(instruction[0], instruction[1], instruction[2])

            prev_depth = prev_circuit.depth()
            current_depth = new_circuit.depth()

            if current_depth > prev_depth and prev_depth in target_depths:
                state_vectors.append(Statevector.from_instruction(prev_circuit))

        # Add the final output of the VQE circuit to the list of statevectors
        state_vectors.append(Statevector.from_instruction(new_circuit))

        if Statevector.from_instruction(optimized_circuit) != state_vectors[-1]:
            print("Something went wrong, the final statevectors don't match.")

        return state_vectors


def generalized_eigenvalue_solver(hamiltonian, state_vectors, tol=1e-14):
    """
    Solves the generalized eigenvalue problem of the hamiltonian projected into a finite subset of the Hilbert space.
    The generalized eigenvalue problem is of the form H.x = ES.x, where H is the hamiltonian projected into the provided
    state vectors, S is the overlap matrix of the state vectors, and E is the eigenvalue.
    :param hamiltonian: Hamiltonian operator used in the VQE calculation
    :param state_vectors: A list of intermediate state vectors obtained from the VQE-optimized circuit
    :param tol: The accepted threshold for small eigenvalues of the overlap matrix. Anything smaller is set equal to tol
    to avoid non-positive definite overlap matrices.
    :return: The eigenvalues obtained from the generalized eigenvalue problem
    """
    n = len(state_vectors)

    h_elems = np.zeros((n, n), dtype=complex)
    s_elems = np.zeros((n, n), dtype=complex)

    for ii in range(n):
        for jj in range(n):
            h_elems[ii, jj] = state_vectors[ii].inner(hamiltonian.eval(state_vectors[jj]).primitive)
            s_elems[ii, jj] = state_vectors[ii].inner(state_vectors[jj])

    # Ensuring the overlap matrix is positive-definite
    su, sv = slg.eigh(a=s_elems)

    for ii, u in enumerate(su):
        if np.abs(u) <= tol:
            su[ii] = tol
        elif u < 0:
            raise RuntimeError(f'Overlap matrix contains a large negative eigenvalue, cannot proceed.'
                               f' Eigenvalue: {u}')

    s_elems = sv @ np.diag(su) @ sv.conj().T

    evals = slg.eigh(h_elems, s_elems, eigvals_only=True)
    return evals


def krylov_vqe(hamiltonian, optimized_circuit: QuantumCircuit, num_states=1, decompositions=1, method='even'):
    """
    Top-level function for solving a generalized eigenvalue problem generated from an optimized VQE circuit
    :param hamiltonian: The Hamiltonian for which the VQE circuit produces the groundstate
    :param optimized_circuit: The optimized circuit produced by VQE
    :param num_states: The desired number of intermediate states to use
    :param decompositions: The number of times to try decomposing the circuit into smaller blocks
    :param method: Method for determining placement of intermediate states
    :return: The groundstate eigenvalue of the generalized eigenvalue problem
    """
    state_vectors = intermediate_vqe_states(optimized_circuit, num_states, decompositions, method)
    evals = generalized_eigenvalue_solver(hamiltonian, state_vectors)

    return min(evals.real)
