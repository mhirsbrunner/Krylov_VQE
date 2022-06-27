from qiskit.circuit import QuantumCircuit
from qiskit.algorithms import MinimumEigensolverResult
from qiskit.quantum_info import Statevector
import numpy as np

import scipy.linalg as slg


def intermediate_vqe_states(optimized_circuit: QuantumCircuit, depth_period=1, decompositions=1):
    """
    Extracts the intermediate statevectors from an optimzed VQE circuit by rebuilding the circuit step by step and
    saving the Statevector at desired circuit depth increments
    :param optimized_circuit: The optimized circuit produced by VQE
    :param depth_period: The depth multiples at which the state vector is desired
    :param decompositions: The number of times to try decomposing the circuit into smaller blocks
    :return: A list of state vectors at each "barrier" instruction of the circuit
    """

    new_circuit = QuantumCircuit(optimized_circuit.num_qubits, optimized_circuit.num_clbits)

    for ii in range(decompositions):
        optimized_circuit = optimized_circuit.decompose()

    if depth_period > optimized_circuit.depth():
        raise ValueError(f'The {depth_period=} cannot be larger than the depth of the optimized VQE'
                         f' circuit={optimized_circuit.depth()}.')

    state_vectors = []
    last_measured_depth = 0

    for instruction in optimized_circuit.data:
        new_circuit.append(instruction[0], instruction[1], instruction[2])
        current_depth = new_circuit.depth()
        if current_depth - last_measured_depth >= depth_period:
            state_vectors.append(Statevector.from_instruction(new_circuit))
            last_measured_depth = current_depth

    # Add the final output of the VQE circuit to the list of statevectors
    state_vectors.append(Statevector.from_instruction(new_circuit))

    if Statevector.from_instruction(optimized_circuit) != Statevector.from_instruction(new_circuit):
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


def krylov_vqe(hamiltonian, optimized_circuit: QuantumCircuit, depth_period=1, decompositions=1):
    """
    Top-level function for solving a generalized eigenvalue problem generated from an optimized VQE circuit
    :param hamiltonian: The Hamiltonian for which the VQE circuit produces the groundstate
    :param optimized_circuit: The optimized circuit produced by VQE
    :param depth_period:
    :param decompositions: The number of times to try decomposing the circuit into smaller blocks
    :return: The groundstate eigenvalue of the generalized eigenvalue problem
    """
    state_vectors = intermediate_vqe_states(optimized_circuit, depth_period, decompositions)
    evals = generalized_eigenvalue_solver(hamiltonian, state_vectors)

    return min(evals)
