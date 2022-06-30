from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

import numpy as np
import scipy.linalg as slg
import random
import itertools


class KrylovVQESolver:
    def __init__(self, hamiltonian, optimized_circuit):
        """
        This class accepts a Hamiltonian and a circuit optimized by a VQE algorithm to produce the groundstate of the
        Hamiltonian. By sampling intermediate statevectors obtained from the circuit and solving a generalized
        eigenvalue problem in this larger subspace, this class improves the accuracy of VQE groundstate eigenvalue
        estimations.
        :param hamiltonian: The Hamiltonian under investigation
        :param optimized_circuit: The circuit optimized by a VQE algorithm to produce the groundstate of the Hamiltonian
        """
        self.hamiltonian = hamiltonian
        self.optimized_circuit = optimized_circuit
        self.circuit_depth = self.optimized_circuit.depth()

        self.intermediate_states = None

        self.tolerance = 1e-14
        self.h_elems = None
        self.s_elems = None
        self.eigenvalues = None
        self.minimum_eigenvalue = None

        if self.circuit_depth == 0:
            raise ValueError("Parameter 'optimized_circuit' must be a QuantumCircuit"
                             " object of depth greater than zero.")

    def solve_krylov_vqe(self, target_depths=None, method=None, num_states=None, num_shots=None):
        if method is None and target_depths is None:
            raise ValueError("Either 'method' or 'target_depths' must be specified to generate the states used for "
                             "the Krylov VQE solution.")

        elif method is not None and target_depths is not None:
            raise ValueError("Both 'method' and 'target_depths' cannot be specified simultaneously.")

        elif target_depths is not None:
            state_vectors = self.generate_intermediate_states(target_depths)
            self.eigenvalues = self.generalized_eigensolver(state_vectors)
            self.minimum_eigenvalue = min(self.eigenvalues)
            return self.minimum_eigenvalue

        else:
            if num_states is None:
                raise ValueError("Need to specifiy parameter 'num_states' when 'method' is specified.")

            if method.lower() == 'even':
                target_depths = list((int(x) for x in np.floor(np.linspace(0, self.circuit_depth,
                                                                           num_states + 1))[1:-1]))
                state_vectors = self.generate_intermediate_states(target_depths)
                self.eigenvalues = self.generalized_eigensolver(state_vectors)
                self.minimum_eigenvalue = min(self.eigenvalues)
                return self.minimum_eigenvalue

            elif any((method.lower() == x for x in ('all_permutations', 'all', 'random'))):
                combinations = list(itertools.combinations(list(range(self.circuit_depth)), num_states - 1))

                if method.lower() == 'random':
                    if num_shots is None:
                        raise ValueError(f"Must specify 'num_shots' if using the '{method}' method.")
                    if num_shots > len(combinations):
                        print("Parameter 'num_shots' is greater than the total number of possible combinations, "
                              "reducing 'num_shots' to the total number of combinations.")
                        num_shots = len(combinations)
                    print(f'Calculating Krylov VQE eigenvalue for {num_shots} random combinations...')
                else:
                    num_shots = len(combinations)
                    print(f'Calculating Krylov VQE eigenvalue for all {len(combinations)} combinations...')

                min_evals = []
                best_permutation = None

                for ii, p in enumerate(random.sample(combinations, num_shots)):
                    state_vectors = self.generate_intermediate_states(p)
                    evals = self.generalized_eigensolver(state_vectors)
                    min_evals.append(min(evals))

                    if self.minimum_eigenvalue is None:
                        self.minimum_eigenvalue = min(evals)
                        best_permutation = p
                    elif min(evals) < self.minimum_eigenvalue:
                        self.minimum_eigenvalue = min(evals)
                        best_permutation = p

                    print(f'Finished permutation {ii + 1}/{num_shots}. New minimum eval: {min(evals):.8f}'
                          f' Best minimum eval: {self.minimum_eigenvalue:.8f}', end="\r")
                print('\n')
                return min_evals, best_permutation

    def generate_intermediate_states(self, target_depths):
        """
        Extracts the intermediate statevectors from an optimzed VQE circuit by rebuilding the circuit step by step and
        saving the Statevector at desired circuit depth increments.
        Importantly, this method assumes the input state is all zeros.
        :param target_depths: A list of integers denoting the circuit depths at which the intermediate statevectors
        should be extracted
        :return: A list of state vectors at each desired circuit depth
        """
        if any(((depth < 0 or depth > self.circuit_depth) for depth in target_depths)):
            raise ValueError("No element of 'target_depths' can be larger than the circuit depth or less than zero")

        if any((depth == self.circuit_depth for depth in target_depths)):
            print("The final output of the VQE circuit is always included in the list of intermediate statevectors,"
                  "there is no need to include it explicitly in 'target_depths'.")

        new_circuit = QuantumCircuit(self.optimized_circuit.num_qubits, self.optimized_circuit.num_clbits)

        state_vectors = []

        for instruction in self.optimized_circuit.data:
            prev_circuit = new_circuit.copy()
            new_circuit.append(instruction[0], instruction[1], instruction[2])

            prev_depth = prev_circuit.depth()
            current_depth = new_circuit.depth()

            if current_depth > prev_depth and prev_depth in target_depths:
                state_vectors.append(Statevector.from_instruction(prev_circuit))

        # Add the final output of the VQE circuit to the list of statevectors
        state_vectors.append(Statevector.from_instruction(new_circuit))

        return state_vectors

    def generalized_eigensolver(self, state_vectors):
        """
        Solves the generalized eigenvalue problem of the hamiltonian projected into a finite subset of the Hilbert
        space. The generalized eigenvalue problem is of the form H.x = ES.x, where H is the hamiltonian projected into
        the provided state vectors, S is the overlap matrix of the state vectors, and E is the eigenvalue.
        :param state_vectors: A list of intermediate state vectors obtained from the VQE-optimized circuit
        :return: The eigenvalues obtained from the generalized eigenvalue problem
        """
        n = len(state_vectors)

        self.h_elems = np.zeros((n, n), dtype=complex)
        self.s_elems = np.zeros((n, n), dtype=complex)

        for ii in range(n):
            for jj in range(n):
                self.h_elems[ii, jj] = state_vectors[ii].inner(self.hamiltonian.eval(state_vectors[jj]).primitive)
                self.s_elems[ii, jj] = state_vectors[ii].inner(state_vectors[jj])

        # Ensuring the overlap matrix is positive-definite
        su, sv = slg.eigh(a=self.s_elems)

        for ii, u in enumerate(su):
            if np.abs(u) <= self.tolerance:
                su[ii] = self.tolerance
            elif u < 0:
                raise RuntimeError(f'Overlap matrix contains a large negative eigenvalue, cannot proceed.'
                                   f' Eigenvalue: {u}')

        self.s_elems = sv @ np.diag(su) @ sv.conj().T

        evals = slg.eigh(self.h_elems, self.s_elems, eigvals_only=True)
        return evals
