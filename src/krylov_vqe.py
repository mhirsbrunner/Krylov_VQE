from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

import numpy as np
import scipy.linalg as slg
import random
from math import comb
import itertools
import matplotlib.pyplot as plt
from pathlib import Path

# File structure
project_src = Path(__file__).parent
project_root = project_src.parent
vqe_data_dir = project_root / 'vqe_circuit_data'
mpl_styles_dir = project_root / 'mpl_styles'
fig_dir = project_root / 'figures'


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
        self.num_qubits = hamiltonian.num_qubits
        self.hamiltonian = hamiltonian.primitive.to_matrix()
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

    def solve_krylov_vqe(self, target_depths=None, method=None, num_states=None, num_shots=None, verbose=False):
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
            elif num_states > self.circuit_depth:
                raise ValueError("Cannot specify more states than the depth of the circuit")

            elif any((method.lower() == x for x in ('even', 'front_loaded', 'back_loaded'))):

                if method.lower() == 'even':
                    target_depths = list((int(x) for x in np.floor(np.linspace(0, self.circuit_depth,
                                                                               num_states + 1))[1:-1]))

                elif method.lower() == 'front_loaded':
                    target_depths = np.arange(1, num_states)

                elif method.lower() == 'back_loaded':
                    target_depths = np.arange(self.circuit_depth - num_states + 1, self.circuit_depth)

                state_vectors = self.generate_intermediate_states(target_depths)
                self.eigenvalues = self.generalized_eigensolver(state_vectors)
                self.minimum_eigenvalue = min(self.eigenvalues)
                return self.minimum_eigenvalue

            elif any((method.lower() == x for x in ('all_permutations', 'all', 'random'))):
                num_combinations = comb(self.circuit_depth - 1, num_states - 1)

                if method.lower() == 'random':
                    if num_shots is None:
                        raise ValueError(f"Must specify 'num_shots' if using the '{method}' method.")
                    if num_shots > num_combinations:
                        if verbose:
                            print("Parameter 'num_shots' is greater than the total number of possible combinations, "
                              "reducing 'num_shots' to the total number of combinations.")
                        num_shots = num_combinations
                    verb = f'Calculating Krylov VQE eigenvalue for {num_shots} random combinations...'
                    if verbose:
                        print(verb, end='\r')

                    def random_combination(iterable, r):
                        pool = tuple(iterable)
                        n = len(pool)
                        indices = sorted(random.sample(range(n), r))
                        return tuple(pool[i] for i in indices)

                    combinations = []
                    while len(combinations) < num_shots:
                        shot = random_combination(range(1, self.circuit_depth), num_states - 1)
                        if not (shot in combinations):
                            combinations.append(shot)
                else:
                    num_shots = num_combinations
                    verb = f'Calculating Krylov VQE eigenvalue for all {num_combinations} combinations...'
                    if verbose:
                        print(verb, end='\r')
                    combinations = list(itertools.combinations(list(range(1, self.circuit_depth)), num_states - 1))

                min_evals = []
                best_permutation = None

                for ii, p in enumerate(combinations):
                    state_vectors = self.generate_intermediate_states(p)
                    evals = self.generalized_eigensolver(state_vectors)
                    min_evals.append(min(evals))

                    if self.minimum_eigenvalue is None:
                        self.minimum_eigenvalue = min(evals)
                        best_permutation = p
                    elif min(evals) < self.minimum_eigenvalue:
                        self.minimum_eigenvalue = min(evals)
                        best_permutation = p

                    if verbose:
                        print(verb + f' Finished permutation {ii + 1}/{num_shots}.', end="\r")
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
        if any(((depth < 1 or depth > self.circuit_depth) for depth in target_depths)):
            raise ValueError("No element of 'target_depths' can be larger than the circuit depth or less than one")

        if any((depth == self.circuit_depth for depth in target_depths)):
            print("The final output of the VQE circuit is always included in the list of intermediate statevectors,"
                  "there is no need to include it explicitly in 'target_depths'.")

        new_circuit = QuantumCircuit(self.optimized_circuit.num_qubits, self.optimized_circuit.num_clbits)

        n = len(target_depths) + 1
        state_vectors = np.zeros((2 ** self.num_qubits, n), dtype=complex)
        ii = 0

        for instruction in self.optimized_circuit.data:
            prev_circuit = new_circuit.copy()
            new_circuit.append(instruction[0], instruction[1], instruction[2])

            prev_depth = prev_circuit.depth()
            current_depth = new_circuit.depth()

            if current_depth > prev_depth and prev_depth in target_depths:
                state_vectors[:, ii] = Statevector.from_instruction(prev_circuit).data
                ii += 1

        # Add the final output of the VQE circuit to the list of statevectors
        state_vectors[:, ii] = Statevector.from_instruction(new_circuit).data

        if (state_vectors[:, -1] != Statevector.from_instruction(self.optimized_circuit).data).all():
            raise RuntimeError('Final extracted statevector does not match the VQE output, something went wrong')

        return state_vectors

    def generalized_eigensolver(self, state_vectors):
        """
        Solves the generalized eigenvalue problem of the hamiltonian projected into a finite subset of the Hilbert
        space. The generalized eigenvalue problem is of the form H.x = ES.x, where H is the hamiltonian projected into
        the provided state vectors, S is the overlap matrix of the state vectors, and E is the eigenvalue.
        :param state_vectors: A list of intermediate state vectors obtained from the VQE-optimized circuit
        :return: The eigenvalues obtained from the generalized eigenvalue problem
        """

        h_elems = state_vectors.conj().T @ self.hamiltonian @ state_vectors
        s_elems = state_vectors.conj().T @ state_vectors

        # Remove small and negative eigenvalues of the overlap matrix
        u, s, vh = slg.svd(s_elems)
        defective_inds = []

        for ii, eigval in enumerate(s):
            if np.abs(eigval) <= self.tolerance:
                defective_inds.append(ii)
            elif eigval < 0:
                print(f'Overlap matrix contains a large negative eigenvalue = {u:.4g}. Proceed with caution.')

        svd_h = slg.inv(u) @ h_elems @ slg.inv(vh)
        s = np.diag(s)

        svd_h = np.delete(np.delete(svd_h, defective_inds, 0), defective_inds, 1)
        u = np.delete(np.delete(u, defective_inds, 0), defective_inds, 1)
        s = np.delete(np.delete(s, defective_inds, 0), defective_inds, 1)
        vh = np.delete(np.delete(vh, defective_inds, 0), defective_inds, 1)

        h_elems = u @ svd_h @ vh
        s_elems = u @ s @ vh

        # Solve generalized eigenvalue problem with the now positive definite overlap matrix
        self.eigenvalues = slg.eigh(h_elems, s_elems, eigvals_only=True)
        self.minimum_eigenvalue = min(self.eigenvalues)

        return self.eigenvalues

    def plot_histogram(self, exact_groundstate_energy: float, num_states: int, method: str, num_shots=None,
                       num_bins=None, fig_fname='histogram'):
        result = self.solve_krylov_vqe(method=method, num_states=num_states, num_shots=num_shots)

        krylov_groundstate_energy = self.minimum_eigenvalue
        vals = result[0]
        abs_errs = np.abs(vals - exact_groundstate_energy)
        mean_err = np.mean(abs_errs)
        std = np.std(abs_errs)

        # Report results
        print(f'Krylov Minimum Error:{min(abs_errs):.2g}')
        print(f'Krylov Average Error:{mean_err:.2g}')
        print(f'Krylov Error Standard Deviation:{std:.2g}\n')

        plt.style.use(mpl_styles_dir / 'line_plot.mplstyle')
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.ticklabel_format(style='sci')

        if num_bins is None:
            num_bins = len(abs_errs) // 10

        ax.hist(abs_errs, bins=num_bins)
        ax.set_xlabel('Error (eV)')
        ax.set_ylabel('Counts')

        plt.tight_layout()
        plt.savefig(fig_dir / (fig_fname + '.pdf'))
        plt.savefig(fig_dir / (fig_fname + '.png'))

    def plot_error_vs_num_states(self, exact_groundstate_energy: float, num_shots: int, max_num_states: int,
                                 plot_chem_acc=False, log_thresh=None, fig_fname='error_vs_states'):
        x = []

        even_errors = []
        front_errors = []
        back_errors = []

        krylov_min_errors = []

        krylov_mean_errors = []
        krylov_stds = []

        for ii in np.arange(1, max_num_states + 1):
            print(f"Sampling over {ii}/{max_num_states} states...", end='\r')

            # Calculate error using even, front-loaded, and back-loaded state distributions
            even_errors.append(np.abs(exact_groundstate_energy
                                      - self.solve_krylov_vqe(method='even', num_states=ii)))
            front_errors.append(np.abs(exact_groundstate_energy
                                       - self.solve_krylov_vqe(method='front_loaded', num_states=ii)))
            back_errors.append(np.abs(exact_groundstate_energy
                                      - self.solve_krylov_vqe(method='back_loaded', num_states=ii)))

            # Calculate error using random state distributions
            result = self.solve_krylov_vqe(method='random', num_states=ii, num_shots=num_shots)

            vals = result[0]
            abs_errs = np.abs(vals - exact_groundstate_energy)
            mean_err = np.mean(abs_errs)
            std = np.std(abs_errs)

            x.append(ii)
            krylov_min_errors.append(min(abs_errs))
            krylov_mean_errors.append(mean_err)
            krylov_stds.append(std)

        print(f"Sampling over {max_num_states}/{max_num_states} states... Done.")

        plt.style.use(mpl_styles_dir / 'line_plot.mplstyle')
        fig, ax = plt.subplots(figsize=(9, 6))

        if plot_chem_acc:
            chem_acc = 43.5e-3
            ax.plot(x, chem_acc * np.ones_like(x), 'k--', label='Chem. Acc.')

        ax.plot(x, krylov_min_errors[0] * np.ones_like(x), 'b--', label='VQE err')

        ax.errorbar(x, krylov_mean_errors, yerr=krylov_stds, fmt='ko', label='mean(err)', zorder=0, markersize=6)

        ax.plot(x, krylov_min_errors, 'ro', label='min(err)')
        ax.plot(x, back_errors, 'ms', label='back loaded')
        ax.plot(x, front_errors, 'c^', label='front loaded')
        ax.plot(x, even_errors, 'gP', label='even')

        ax.legend()

        ax.set_xlabel('# States')
        ax.set_ylabel('Error (eV)')

        if log_thresh is None:
            ax.set_yscale('log')
        else:
            ax.set_yscale('symlog', linthresh=log_thresh)

        plt.tight_layout()
        plt.savefig(fig_dir / (fig_fname + '.pdf'))
        plt.savefig(fig_dir / (fig_fname + '.png'))
