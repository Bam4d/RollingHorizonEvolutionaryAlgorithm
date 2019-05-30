import numpy as np
import logging

class RollingHorizonEvolutionaryAlgorithm():

    def __init__(self, rollout_actions_length, environment, mutation_probability, num_evals, use_shift_buffer=True,
                 flip_at_least_one=True, discount_factor=None, ignore_frames=0):

        self._logger = logging.getLogger('RHEA')

        self._rollout_actions_length = rollout_actions_length
        self._environment = environment
        self._use_shift_buffer = use_shift_buffer
        self._flip_at_least_one = flip_at_least_one
        self._mutation_probability = mutation_probability
        self._discount_factor = discount_factor
        self._num_evals = num_evals
        self._ignore_frames = ignore_frames

        # Initialize the solution to a random sequence
        if self._use_shift_buffer:
            self._solution = self._random_solution()

    def _get_next_action(self):
        """
        Get the next best action by evaluating a bunch of mutated solutions
        """

        if self._use_shift_buffer:
            solution = self._shift_and_append(self._solution)
        else:
            solution = self._random_solution()

        candidate_solutions = self._mutate(solution, self._mutation_probability)

        mutated_scores = self._environment.evaluate_rollout(candidate_solutions, self._discount_factor,
                                                            self._ignore_frames)
        best_idx = np.argmax(mutated_scores, axis=0)

        best_score_in_evaluations = mutated_scores[best_idx]

        self._solution = candidate_solutions[best_idx]

        self._logger.info('Best score in evaluations: %.2f' % best_score_in_evaluations)

        # The next best action is the first action from the solution space
        return self._solution[0]

    def _shift_and_append(self, solution):
        """
        Remove the first element and add a random action on the end
        """
        new_solution = np.copy(solution[1:])
        new_solution = np.vstack([new_solution , self._environment.get_random_action()])
        return new_solution

    def _random_solution(self):
        """
        Create a random set fo actions
        """
        return np.array([self._environment.get_random_action() for _ in range(self._rollout_actions_length)])

    def _mutate(self, solution, mutation_probability):
        """
        Mutate the solution
        """

        candidate_solutions = []
        # Solution here is 2D of rollout_actions x batch_size
        for b in range(self._num_evals):
            # Create a set of indexes in the solution that we are going to mutate
            mutation_indexes = set()
            solution_length = len(solution)
            if self._flip_at_least_one:
                mutation_indexes.add(np.random.randint(solution_length))

            mutation_indexes = mutation_indexes.union(
                set(np.where(np.random.random([solution_length]) < mutation_probability)[0]))

            # Create the number of mutations that is the same as the number of mutation indexes
            num_mutations = len(mutation_indexes)
            mutations = [self._environment.get_random_action() for _ in range(num_mutations)]

            # Replace values in the solutions with mutated values
            new_solution = np.copy(solution)
            new_solution[list(mutation_indexes)] = mutations
            candidate_solutions.append(new_solution)

        return np.stack(candidate_solutions)

    def run(self):

        while not self._environment.is_game_over():
            action = self._get_next_action()
            self._environment.perform_action(action)

            for _ in range(self._ignore_frames):
                self._environment.perform_action(action)


        score = self._environment.get_current_score()
        self._logger.info('Final score: %.2f' % (score))

