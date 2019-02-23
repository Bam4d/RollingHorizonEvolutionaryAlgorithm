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

        best_score_in_evaluations = float("-inf")

        if self._use_shift_buffer:
            solution = self._shift_and_append(self._solution)
        else:
            solution = self._random_solution()

        rollout_scores = []

        for i in range(self._num_evals):

            # Keep the best solution from previous iterations
            if i == 0:
                mutated_solution = solution
            else:
                mutated_solution = self._mutate(solution, self._mutation_probability)

            mutated_score = self._environment.evaluate_rollout(mutated_solution, self._discount_factor, self._ignore_frames)
            if mutated_score > best_score_in_evaluations:
                solution = mutated_solution
                best_score_in_evaluations = mutated_score

            rollout_scores.append(mutated_score)

        self._solution = solution

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

        # Create a set of indexes in the solution that we are going to mutate
        mutation_indexes = set()
        solution_length = len(solution)
        if self._flip_at_least_one:
            mutation_indexes.add(np.random.randint(solution_length))

        mutation_indexes = mutation_indexes.union(set(np.where(np.random.random([solution_length]) < mutation_probability)[0]))

        # Create the number of mutations that is the same as the number of mutation indexes
        num_mutations = len(mutation_indexes)
        mutations = [self._environment.get_random_action() for _ in range(num_mutations)]

        # Replace values in the solutions with mutated values
        new_solution = np.copy(solution)
        new_solution[list(mutation_indexes)] = mutations
        return new_solution

    def run(self):

        while not self._environment.is_game_over():
            action = self._get_next_action()
            self._environment.perform_action(action)

            for _ in range(self._ignore_frames):
                self._environment.perform_action(action)


        score = self._environment.get_current_score()
        self._logger.info('Final score: %.2f' % (score))

