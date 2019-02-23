import logging
import numpy as np

from RHEA import RollingHorizonEvolutionaryAlgorithm
from RHEA.environment import Environment

class MMaxGame(Environment):
    '''
    The aim of this game is to get all the numbers in an array to equal the value 'm'

    An action is either to increase or decrease a number at any index

    At a single time-step, only one action can be performed
    '''

    def __init__(self, num_dims, m):
        super(MMaxGame, self).__init__("M-Max Game")

        self._num_dims = num_dims
        self._game_state = np.zeros(self._num_dims)

        self._goal_state = np.ones(self._num_dims) * m

    def _score_state(self, state):
        return -np.abs(self._goal_state - state).mean()

    def evaluate_rollout(self, solution, discount_factor=0, ignore_frames=0):

        copied_state = np.copy(self._game_state)

        for action in solution:
            copied_state[action[0]] += action[1]

        return self._score_state(copied_state)

    def perform_action(self, action):
        self._game_state[action[0]] += action[1]

    def get_random_action(self):
        '''
        A single action is an array containing two integers.

        The first integer denotes the position in the game state array where the action will be performed.
        The second is either -1, 0 or 1 depending on whether to increase or decrease the value

        for example an action [3,-1] will decrease the value at index 3 ([3,1] will increase the value)
        '''
        return [np.random.randint(0, self._num_dims), np.random.randint(-1,2)]

    def is_game_over(self):
        return np.all(self._goal_state==self._game_state)

    def get_current_score(self):
        return self._score_state(self._game_state)

    def ignore_frame(self):
        return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    num_dims = 600
    m = 50
    num_evals = 50
    rollout_length = 10
    mutation_probability = 0.1

    # Set up the problem domain as one-max problem
    environment = MMaxGame(num_dims, m)

    rhea = RollingHorizonEvolutionaryAlgorithm(rollout_length, environment, mutation_probability, num_evals)

    rhea.run()

