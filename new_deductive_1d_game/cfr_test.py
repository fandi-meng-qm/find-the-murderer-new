import random

from open_spiel.python.algorithms import cfr
import itertools as it
from new_deductive_1d_game.version1.game_core import MurderGame, MurderParams
from matplotlib import pyplot as plt
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr


params = MurderParams(4, 1, 1)
game = MurderGame(game_params=params)


def print_policy(policy):
    for state, probs in zip(it.chain(*policy.states_per_player),
                            policy.action_probability_array):
        print(f'{state:6}   p={probs}')


def get_state0_policy(policy):
    state0_policy = []
    for state, probs in zip(it.chain(*policy.states_per_player[0]),
                            policy.action_probability_array[0]):
        state0_policy.append(probs)
    return state0_policy

cfr_steps = []
optimal_action_prob = []

def get_cfr_policy(game: MurderGame, n: int):
    cfr_solver = cfr.CFRSolver(game)
    average_policy = None
    for i in range(n):
        cfr_solver.evaluate_and_update_policy()
        average_policy = cfr_solver.average_policy()
        state0_policy = get_state0_policy(average_policy)
        if i & (i - 1) == 0:
            cfr_steps.append(i)
            optimal_action_prob.append(state0_policy[int(params.m_grid/2-1)])
    return average_policy


def get_mccfr_policy(game: MurderGame, n: int):
    cfr_solver = external_mccfr.ExternalSamplingSolver(
        game, external_mccfr.AverageType.SIMPLE)
    average_policy = None
    for i in range(n):
        cfr_solver.iteration()
        average_policy = cfr_solver.average_policy()
        inin_state_policy = average_policy.action_probabilities(game.new_initial_state().child(random.choice(range(params.m_grid))))
        if i & (i - 1) == 0:
            cfr_steps.append(i)
            optimal_action_prob.append(inin_state_policy[int(params.m_grid/2-1)])
    return average_policy

def test_cfr(n):
    cfr_policy = get_cfr_policy(game, n)
    print_policy(cfr_policy)
    fig, ax = plt.subplots()
    ax.set_title("Policy Conv by CFR Iteration")
    ax.plot(cfr_steps, optimal_action_prob)
    fig.show()

def test_mccfr(n):
    cfr_policy = get_mccfr_policy(game, n)
    inin_state_policy = cfr_policy.action_probabilities(
        game.new_initial_state().child(random.choice(range(params.m_grid))))
    print(inin_state_policy)
    fig, ax = plt.subplots()
    ax.set_title("Policy Conv by MCCFR Iteration")
    ax.plot(cfr_steps, optimal_action_prob)
    fig.show()

if __name__ == '__main__':
    # test_cfr(257)
    test_mccfr(2049)

