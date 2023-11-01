from typing import List, Tuple

import pyspiel
from pyspiel import TabularPolicy, State, PlayerId, Game
from open_spiel.python.algorithms import exploitability, cfr
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
import numpy as np

from murder_game_core import MurderGame, MurderParams
import itertools as it


def policy_as_list(policy: TabularPolicy, state: MurderGame):
    # print(state.current_player())
    # print(policy.states_per_player)
    # print("information_string:",  state.information_state_string())
    policy_list = list(enumerate(policy.policy_for_key(state.information_state_string())))
    # print("Returning: ", policy_list)
    return policy_list


def print_policy(policy: TabularPolicy) -> None:
    for state, probs in zip(it.chain(*policy.states_per_player),
                            policy.action_probability_array):
        print(f'{state:6}   p={probs}')


def sample(actions_and_probs: List[Tuple[int, float]]) -> int:
    actions, probs = zip(*actions_and_probs)
    return np.random.choice(actions, p=probs)

eval_steps = []
eval_nash_conv = []
def get_cfr_policy(game: MurderGame, n: int):
    cfr_solver = external_mccfr.ExternalSamplingSolver(
        game, external_mccfr.AverageType.SIMPLE)
    # cfr_solver = cfr.CFRSolver(game)
    # cfr_solver.evaluate_and_update_policy()
    # average_policy = cfr_solver.average_policy()
    # average_policy = None
    for i in range(n):
        # cfr_solver.evaluate_and_update_policy()
        # average_policy = cfr_solver.average_policy()
        cfr_solver.iteration()
        if i & (i - 1) == 0:
            conv = exploitability.nash_conv(game, cfr_solver.average_policy())
            eval_steps.append(i)
            # eval_nash_conv.append(nc)
            # print(f'Nash conv after step {i} is {nc}')
            print("Iteration {} exploitability {}".format(i, conv))
    average_policy=cfr_solver.average_policy()
    print_policy(average_policy)
    return average_policy,eval_steps,eval_nash_conv



def advantage_as_policy_player(state: State, player: TabularPolicy, opponent: TabularPolicy, player_role: int) -> float:
    """
    Computes the advantage (expected reward) for player compared to opponent when player only plays as player one
    """
    if state.is_terminal():
        return state.returns()[player_role]
    elif state.current_player() == PlayerId.CHANCE:
        ap = state.chance_outcomes()
    elif state.current_player() == player_role:
        ap = policy_as_list(player, state)
    elif state.current_player() == 1 - player_role:
        ap = policy_as_list(opponent, state)
    else:
        raise Exception("Should not be here")

    return sum(p * advantage_as_policy_player(state.child(a), player, opponent, player_role) for a, p in ap if p != 0.0)


def total_advantage(game: Game, player: TabularPolicy, opponent: TabularPolicy) -> Tuple[float, float]:
    """
    Computes the total advantage (expected reward) for player compared to opponent.
    """
    results = (advantage_as_policy_player(game.new_initial_state(), player, opponent, player_role=0),
               advantage_as_policy_player(game.new_initial_state(), player, opponent, player_role=1))
    print(f"killer's advantage with policy: {results[0]},detective's advantage with policy: {results[1]}")
    return results


'''

people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 0),points10,cost_list[0, 2, 3, 4, 5]   p=[0.         0.22733007 0.38247681 0.28827349 0.10191963]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 0),points8,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.98276392 0.00861804 0.00861804]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 3)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 2), (0, 3)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 0),points8,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.01313194 0.97552706 0.01134099]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 2)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 3), (0, 2)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 0),points8,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.0108812  0.0028198  0.98629899]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 4), (0, 3)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 4), (0, 2)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 2)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 0),points7,cost_list[0, 2, 3, 4, 5]   p=[0.         0.98201552 0.         0.00167599 0.0163085 ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 3)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 1), (0, 3)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 0),points7,cost_list[0, 2, 3, 4, 5]   p=[0.         0.00167599 0.         0.98201552 0.0163085 ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 3), (0, 1)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 0),points7,cost_list[0, 2, 3, 4, 5]   p=[0.         0.01903531 0.         0.01903531 0.96192938]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 4), (0, 1)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 1)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 1)],killer(0, 0),points6,cost_list[0, 2, 3, 4, 5]   p=[0.         0.98538703 0.0123893  0.         0.00222367]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 2)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 2)],killer(0, 0),points6,cost_list[0, 2, 3, 4, 5]   p=[0.         0.01720639 0.98056994 0.         0.00222367]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 4)],killer(0, 0),points6,cost_list[0, 2, 3, 4, 5]   p=[0.         0.5041478  0.49362853 0.         0.00222367]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 1)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.98051256 0.01319792 0.00628952 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 2)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 3)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 2)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.01319792 0.98051256 0.00628952 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 1)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 3)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 3)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.59108261 0.40262787 0.00628952 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 1),points10,cost_list[2, 0, 2, 4, 5]   p=[0.10843811 0.         0.39607977 0.29434643 0.20113569]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.91945073 0.04025028 0.04029899]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 3)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 2), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.0183283  0.97576026 0.00591144]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 2)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 3), (0, 2)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.01909682 0.87437133 0.10653185]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 4), (0, 3)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 4), (0, 2)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 4), (0, 2)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.97391361 0.         0.         0.02446796 0.00161843]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 3)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 0), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.01603994 0.         0.         0.96792012 0.01603994]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 3), (0, 0)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.00161843 0.         0.         0.02446796 0.97391361]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 4), (0, 0)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 0)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 0)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.98678265 0.         0.01103956 0.         0.00217779]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 2)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 2)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.01330736 0.         0.98451484 0.         0.00217779]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 4)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.50733243 0.         0.49048977 0.         0.00217779]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 0)],killer(0, 1),points5,cost_list[2, 0, 2, 4, 5]   p=[0.48753252 0.         0.50928045 0.00318703 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 0), (0, 2)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 0), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 2)],killer(0, 1),points5,cost_list[2, 0, 2, 4, 5]   p=[0.10985469 0.         0.88695827 0.00318703 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 0)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 3)],killer(0, 1),points5,cost_list[2, 0, 2, 4, 5]   p=[0.01718311 0.         0.97962985 0.00318703 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 2),points11,cost_list[5, 3, 0, 3, 5]   p=[0.20064457 0.29935543 0.         0.29935543 0.20064457]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.         0.11096731 0.         0.88583786 0.00319483]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 1), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.         0.00833181 0.         0.98847336 0.00319483]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.         0.04165271 0.         0.95515246 0.00319483]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 4), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 4), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.01503377 0.         0.         0.48490637 0.50005986]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 0), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.5 0.  0.  0.  0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 0), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.00594995 0.         0.         0.97368865 0.0203614 ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.5 0.  0.  0.  0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.5 0.  0.  0.  0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 3), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.00249302 0.         0.         0.00249302 0.99501397]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 4), (0, 0)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.5 0.  0.  0.  0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 0)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.99501397 0.00249302 0.         0.         0.00249302]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 0), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 1)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.0203614  0.97368865 0.         0.         0.00594995]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.5 0.  0.  0.  0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.5 0.  0.  0.  0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 1), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 4)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.50005986 0.48490637 0.         0.         0.01503377]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 4), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 4), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 0)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.00319483 0.95515246 0.         0.04165271 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 1)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.00319483 0.98847336 0.         0.00833181 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 3)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.00319483 0.88583786 0.         0.11096731 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 3), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 3), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 3),points10,cost_list[5, 4, 2, 0, 2]   p=[0.20113569 0.29434643 0.39607977 0.         0.10843811]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 3),points5,cost_list[5, 4, 2, 0, 2]   p=[0.         0.00318703 0.97962985 0.         0.01718311]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 1), (0, 2)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 3),points5,cost_list[5, 4, 2, 0, 2]   p=[0.         0.00318703 0.88695827 0.         0.10985469]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 2), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 3),points5,cost_list[5, 4, 2, 0, 2]   p=[0.         0.00318703 0.50928045 0.         0.48753252]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 4), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 4), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 4), (0, 2)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.00217779 0.         0.49048977 0.         0.50733243]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 0), (0, 2)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.00217779 0.         0.98451484 0.         0.01330736]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 2), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.00217779 0.         0.01103956 0.         0.98678265]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 4), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 2)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.97391361 0.02446796 0.         0.         0.00161843]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 0), (0, 1)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.01603994 0.96792012 0.         0.         0.01603994]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 1), (0, 0)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.00161843 0.02446796 0.         0.         0.97391361]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 0)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 1)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 0)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.10653185 0.87437133 0.01909682 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 0), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 0), (0, 2)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 1)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.00591144 0.97576026 0.0183283  0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 2)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 2)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.04029899 0.04025028 0.91945073 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 0)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 1)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 4),points10,cost_list[5, 4, 3, 2, 0]   p=[0.10191963 0.28827349 0.38247681 0.22733007 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.         0.00628952 0.40262787 0.59108261 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 1), (0, 3)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 1), (0, 2)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.         0.00628952 0.98051256 0.01319792 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 3)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.         0.00628952 0.01319792 0.98051256 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 2)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 4),points6,cost_list[5, 4, 3, 2, 0]   p=[0.00222367 0.         0.49362853 0.5041478  0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 0), (0, 3)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 0), (0, 2)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 4),points6,cost_list[5, 4, 3, 2, 0]   p=[0.00222367 0.         0.98056994 0.01720639 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 3)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 4),points6,cost_list[5, 4, 3, 2, 0]   p=[0.00222367 0.         0.0123893  0.98538703 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 2)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 4),points7,cost_list[5, 4, 3, 2, 0]   p=[0.96192938 0.01903531 0.         0.01903531 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 3)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 4),points7,cost_list[5, 4, 3, 2, 0]   p=[0.0163085  0.98201552 0.         0.00167599 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 3)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 4),points7,cost_list[5, 4, 3, 2, 0]   p=[0.0163085  0.00167599 0.         0.98201552 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 0)],killer(0, 4),points8,cost_list[5, 4, 3, 2, 0]   p=[0.98629899 0.0028198  0.0108812  0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 2)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 1)],killer(0, 4),points8,cost_list[5, 4, 3, 2, 0]   p=[0.01134099 0.97552706 0.01313194 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 2)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 2)],killer(0, 4),points8,cost_list[5, 4, 3, 2, 0]   p=[0.00861804 0.00861804 0.98276392 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.5 0.5 0.  0.  0. ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.5 0.5 0.  0.  0. ]
alive[(0, 0), (0, 2), (0, 3), (0, 4)],dead[(0, 1)],accused[]   p=[0.22392076 0.         0.36600721 0.32187775 0.08819428]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 1), (0, 2)],accused[(0, 2)]   p=[0.02030033 0.         0.         0.77268974 0.20700994]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 1), (0, 3)],accused[(0, 2)]   p=[0.03490409 0.         0.         0.         0.96509591]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 1), (0, 4)],accused[(0, 2)]   p=[0.03257448 0.         0.         0.96742552 0.        ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 1), (0, 2)],accused[(0, 3)]   p=[0.39891575 0.         0.         0.         0.60108425]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 1), (0, 3)],accused[(0, 3)]   p=[0.00499812 0.         0.73017958 0.         0.2648223 ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 1), (0, 4)],accused[(0, 3)]   p=[0.04318028 0.         0.95681972 0.         0.        ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 1), (0, 2)],accused[(0, 4)]   p=[0.3650229 0.        0.        0.6349771 0.       ]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 1), (0, 3)],accused[(0, 4)]   p=[0.08459737 0.         0.91540263 0.         0.        ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 1), (0, 4)],accused[(0, 4)]   p=[0.02037351 0.         0.7440577  0.23556879 0.        ]
alive[(0, 0), (0, 1), (0, 3), (0, 4)],dead[(0, 2)],accused[]   p=[0.10783446 0.39216554 0.         0.39216554 0.10783446]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 2), (0, 1)],accused[(0, 1)]   p=[0.28162824 0.         0.         0.43674351 0.28162824]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 2), (0, 3)],accused[(0, 1)]   p=[0.5 0.  0.  0.  0.5]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 2), (0, 4)],accused[(0, 1)]   p=[0.44892267 0.         0.         0.55107733 0.        ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 2), (0, 1)],accused[(0, 3)]   p=[0.5 0.  0.  0.  0.5]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 2), (0, 3)],accused[(0, 3)]   p=[0.28162824 0.43674351 0.         0.         0.28162824]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 2), (0, 4)],accused[(0, 3)]   p=[0.44892267 0.55107733 0.         0.         0.        ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 2), (0, 1)],accused[(0, 4)]   p=[0.30475791 0.         0.         0.69524209 0.        ]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 2), (0, 3)],accused[(0, 4)]   p=[0.30475791 0.69524209 0.         0.         0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 2), (0, 4)],accused[(0, 4)]   p=[0.0889533  0.45552335 0.         0.45552335 0.        ]
alive[(0, 0), (0, 1), (0, 2), (0, 4)],dead[(0, 3)],accused[]   p=[0.08819428 0.32187775 0.36600721 0.         0.22392076]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 3), (0, 1)],accused[(0, 1)]   p=[0.2648223  0.         0.73017958 0.         0.00499812]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 3), (0, 2)],accused[(0, 1)]   p=[0.60108425 0.         0.         0.         0.39891575]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 3), (0, 1)],accused[(0, 2)]   p=[0.96509591 0.         0.         0.         0.03490409]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 3), (0, 2)],accused[(0, 2)]   p=[0.20700994 0.77268974 0.         0.         0.02030033]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 3), (0, 1)],accused[(0, 4)]   p=[0.51308633 0.         0.48691367 0.         0.        ]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 3), (0, 2)],accused[(0, 4)]   p=[0.50949377 0.49050623 0.         0.         0.        ]
alive[(0, 0), (0, 1), (0, 2), (0, 3)],dead[(0, 4)],accused[]   p=[0.21872653 0.08374804 0.16587342 0.53165201 0.        ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 4), (0, 1)],accused[(0, 1)]   p=[0.0462061  0.         0.93475271 0.01904118 0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 4), (0, 2)],accused[(0, 1)]   p=[0.11631855 0.         0.         0.88368145 0.        ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 4), (0, 1)],accused[(0, 2)]   p=[0.04932478 0.         0.         0.95067522 0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 4), (0, 2)],accused[(0, 2)]   p=[0.03982841 0.94390864 0.         0.01626295 0.        ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 4), (0, 1)],accused[(0, 3)]   p=[0.01753106 0.         0.98246894 0.         0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 4), (0, 2)],accused[(0, 3)]   p=[0.02058088 0.97941912 0.         0.         0.        ]
alive[(0, 1), (0, 2), (0, 3), (0, 4)],dead[(0, 0)],accused[]   p=[0.         0.53165201 0.16587342 0.08374804 0.21872653]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 0), (0, 2)],accused[(0, 2)]   p=[0.         0.01626295 0.         0.94390864 0.03982841]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 0), (0, 3)],accused[(0, 2)]   p=[0.         0.95067522 0.         0.         0.04932478]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 0), (0, 4)],accused[(0, 2)]   p=[0.         0.04570794 0.         0.95429206 0.        ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 0), (0, 2)],accused[(0, 3)]   p=[0.         0.88368145 0.         0.         0.11631855]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 0), (0, 3)],accused[(0, 3)]   p=[0.         0.01904118 0.93475271 0.         0.0462061 ]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 0), (0, 4)],accused[(0, 3)]   p=[0.  0.5 0.5 0.  0. ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 0), (0, 2)],accused[(0, 4)]   p=[0.         0.02779123 0.         0.97220877 0.        ]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 0), (0, 3)],accused[(0, 4)]   p=[0.         0.00841626 0.99158374 0.         0.        ]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 0), (0, 4)],accused[(0, 4)]   p=[0.         0.02701021 0.00293072 0.97005907 0.        ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 2), (0, 0)],accused[(0, 0)]   p=[0.         0.45552335 0.         0.45552335 0.0889533 ]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 2), (0, 3)],accused[(0, 0)]   p=[0.         0.69524209 0.         0.         0.30475791]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 2), (0, 4)],accused[(0, 0)]   p=[0.  0.5 0.  0.5 0. ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 2), (0, 0)],accused[(0, 3)]   p=[0.         0.55107733 0.         0.         0.44892267]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 2), (0, 0)],accused[(0, 4)]   p=[0.  0.5 0.  0.5 0. ]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 3), (0, 0)],accused[(0, 0)]   p=[0.         0.23556879 0.7440577  0.         0.02037351]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 3), (0, 2)],accused[(0, 0)]   p=[0.        0.6349771 0.        0.        0.3650229]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 3), (0, 0)],accused[(0, 2)]   p=[0.         0.96742552 0.         0.         0.03257448]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 3), (0, 0)],accused[(0, 4)]   p=[0.         0.51265082 0.48734918 0.         0.        ]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 4), (0, 0)],accused[(0, 0)]   p=[0.         0.97005907 0.00293072 0.02701021 0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 4), (0, 2)],accused[(0, 0)]   p=[0.         0.97220877 0.         0.02779123 0.        ]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 4), (0, 0)],accused[(0, 2)]   p=[0.         0.95429206 0.         0.04570794 0.        ]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 4), (0, 0)],accused[(0, 3)]   p=[0.         0.99095707 0.00904293 0.         0.        ]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 0), (0, 1)],accused[(0, 1)]   p=[0.         0.         0.98794276 0.00602862 0.00602862]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 0), (0, 3)],accused[(0, 1)]   p=[0.         0.         0.98246894 0.         0.01753106]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 0), (0, 1)],accused[(0, 3)]   p=[0.         0.         0.98851868 0.         0.01148132]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 0), (0, 1)],accused[(0, 4)]   p=[0.         0.         0.99560392 0.00439608 0.        ]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 1), (0, 0)],accused[(0, 0)]   p=[0.         0.         0.99427453 0.00286273 0.00286273]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 1), (0, 3)],accused[(0, 0)]   p=[0.         0.         0.48691367 0.         0.51308633]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 1), (0, 4)],accused[(0, 0)]   p=[0.         0.         0.48734918 0.51265082 0.        ]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 1), (0, 0)],accused[(0, 3)]   p=[0.         0.         0.98930622 0.         0.01069378]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 1), (0, 0)],accused[(0, 4)]   p=[0.         0.         0.91540263 0.08459737 0.        ]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 3), (0, 1)],accused[(0, 0)]   p=[0.         0.         0.91540263 0.         0.08459737]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 3), (0, 4)],accused[(0, 0)]   p=[0.         0.08459737 0.91540263 0.         0.        ]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 3), (0, 0)],accused[(0, 1)]   p=[0.         0.         0.95681972 0.         0.04318028]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 3), (0, 4)],accused[(0, 1)]   p=[0.01069378 0.         0.98930622 0.         0.        ]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 3), (0, 4)],accused[(0, 4)]   p=[0.00286273 0.00286273 0.99427453 0.         0.        ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 4), (0, 1)],accused[(0, 0)]   p=[0.         0.         0.99158374 0.00841626 0.        ]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 4), (0, 3)],accused[(0, 0)]   p=[0.         0.00439608 0.99560392 0.         0.        ]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 4), (0, 3)],accused[(0, 1)]   p=[0.01148132 0.         0.98851868 0.         0.        ]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 4), (0, 3)],accused[(0, 3)]   p=[0.00602862 0.00602862 0.98794276 0.         0.        ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 0), (0, 2)],accused[(0, 1)]   p=[0.         0.         0.         0.97941912 0.02058088]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 0), (0, 4)],accused[(0, 1)]   p=[0.         0.         0.00904293 0.99095707 0.        ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 1), (0, 2)],accused[(0, 0)]   p=[0.         0.         0.         0.49050623 0.50949377]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 2), (0, 1)],accused[(0, 0)]   p=[0.         0.         0.         0.69524209 0.30475791]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 2), (0, 0)],accused[(0, 1)]   p=[0.         0.         0.         0.55107733 0.44892267]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 4), (0, 0)],accused[(0, 1)]   p=[0.  0.  0.5 0.5 0. ]
'''

