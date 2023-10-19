from typing import List, Tuple

import pyspiel
from pyspiel import TabularPolicy, State, PlayerId, Game
from open_spiel.python.algorithms import exploitability, cfr

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
    cfr_solver = cfr.CFRSolver(game)
    cfr_solver.evaluate_and_update_policy()
    average_policy = cfr_solver.average_policy()
    # average_policy = None
    for i in range(n):
        cfr_solver.evaluate_and_update_policy()
        average_policy = cfr_solver.average_policy()
        if i & (i - 1) == 0:
            nc = exploitability.nash_conv(game, average_policy)
            eval_steps.append(i)
            eval_nash_conv.append(nc)
            print(f'Nash conv after step {i} is {nc}')
    # print(average_policy)
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

