import numpy as np
import pickle
from murder_game_core import MurderGame, MurderParams
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib
from test_utilities import print_policy,policy_as_list

params = MurderParams(1, 5, 5)
game = MurderGame(game_params=params)

fixed_policy = policy_lib.TabularPolicy(game)

def evaluate(state, rl_policy, player):
    if state.is_terminal():
        return state.returns()[player]
    elif state.current_player() == player:
        ap = policy_as_list(rl_policy, state)
    elif state.is_chance_node():
        ap = state.chance_outcomes()
    else:
        ap = policy_as_list(fixed_policy, state)

    return sum(p * evaluate(state.child(a), rl_policy, player) for a, p in ap if p != 0.0)


def eval(rl_policy):
    p0 = evaluate(game.new_initial_state(), rl_policy, player=0)
    p1 = evaluate(game.new_initial_state(), rl_policy, player=1)
    print()
    print(f"p0: {p0}")
    print(f"p1: {p1}")
    print()
    return (p0 + p1)

with open("cfr_policy.pkl", "rb") as file:
    cfr_policy_array = pickle.load(file)

policy = policy_lib.TabularPolicy(game)
policy.action_probability_array = cfr_policy_array
cfr_policy = policy

# Evaluate the greedy policy
greedy_policy = policy_lib.TabularPolicy(game)
greedy_policy.action_probability_array = (np.eye(game.num_distinct_actions())
              [np.argmax(cfr_policy.action_probability_array, axis=-1)])

print_policy(greedy_policy)
eval(greedy_policy)

greedy_policy=greedy_policy.action_probability_array
with open("greedy_policy.pkl", "wb") as file:
    pickle.dump(greedy_policy, file)