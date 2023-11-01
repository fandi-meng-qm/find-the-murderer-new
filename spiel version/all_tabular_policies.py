from test_utilities import get_cfr_policy, total_advantage, print_policy, advantage_as_policy_player
from murder_game_core import MurderGame, MurderParams
from open_spiel.python import policy as policy_lib
import pickle
from matplotlib import pyplot as plt
from test_utilities import policy_as_list, sample

params = MurderParams(1, 5, 5)
game = MurderGame(game_params=params)

uniform_random_policy = policy_lib.TabularPolicy(game)

with open("cfr_policy.pkl", "rb") as file:
    cfr_policy_array = pickle.load(file)

policy = policy_lib.TabularPolicy(game)
policy.action_probability_array = cfr_policy_array
cfr_policy = policy


with open("reinforce_policy.pkl", "rb") as file:
    reinforce_policy_array = pickle.load(file)
rl_policy = policy_lib.TabularPolicy(game)
rl_policy.action_probability_array = reinforce_policy_array
reinforce_policy=rl_policy


with open("greedy_policy.pkl", "rb") as file:
    greedy_policy_array = pickle.load(file)
greedy_policy = policy_lib.TabularPolicy(game)
greedy_policy.action_probability_array = greedy_policy_array
greedy_policy=greedy_policy

