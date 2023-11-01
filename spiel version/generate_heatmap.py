import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from murder_game_core import MurderParams,MurderGame
from open_spiel.python import policy as policy_lib
import itertools as it

with open("cfr_policy.pkl", "rb") as file:
    cfr_policy_array = pickle.load(file)

with open("reinforce_policy.pkl", "rb") as file:
    reinforce_policy_array = pickle.load(file)

with open("greedy_policy.pkl", "rb") as file:
    greedy_policy_array = pickle.load(file)

def print_policy(policy):
    for state, probs in zip(it.chain(*policy.states_per_player),
                            policy.action_probability_array):
        print(f'{state:6}   p={probs}')

params = MurderParams(1, 5, 5)
game = MurderGame(game_params=params)

# fixed_policy = policy_lib.TabularPolicy(game)
# random_policy_array = fixed_policy.action_probability_array.copy()
data_array = np.array(cfr_policy_array[289:295])

print(data_array)


plt.figure(figsize=(3, 3))
sns.heatmap(data_array, cmap="YlGnBu", annot=True, cbar=True)

plt.show()
