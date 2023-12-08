import copy
import numpy as np
from tabular_policy import InfoSetTabularPolicy
from game_core import MurderGame, MurderParams
from typing import List, Tuple, Union, Dict
import random

def generate_trajectory(game,policy):
    trajectory = []
    state = game.new_initial_state()
    if state.step == 0:
        action = random.choice(state.legal_actions())
        state.apply_action(action)
        while not state.is_terminal():
            # action = random.choices(state.legal_actions(), weights=softmax(policy[tuple(state.information_state)]), k=1)[0]
            action = np.argmax(policy[tuple(state.information_state)])
            trajectory.append((tuple(state.information_state), action))
            state.apply_action(action)

    return trajectory, state.returns()[0]

def get_regret_dict(policy):
    regret_dict=dict()
    for key,value in policy.items():
        regret_dict[key]=[0]*len(value)
    return regret_dict

def mccfr(n):
    regret_dict = get_regret_dict(policy)



if __name__ == '__main__':
    params = MurderParams(4, 1, 1)
    game = MurderGame(game_params=params)
    policy = InfoSetTabularPolicy(game).uniform_random()
