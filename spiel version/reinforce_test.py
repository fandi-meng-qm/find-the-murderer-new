import itertools as it
from matplotlib import pyplot as plt
import numpy as np
import pickle
from murder_game_core import MurderGame, MurderParams
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib


def sample(actions_and_probs):
    actions, probs = zip(*actions_and_probs)
    return np.random.choice(actions, p=probs)


def policy_as_list(policy, state):
    return list(enumerate(policy.policy_for_key(state.information_state_string())))


def env_action(state):
    if state.is_chance_node():
        p = state.chance_outcomes()
    else:
        p = policy_as_list(fixed_policy, state)
    return sample(p)


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)


def generate_trajectory(state, player):
    trajectory = []
    while not state.is_terminal():
        if state.current_player() == player:
            action = sample(policy_as_list(rl_policy, state))
            trajectory.append((rl_policy.state_index(state), action))
        else:
            action = env_action(state)
        state.apply_action(action)
    return trajectory, state.returns()[player]


params = MurderParams(1, 5, 5)
game = MurderGame(game_params=params)

fixed_policy = policy_lib.TabularPolicy(game)
rl_policy = policy_lib.TabularPolicy(game)
for _ in range(5):
    print(generate_trajectory(game.new_initial_state(), player=0))

# Run REINFORCE
N = 10000
lr = 0.01
for step in range(N):
  for player in (0, 1):
    trajectory, reward = generate_trajectory(game.new_initial_state(), player)
    for s, a in trajectory:
      logits = np.log(rl_policy.action_probability_array[s])
      logits[a] += lr * reward
      rl_policy.action_probability_array[s] = softmax(logits)


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

def print_policy(policy):
    for state, probs in zip(it.chain(*policy.states_per_player),
                            policy.action_probability_array):
        print(f'{state:6}   p={probs}')

eval(rl_policy)
print_policy(rl_policy)

rl_policy=rl_policy.action_probability_array

# with open("reinforce_policy.pkl", "wb") as file:
#     pickle.dump(rl_policy, file)

'''
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 0),points10,cost_list[0, 2, 3, 4, 5]   p=[0.         0.24643538 0.27399224 0.23115761 0.24841477]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 0),points8,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.33260829 0.34136951 0.3260222 ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 3)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.         0.50599971 0.49400029]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.         0.49800001 0.50199999]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 2), (0, 3)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 0),points8,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.33551465 0.34850955 0.3159758 ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.         0.50199999 0.49800001]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 2)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 3), (0, 2)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 0),points8,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.31228997 0.35210595 0.33560408]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 4), (0, 3)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.         0.50599971 0.49400029]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 4), (0, 2)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 2)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 0),points7,cost_list[0, 2, 3, 4, 5]   p=[0.         0.33476438 0.         0.35122496 0.31401066]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 3)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.    0.    0.    0.499 0.501]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.         0.         0.49400029 0.50599971]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 1), (0, 3)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 0),points7,cost_list[0, 2, 3, 4, 5]   p=[0.         0.33194767 0.         0.34137355 0.32667877]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 3), (0, 1)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 0),points7,cost_list[0, 2, 3, 4, 5]   p=[0.         0.34182652 0.         0.33106124 0.32711224]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 4), (0, 1)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 1)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 1)],killer(0, 0),points6,cost_list[0, 2, 3, 4, 5]   p=[0.         0.34452824 0.33035771 0.         0.32511405]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 2)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 0),points4,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 2)],killer(0, 0),points6,cost_list[0, 2, 3, 4, 5]   p=[0.         0.32860558 0.35313775 0.         0.31825667]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 4)],killer(0, 0),points6,cost_list[0, 2, 3, 4, 5]   p=[0.         0.3279999  0.33129636 0.         0.34070374]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 1)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.32916877 0.32200614 0.34882509 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 2)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 3)],killer(0, 0),points3,cost_list[0, 2, 3, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 2)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.33177062 0.33712164 0.33110774 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 1)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 3)],killer(0, 0),points2,cost_list[0, 2, 3, 4, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 3)],killer(0, 0),points5,cost_list[0, 2, 3, 4, 5]   p=[0.         0.33354214 0.3368943  0.32956356 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 1),points10,cost_list[2, 0, 2, 4, 5]   p=[0.26462313 0.         0.26781774 0.23658479 0.23097435]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.32810695 0.34769985 0.3241932 ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 3)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.         0.50299996 0.49700004]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.         0.50599971 0.49400029]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 2), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.33321958 0.34405505 0.32272536]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.  0.  0.  0.5 0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 2)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 3), (0, 2)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.33802345 0.34554237 0.31643418]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 4), (0, 3)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.         0.49800001 0.50199999]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 4), (0, 2)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 4), (0, 2)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.34358759 0.         0.         0.31213696 0.34427545]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 3)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.         0.49400029 0.50599971]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.         0.         0.         0.49700004 0.50299996]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 0), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.33555211 0.         0.         0.32956621 0.33488168]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 3), (0, 0)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 1),points8,cost_list[2, 0, 2, 4, 5]   p=[0.33111483 0.         0.         0.33444259 0.33444259]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 4), (0, 0)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 0)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 0)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.33801342 0.         0.32933834 0.         0.33264824]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 2)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 2)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.34554237 0.         0.33802345 0.         0.31643418]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 1),points4,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 4)],killer(0, 1),points6,cost_list[2, 0, 2, 4, 5]   p=[0.32405528 0.         0.33459275 0.         0.34135197]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 0)],killer(0, 1),points5,cost_list[2, 0, 2, 4, 5]   p=[0.33842347 0.         0.33707248 0.32450404 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 0), (0, 2)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 0), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 2)],killer(0, 1),points5,cost_list[2, 0, 2, 4, 5]   p=[0.33528718 0.         0.33933487 0.32537795 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 0)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 3)],killer(0, 1),points3,cost_list[2, 0, 2, 4, 5]   p=[1. 0. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 3)],killer(0, 1),points5,cost_list[2, 0, 2, 4, 5]   p=[0.34107403 0.         0.33835632 0.32056964 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 2),points11,cost_list[5, 3, 0, 3, 5]   p=[0.23315238 0.25870599 0.         0.23833858 0.26980305]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.         0.33022553 0.         0.33689653 0.33287793]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 1), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.         0.33667485 0.         0.33199424 0.33133091]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.         0.35316653 0.         0.32406357 0.3227699 ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 4), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 4), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.31987906 0.         0.         0.3336001  0.34652084]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 0), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 2), (0, 3), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.4880023 0.        0.        0.        0.5119977]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 0), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.3237926  0.         0.         0.35499481 0.32121259]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.49700004 0.         0.         0.         0.50299996]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 4)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.499 0.    0.    0.    0.501]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 3), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.33845482 0.         0.         0.32779572 0.33374946]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 4), (0, 0)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.49500017 0.         0.         0.         0.50499983]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 0)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.33018528 0.34160461 0.         0.         0.3282101 ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 0), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 1)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.3293295  0.33131142 0.         0.         0.33935908]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.5 0.  0.  0.  0.5]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 2),points5,cost_list[5, 3, 0, 3, 5]   p=[0.499 0.    0.    0.    0.501]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 1), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 4)],killer(0, 2),points8,cost_list[5, 3, 0, 3, 5]   p=[0.33889829 0.33418677 0.         0.         0.32691494]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 4), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 4), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 0)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.30634237 0.34128003 0.         0.3523776  0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 1)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.31536672 0.34231664 0.         0.34231664 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 3)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 3)],killer(0, 2),points6,cost_list[5, 3, 0, 3, 5]   p=[0.31579697 0.34073311 0.         0.34346991 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 3), (0, 0)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2)],accused[(0, 3), (0, 1)],killer(0, 2),points3,cost_list[5, 3, 0, 3, 5]   p=[0. 1. 0. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 3),points10,cost_list[5, 4, 2, 0, 2]   p=[0.24366531 0.2561583  0.2515887  0.         0.24858768]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 3),points5,cost_list[5, 4, 2, 0, 2]   p=[0.         0.31383911 0.33862065 0.         0.34754024]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 1), (0, 2)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 3),points5,cost_list[5, 4, 2, 0, 2]   p=[0.         0.34407672 0.33058528 0.         0.325338  ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 2), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 3),points5,cost_list[5, 4, 2, 0, 2]   p=[0.         0.31534556 0.34711949 0.         0.33753495]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 4), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 4), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 4), (0, 2)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.32037937 0.         0.32554667 0.         0.35407396]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 0), (0, 2)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.3308614  0.         0.34025643 0.         0.32888217]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 4)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 2), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.34765858 0.         0.31964817 0.         0.33269325]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 4), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 4), (0, 2)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.3476672  0.33203676 0.         0.         0.32029604]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 4)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 0), (0, 1)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.50299996 0.49700004 0.         0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.32713866 0.33777639 0.         0.         0.33508496]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 4)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 0. 0. 1.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 1), (0, 0)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.51099823 0.48900177 0.         0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 4)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.32734725 0.33262693 0.         0.         0.34002581]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 0)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.50299996 0.49700004 0.         0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 4), (0, 1)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.50199999 0.49800001 0.         0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 0)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.32884024 0.34363225 0.32752751 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 0), (0, 1)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3)],accused[(0, 0), (0, 2)],killer(0, 3),points3,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 1)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.3238077  0.3445192  0.33167309 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 0)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3)],accused[(0, 1), (0, 2)],killer(0, 3),points4,cost_list[5, 4, 2, 0, 2]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3)],accused[(0, 2)],killer(0, 3),points8,cost_list[5, 4, 2, 0, 2]   p=[0.33911979 0.33440519 0.32647501 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 0)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.49400029 0.50599971 0.         0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3)],accused[(0, 2), (0, 1)],killer(0, 3),points6,cost_list[5, 4, 2, 0, 2]   p=[0.49300046 0.50699954 0.         0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],accused[],killer(0, 4),points10,cost_list[5, 4, 3, 2, 0]   p=[0.24670933 0.24720325 0.26619014 0.23989729 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.         0.32501228 0.34997544 0.32501228 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 1), (0, 3)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 1), (0, 2)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.         0.33352347 0.33890277 0.32757376 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 2), (0, 3)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.         0.33352347 0.33890277 0.32757376 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 3), (0, 2)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 4),points6,cost_list[5, 4, 3, 2, 0]   p=[0.32735782 0.         0.3346395  0.33800268 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 0), (0, 3)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 0), (0, 2)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 2)],killer(0, 4),points6,cost_list[5, 4, 3, 2, 0]   p=[0.30692993 0.         0.35588921 0.33718086 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 2), (0, 3)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 4),points6,cost_list[5, 4, 3, 2, 0]   p=[0.31903987 0.         0.33741583 0.3435443  0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 3), (0, 2)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 0)],killer(0, 4),points7,cost_list[5, 4, 3, 2, 0]   p=[0.32934364 0.33666949 0.         0.33398688 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 3), (0, 4)],accused[(0, 0), (0, 3)],killer(0, 4),points2,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.501 0.499 0.    0.    0.   ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 1)],killer(0, 4),points7,cost_list[5, 4, 3, 2, 0]   p=[0.34225862 0.3242668  0.         0.33347458 0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 3), (0, 4)],accused[(0, 1), (0, 3)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 0. 1. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.51399634 0.48600366 0.         0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 3), (0, 4)],accused[(0, 3)],killer(0, 4),points7,cost_list[5, 4, 3, 2, 0]   p=[0.33866588 0.32603802 0.         0.3352961  0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 0)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.51399634 0.48600366 0.         0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 3), (0, 1)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.499 0.501 0.    0.    0.   ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 0)],killer(0, 4),points8,cost_list[5, 4, 3, 2, 0]   p=[0.34475579 0.32991542 0.32532879 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 1)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 1), (0, 2), (0, 4)],accused[(0, 0), (0, 2)],killer(0, 4),points3,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 1)],killer(0, 4),points8,cost_list[5, 4, 3, 2, 0]   p=[0.33375613 0.32845853 0.33778534 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 0)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 2), (0, 4)],accused[(0, 1), (0, 2)],killer(0, 4),points4,cost_list[5, 4, 3, 2, 0]   p=[0. 0. 1. 0. 0.]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 2), (0, 4)],accused[(0, 2)],killer(0, 4),points8,cost_list[5, 4, 3, 2, 0]   p=[0.32710476 0.33039222 0.34250302 0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 0)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.50999867 0.49000133 0.         0.         0.        ]
people[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],alive[(0, 0), (0, 1), (0, 4)],accused[(0, 2), (0, 1)],killer(0, 4),points5,cost_list[5, 4, 3, 2, 0]   p=[0.49400029 0.50599971 0.         0.         0.        ]
alive[(0, 0), (0, 2), (0, 3), (0, 4)],dead[(0, 1)],accused[]   p=[0.08590362 0.         0.75387454 0.03308983 0.12713201]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 1), (0, 2)],accused[(0, 2)]   p=[0.3520246  0.         0.         0.32495966 0.32301574]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 1), (0, 3)],accused[(0, 2)]   p=[0.50949886 0.         0.         0.         0.49050114]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 1), (0, 4)],accused[(0, 2)]   p=[0.50299996 0.         0.         0.49700004 0.        ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 1), (0, 2)],accused[(0, 3)]   p=[0.48300655 0.         0.         0.         0.51699345]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 1), (0, 3)],accused[(0, 3)]   p=[0.33287793 0.         0.33022553 0.         0.33689653]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 1), (0, 4)],accused[(0, 3)]   p=[0.49500017 0.         0.50499983 0.         0.        ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 1), (0, 2)],accused[(0, 4)]   p=[0.50549978 0.         0.         0.49450022 0.        ]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 1), (0, 3)],accused[(0, 4)]   p=[0.4880023 0.        0.5119977 0.        0.       ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 1), (0, 4)],accused[(0, 4)]   p=[0.33958028 0.         0.33219109 0.32822862 0.        ]
alive[(0, 0), (0, 1), (0, 3), (0, 4)],dead[(0, 2)],accused[]   p=[0.06529878 0.88268935 0.         0.02735702 0.02465484]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 2), (0, 1)],accused[(0, 1)]   p=[0.33194767 0.         0.         0.32667877 0.34137355]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 2), (0, 3)],accused[(0, 1)]   p=[0.49700004 0.         0.         0.         0.50299996]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 2), (0, 4)],accused[(0, 1)]   p=[0.52497919 0.         0.         0.47502081 0.        ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 2), (0, 1)],accused[(0, 3)]   p=[0.48350599 0.         0.         0.         0.51649401]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 2), (0, 3)],accused[(0, 3)]   p=[0.3364356  0.32780101 0.         0.         0.3357634 ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 2), (0, 4)],accused[(0, 3)]   p=[0.4985 0.5015 0.     0.     0.    ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 2), (0, 1)],accused[(0, 4)]   p=[0.49400029 0.         0.         0.50599971 0.        ]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 2), (0, 3)],accused[(0, 4)]   p=[0.49700004 0.50299996 0.         0.         0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 2), (0, 4)],accused[(0, 4)]   p=[0.33399755 0.33466622 0.         0.33133623 0.        ]
alive[(0, 0), (0, 1), (0, 2), (0, 4)],dead[(0, 3)],accused[]   p=[0.17318852 0.08312793 0.60934263 0.         0.13434092]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 3), (0, 1)],accused[(0, 1)]   p=[0.32530772 0.         0.32857712 0.         0.34611516]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 3), (0, 2)],accused[(0, 1)]   p=[0.5015 0.     0.     0.     0.4985]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 3), (0, 1)],accused[(0, 2)]   p=[0.5274723 0.        0.        0.        0.4725277]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 3), (0, 2)],accused[(0, 2)]   p=[0.33980312 0.33240909 0.         0.         0.32778779]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 3), (0, 1)],accused[(0, 4)]   p=[0.49800001 0.         0.50199999 0.         0.        ]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 3), (0, 2)],accused[(0, 4)]   p=[0.50249998 0.49750002 0.         0.         0.        ]
alive[(0, 0), (0, 1), (0, 2), (0, 3)],dead[(0, 4)],accused[]   p=[0.00927644 0.01838395 0.96052332 0.01181628 0.        ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 4), (0, 1)],accused[(0, 1)]   p=[0.34723913 0.         0.32311674 0.32964413 0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 4), (0, 2)],accused[(0, 1)]   p=[0.49700004 0.         0.         0.50299996 0.        ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 4), (0, 1)],accused[(0, 2)]   p=[0.49600009 0.         0.         0.50399991 0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 4), (0, 2)],accused[(0, 2)]   p=[0.3519653  0.33146844 0.         0.31656627 0.        ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 4), (0, 1)],accused[(0, 3)]   p=[0.50949886 0.         0.49050114 0.         0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 4), (0, 2)],accused[(0, 3)]   p=[0.49400029 0.50599971 0.         0.         0.        ]
alive[(0, 1), (0, 2), (0, 3), (0, 4)],dead[(0, 0)],accused[]   p=[0.         0.06330379 0.67313448 0.08193655 0.18162518]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 0), (0, 2)],accused[(0, 2)]   p=[0.         0.32381404 0.         0.32187697 0.35430899]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 0), (0, 3)],accused[(0, 2)]   p=[0.         0.49350037 0.         0.         0.50649963]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 0), (0, 4)],accused[(0, 2)]   p=[0.         0.50949886 0.         0.49050114 0.        ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 0), (0, 2)],accused[(0, 3)]   p=[0.         0.48400546 0.         0.         0.51599454]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 0), (0, 3)],accused[(0, 3)]   p=[0.         0.33621495 0.33554319 0.         0.32824185]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 0), (0, 4)],accused[(0, 3)]   p=[0.         0.52048852 0.47951148 0.         0.        ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 0), (0, 2)],accused[(0, 4)]   p=[0.         0.50549978 0.         0.49450022 0.        ]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 0), (0, 3)],accused[(0, 4)]   p=[0.         0.50399991 0.49600009 0.         0.        ]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 0), (0, 4)],accused[(0, 4)]   p=[0.         0.33033029 0.32378931 0.3458804  0.        ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 2), (0, 0)],accused[(0, 0)]   p=[0.         0.33152957 0.         0.32888791 0.33958252]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 2), (0, 3)],accused[(0, 0)]   p=[0.         0.52647522 0.         0.         0.47352478]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 2), (0, 4)],accused[(0, 0)]   p=[0.         0.51399634 0.         0.48600366 0.        ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 2), (0, 0)],accused[(0, 3)]   p=[0.        0.4725277 0.        0.        0.5274723]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 2), (0, 0)],accused[(0, 4)]   p=[0.     0.5005 0.     0.4995 0.    ]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 3), (0, 0)],accused[(0, 0)]   p=[0.         0.32358452 0.33144449 0.         0.34497099]
alive[(0, 0), (0, 1), (0, 4)],dead[(0, 3), (0, 2)],accused[(0, 0)]   p=[0.        0.5149955 0.        0.        0.4850045]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 3), (0, 0)],accused[(0, 2)]   p=[0.         0.50349994 0.         0.         0.49650006]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 3), (0, 0)],accused[(0, 4)]   p=[0.         0.49150082 0.50849918 0.         0.        ]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 4), (0, 0)],accused[(0, 0)]   p=[0.         0.34107403 0.32056964 0.33835632 0.        ]
alive[(0, 0), (0, 1), (0, 3)],dead[(0, 4), (0, 2)],accused[(0, 0)]   p=[0.         0.49500017 0.         0.50499983 0.        ]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 4), (0, 0)],accused[(0, 2)]   p=[0.        0.5124974 0.        0.4875026 0.       ]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 4), (0, 0)],accused[(0, 3)]   p=[0.         0.52048852 0.47951148 0.         0.        ]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 0), (0, 1)],accused[(0, 1)]   p=[0.         0.         0.33734124 0.32934143 0.33331733]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 0), (0, 3)],accused[(0, 1)]   p=[0.         0.         0.48550406 0.         0.51449594]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 0), (0, 1)],accused[(0, 3)]   p=[0.         0.         0.50399991 0.         0.49600009]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 0), (0, 1)],accused[(0, 4)]   p=[0.        0.        0.5124974 0.4875026 0.       ]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 1), (0, 0)],accused[(0, 0)]   p=[0.         0.         0.35179755 0.32410122 0.32410122]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 1), (0, 3)],accused[(0, 0)]   p=[0.         0.         0.51899086 0.         0.48100914]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 1), (0, 4)],accused[(0, 0)]   p=[0.         0.         0.49400029 0.50599971 0.        ]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 1), (0, 0)],accused[(0, 3)]   p=[0.         0.         0.50399991 0.         0.49600009]
alive[(0, 2), (0, 3), (0, 4)],dead[(0, 1), (0, 0)],accused[(0, 4)]   p=[0.         0.         0.52098766 0.47901234 0.        ]
alive[(0, 0), (0, 2), (0, 4)],dead[(0, 3), (0, 1)],accused[(0, 0)]   p=[0.        0.        0.5119977 0.        0.4880023]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 3), (0, 4)],accused[(0, 0)]   p=[0.         0.49000133 0.50999867 0.         0.        ]
alive[(0, 1), (0, 2), (0, 4)],dead[(0, 3), (0, 0)],accused[(0, 1)]   p=[0.         0.         0.48850203 0.         0.51149797]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 3), (0, 4)],accused[(0, 1)]   p=[0.50699954 0.         0.49300046 0.         0.        ]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 3), (0, 4)],accused[(0, 4)]   p=[0.33399933 0.33399933 0.33200134 0.         0.        ]
alive[(0, 0), (0, 2), (0, 3)],dead[(0, 4), (0, 1)],accused[(0, 0)]   p=[0.         0.         0.51649401 0.48350599 0.        ]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 4), (0, 3)],accused[(0, 0)]   p=[0.         0.49800001 0.50199999 0.         0.        ]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 4), (0, 3)],accused[(0, 1)]   p=[0.48850203 0.         0.51149797 0.         0.        ]
alive[(0, 0), (0, 1), (0, 2)],dead[(0, 4), (0, 3)],accused[(0, 3)]   p=[0.32647501 0.33440519 0.33911979 0.         0.        ]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 0), (0, 2)],accused[(0, 1)]   p=[0.         0.         0.         0.49600009 0.50399991]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 0), (0, 4)],accused[(0, 1)]   p=[0.     0.     0.4985 0.5015 0.    ]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 1), (0, 2)],accused[(0, 0)]   p=[0.         0.         0.         0.48150844 0.51849156]
alive[(0, 0), (0, 3), (0, 4)],dead[(0, 2), (0, 1)],accused[(0, 0)]   p=[0.         0.         0.         0.49600009 0.50399991]
alive[(0, 1), (0, 3), (0, 4)],dead[(0, 2), (0, 0)],accused[(0, 1)]   p=[0.         0.         0.         0.49200068 0.50799932]
alive[(0, 1), (0, 2), (0, 3)],dead[(0, 4), (0, 0)],accused[(0, 1)]   p=[0.        0.        0.4850045 0.5149955 0.       ]

'''