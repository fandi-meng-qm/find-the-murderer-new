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


eval(rl_policy)

rl_policy=rl_policy.action_probability_array
with open("reinforce_policy.pkl", "wb") as file:
    pickle.dump(rl_policy, file)
