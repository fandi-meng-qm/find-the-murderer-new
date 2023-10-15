import numpy as np
from cfr_test import policy_as_list,sample ,evaluate, eval
import policy
from game_core import MurderGameModel

game = MurderGameModel(1,5 , 5)

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


fixed_policy = policy.TabularPolicy(game)
rl_policy = policy.TabularPolicy(game)
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

eval(rl_policy)