import itertools as it
from matplotlib import pyplot as plt
import numpy as np
import pyspiel

from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

game = pyspiel.load_game('kuhn_poker')
# print(game.get_type().pretty_print())

policy = policy_lib.TabularPolicy(game)
# print(policy.states)
# print(policy.state_lookup)
print(policy.states_per_player)
# print(policy.action_probability_array)
# print(policy.state_in)

def print_policy(policy):
  for state, probs in zip(it.chain(*policy.states_per_player),
                          policy.action_probability_array):
    print(f'{state:6}   p={probs}')

# print_policy(policy)


# print(exploitability.nash_conv(game, policy))


def new_reach(so_far, player, action_prob):
    """Returns new reach probabilities."""
    new = np.array(so_far)
    new[player] *= action_prob
    return new


def calc_cfr(state, reach):
    """Updates regrets; returns utility for all players."""
    if state.is_terminal():
        return state.returns()
    elif state.is_chance_node():
        return sum(prob * calc_cfr(state.child(action), new_reach(reach, -1, prob))
                   for action, prob in state.chance_outcomes())
    else:
        # We are at a player decision point.
        player = state.current_player()
        index = policy.state_index(state)

        # Compute utilities after each action, updating regrets deeper in the tree.
        utility = np.zeros((game.num_distinct_actions(), game.num_players()))
        for action in state.legal_actions():
            prob = curr_policy[index][action]
            utility[action] = calc_cfr(state.child(action), new_reach(reach, player, prob))

        # Compute regrets at this state.
        cfr_prob = np.prod(reach[:player]) * np.prod(reach[player + 1:])
        value = np.einsum('ap,a->p', utility, curr_policy[index])
        for action in state.legal_actions():
            regrets[index][action] += cfr_prob * (utility[action][player] - value[player])
        # print(regrets)
        # Return the value of this state for all players.
        return value


game = pyspiel.load_game('kuhn_poker')
# game = pyspiel.load_game('turn_based_simultaneous_game(game=goofspiel(imp_info=true,num_cards=4,players=2,points_order=descending))')
policy = policy_lib.TabularPolicy(game)
initial_state = game.new_initial_state()
# print(initial_state)
curr_policy = policy.action_probability_array.copy()
regrets = np.zeros_like(policy.action_probability_array)
eval_steps = []
eval_nash_conv = []
for step in range(129):
  # Compute regrets
  calc_cfr(initial_state, np.ones(1 + game.num_players()))
  # print(curr_policy)
  # Find the new regret-matching policy
  floored_regrets = np.maximum(regrets, 1e-16)
  sum_floored_regrets = np.sum(floored_regrets, axis=1, keepdims=True)
  curr_policy = floored_regrets / sum_floored_regrets
  # print(curr_policy)
  # Update the average policy
  lr = 1 / (1 + step)
  policy.action_probability_array *= (1 - lr)
  policy.action_probability_array += curr_policy * lr

  # Evaluate the average policy
  if step & (step-1) == 0:
    nc = exploitability.nash_conv(game, policy)
    eval_steps.append(step)
    eval_nash_conv.append(nc)
    print(f'Nash conv after step {step} is {nc}')


fig, ax = plt.subplots()
ax.set_title("NashConv by CFR Iteration")
ax.plot(eval_steps, eval_nash_conv)
fig.show()

fig, ax = plt.subplots()
ax.set_title("NashConv by CFR Iteration (log-log scale)")
ax.loglog(eval_steps, eval_nash_conv)
fig.show()

# Display the whole policy
print_policy(policy)

# How likely are we to bet with a Jack?
alpha = policy.action_probability_array[policy.state_lookup['0']][1]
print(f'P(bet with Jack) = alpha = {alpha:.3}')

# How likely are we to bet with a King?
pK = policy.action_probability_array[policy.state_lookup['2']][1]
print(f'P(bet with King) = {pK:.3}, cf {alpha * 3:.3}')

# How likely are we to call with a Queen?
pQ = policy.action_probability_array[policy.state_lookup['1pb']][1]
print(f'P(call with Queen after checking) = {pQ:.3}, cf {alpha + 1/3:.3}')

def sample(actions_and_probs):
  actions, probs = zip(*actions_and_probs)
  return np.random.choice(actions, p=probs)

def policy_as_list(policy, state):
  return list(enumerate(policy.policy_for_key(state.information_state_string())))
#
# def env_action(state):
#   if state.is_chance_node():
#     p = state.chance_outcomes()
#   else:
#     p = policy_as_list(fixed_policy, state)
#   return sample(p)
#
# def softmax(x):
#   x = np.exp(x - np.max(x, axis=-1, keepdims=True))
#   return x / np.sum(x, axis=-1, keepdims=True)
#
# def generate_trajectory(state, player):
#   trajectory = []
#   while not state.is_terminal():
#     if state.current_player() == player:
#       action = sample(policy_as_list(rl_policy, state))
#       trajectory.append((rl_policy.state_index(state), action))
#     else:
#       action = env_action(state)
#     state.apply_action(action)
#   return trajectory, state.returns()[player]
#
fixed_policy = policy_lib.TabularPolicy(game)
# rl_policy = policy_lib.TabularPolicy(game)
# for _ in range(5):
#   print(generate_trajectory(game.new_initial_state(), player=0))
#
# # Run REINFORCE
# N = 10000
# lr = 0.01
# for step in range(N):
#   for player in (0, 1):
#     trajectory, reward = generate_trajectory(game.new_initial_state(), player)
#     for s, a in trajectory:
#       logits = np.log(rl_policy.action_probability_array[s])
#       logits[a] += lr * reward
#       rl_policy.action_probability_array[s] = softmax(logits)

# Evaluate the policy
def evaluate(state, rl_policy, player):
  if state.is_terminal():
    return state.returns()[player]
  elif state.current_player() == player:
    ap = policy_as_list(rl_policy, state)
  elif state.is_chance_node():
    ap = state.chance_outcomes()
  else:
    ap = policy_as_list(fixed_policy, state)
  return sum(p * evaluate(state.child(a), rl_policy, player) for a, p in ap)

def eval(rl_policy):
  p0 = evaluate(game.new_initial_state(), rl_policy, player=0)
  p1 = evaluate(game.new_initial_state(), rl_policy, player=1)
  print()
  print(f"p0: {p0}")
  print(f"p1: {p1}")
  print()
  return (p0 + p1)

eval(policy)

# # Evaluate the greedy policy
# greedy_policy = policy_lib.TabularPolicy(game)
# greedy_policy.action_probability_array = (np.eye(game.num_distinct_actions())
#               [np.argmax(rl_policy.action_probability_array, axis=-1)])
#
# print_policy(greedy_policy)
# eval(greedy_policy)
