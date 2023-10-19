import copy
import itertools as it

import numpy
from matplotlib import pyplot as plt
import numpy as np
import logging
import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib
# import exploitability
from murder_game_core import MurderGame, MurderParams

# from policy import TabularPolicy


params = MurderParams(1, 5, 5)
game = MurderGame(game_params=params)

fixed_policy = policy_lib.TabularPolicy(game)

policy = policy_lib.TabularPolicy(game)


# print(exploitability.nash_conv(game, policy))

def print_policy(policy):
    for state, probs in zip(it.chain(*policy.states_per_player),
                            policy.action_probability_array):
        print(f'{state:6}   p={probs}')
        # print(probs)


# print_policy(fixed_policy)
print_policy(policy)
print(exploitability.nash_conv(game, policy))


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

        # Return the value of this state for all players.
        return value


initial_state = game.new_initial_state()
curr_policy = policy.action_probability_array.copy()
legal_mask = copy.deepcopy(curr_policy)
legal_mask[legal_mask > 0] = 1
regrets = np.zeros_like(policy.action_probability_array)
eval_steps = []
eval_nash_conv = []
for step in range(129):
    # Compute regrets
    calc_cfr(initial_state, np.ones(1 + game.num_players()))

    # Find the new regret-matching policy
    floored_regrets = np.maximum(regrets, 1e-16) * legal_mask
    sum_floored_regrets = np.sum(floored_regrets, axis=1, keepdims=True)
    curr_policy = floored_regrets / sum_floored_regrets

    # Update the average policy
    lr = 1 / (1 + step)
    policy.action_probability_array *= (1 - lr)
    policy.action_probability_array += curr_policy * lr

    # Evaluate the average policy
    if step & (step - 1) == 0:
        nc = exploitability.nash_conv(game, policy)
        eval_steps.append(step)
        eval_nash_conv.append(nc)
        print(f'Nash conv after step {step} is {nc}')

fig, ax = plt.subplots()
ax.set_title("NashConv by CFR Iteration")
ax.plot(eval_steps, eval_nash_conv)
fig.show()

print_policy(policy)


def sample(actions_and_probs):
    actions, probs = zip(*actions_and_probs)
    return np.random.choice(actions, p=probs)


def policy_as_list(policy, state):
    return list(enumerate(policy.policy_for_key(state.information_state_string())))


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


eval(policy)
