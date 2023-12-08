import random
from typing import List, Tuple
from evolutionary import evaluate_es, evaluate_policy
import numpy as np
from tabular_policy import InfoSetTabularPolicy
from game_core import MurderGame, MurderParams
import matplotlib.pyplot as plt

def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)

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






def reinforce(n_iterations, game, lr):
    policy = InfoSetTabularPolicy(game).uniform_random()
    return_list = []
    for i in range(n_iterations):
        trajectory, reward = generate_trajectory(game,policy)
        for s, a in trajectory:
            logits = np.log(policy[s])
            logits[a] += lr * (10+reward)
            policy [s] = softmax(logits)

        current_fitness = evaluate_policy(policy, game)
        return_list.append(current_fitness)
    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('iterations')
    # plt.ylabel('Average returns of policy')
    # plt.title('REINFORCE on 1-D Simple Game')
    # plt.show()
    # print(np.argmax(policy[(1,1,1,1)]))
    return policy, return_list







if __name__ == '__main__':
    params = MurderParams(4, 1, 1)
    game = MurderGame(game_params=params)
    reinforce(1000, game, 0.001)
    print(game.new_initial_state().init_actions)

#     3,5,6,9,10,12
