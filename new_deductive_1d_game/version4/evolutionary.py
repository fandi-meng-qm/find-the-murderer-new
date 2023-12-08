import copy
import numpy as np
from tabular_policy import InfoSetTabularPolicy
from game_core import MurderGame, MurderParams
from typing import List, Tuple, Union, Dict
import random
import matplotlib.pyplot as plt


def softmax(x) -> List[float]:
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum(axis=0)).tolist()


def mutate_es(policy_dict, mutation_rate, sigma) -> List[float]:
    """
    Mutates a solution for the 1+1 ES.
    This involves making a small random change to action probability.
    """
    mutated_policy = copy.deepcopy(policy_dict)
    for p in mutated_policy:
        if np.random.rand() < mutation_rate:
            for i in range(len(mutated_policy[p])):
                mutated_policy[p][i] += np.random.normal(0, sigma)
            mutated_policy[p] = softmax(mutated_policy[p])

    return mutated_policy


def fitness_es(policy_dict, game) -> int:
    """
    Evaluates the effectiveness of a solution for the 1+1 ES. A better solution means finding the treasure in fewer steps.
    The fitness is the negative number of steps taken, so a higher fitness is better.
    """
    state = game.new_initial_state()
    if state.step == 0:
        action = random.choice(state.legal_actions())
        state.apply_action(action)
    while not state.is_terminal():
        # action = random.choices(state.legal_actions(), weights=policy_dict[tuple(state.information_state)], k=1)[0]
        action = np.argmax(policy_dict[tuple(state.information_state)])
        state.apply_action(action)
    return -state.step




def evaluate_es(policy_dict, game) -> float:
    fitness_list = []
    for i in range(20):
        fitness_list.append(fitness_es(policy_dict, game))
    return sum(fitness_list) / len(fitness_list)

def evaluate_policy(policy_dict, game) -> float:
    fitness_list = []
    for i in range(game.game_params.m_grid):
        state = game.new_initial_state()
        if state.step == 0:
            state.apply_action(i)

        while not state.is_terminal():
            # action = random.choices(state.legal_actions(), weights=policy_dict[tuple(state.information_state)], k=1)[0]
            action = np.argmax(policy_dict[tuple(state.information_state)])
            state.apply_action(action)
        fitness_list.append(-state.step+1)
    return sum(fitness_list) / len(fitness_list)

def evaluate(policy_dict, game) -> float:
    fitness_list = []
    for i in range(game.game_params.m_grid):
        state = game.new_initial_state()
        if state.step == 0:
            state.apply_action(i)
        while not state.is_terminal():
            # action = random.choices(state.legal_actions(), weights=policy_dict[tuple(state.information_state)], k=1)[0]
            action = np.argmax(policy_dict[tuple(state.information_state)])
            # print(state.information_state)
            # print(state.init_actions[action])
            state.apply_action(action)
        fitness_list.append(-state.step+1)
        # print(state.step)
        # print(state.returns()[0])
    # print(fitness_list)
    return sum(fitness_list) / len(fitness_list)

def one_plus_one_es(game, max_iterations, mutation_rate, sigma) -> Tuple[Union[Dict[tuple, List[float]], List[float]], List[float]]:
    """
    1+1 Evolution Strategy for solving the treasure finding game.
    """
    # curr_policy = InfoSetTabularPolicy(game).uniform_random()
    curr_policy = InfoSetTabularPolicy(game).uniform_random()
    current_fitness = evaluate_policy(curr_policy, game)
    return_list=[current_fitness]
    for _ in range(max_iterations):
        # Mutate: create a new solution by slightly modifying the current one
        new_policy = mutate_es(curr_policy, mutation_rate, sigma)
        new_fitness = evaluate_policy(new_policy, game)
        # Selection: if the new solution is better, replace the current solution
        # print('time'+str(_))
        if new_fitness > current_fitness:
            curr_policy = new_policy
            current_fitness = new_fitness
        # print(evaluate_es(curr_policy, game))
        # print(evaluate(curr_policy, game))
        return_list.append(current_fitness)

    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('iterations')
    # plt.ylabel('Average returns of policy')
    # plt.title('1+1ES on 1-D Simple Game')
    # plt.show()
    # print(curr_policy[(1,1,1,1)])
    # print(return_list)

    return curr_policy, return_list


if __name__ == "__main__":
    params = MurderParams(4, 1, 1)
    game = MurderGame(game_params=params)
    print(game.new_initial_state().init_actions)
    one_plus_one_es(game, max_iterations=1000, mutation_rate=0.9, sigma=0.6)

