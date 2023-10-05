import itertools
from typing import NamedTuple, Set, Tuple, List
from game_interface_multi import *
from open_spiel.python import games  # pylint:disable=unused-import
import pyspiel
from game_core import *


class TreeNode:
    def __init__(self, state):
        self.state = state
        self.children = []


def step_kill(state, victim: Person):
    new_state = copy.deepcopy(state)
    new_state.alive.remove(victim)
    new_state.dead.append(victim)
    new_state.points -= state.cost_list[state.people.index(victim)]
    new_state.player = 1

    return new_state


def step_accuse(state, suspect: Person):
    new_state = copy.deepcopy(state)
    new_state.accused.append(suspect)
    new_state.player = 0

    return new_state


def _is_terminal(node):
    state = node.state
    if state.killer in set(state.accused):
        score = len(state.alive) / len(state.people)
        return score
    else:
        return None


def create_tree(parent_node, model):
    if parent_node.state == 'Root':
        chance_states = MurderGameState.get_init_states(model.state)
        # print(chance_states)
        for i in chance_states:
            child_node = TreeNode(i)
            parent_node.children.append(child_node)
            child_node.state.step = 0
            child_node.state.player = 0
            create_tree(child_node, model)
    else:
        if _is_terminal(parent_node) is None:
            if parent_node.state.player == 0:
                killer_obs = ObservationForKiller.from_game_state(parent_node.state)
                people, victims, points, cost_list = KillerInterface.get_actions(killer_obs)
                if victims is None or victims == []:
                    child_node = copy.deepcopy(parent_node)
                    parent_node.children.append(child_node)
                    child_node.state.player = 1
                    create_tree(child_node, model)
                else:
                    for victim in victims:
                        new_state = step_kill(parent_node.state, victim)
                        child_node = TreeNode(new_state)
                        parent_node.children.append(child_node)
                        create_tree(child_node, model)

            if parent_node.state.player == 1:
                detective_obs = ObservationForDetective.from_game_state(parent_node.state)
                suspects = DetectiveInterface.get_actions(detective_obs)
                for suspect in suspects:
                    new_state = step_accuse(parent_node.state, suspect)
                    child_node = TreeNode(new_state)
                    parent_node.children.append(child_node)
                    create_tree(child_node, model)

        else:
            child_node = TreeNode(_is_terminal(parent_node))
            parent_node.children.append(child_node)

    return parent_node





states = []

def get_states(root):
    if not root:
        return
    for child in root.children:
        states.append(child.state)
        get_states(child)
    for i in states:
        if type(i) ==float:
            states.remove(i)
    return states

def get_all_states(game):
    Root = TreeNode('Root')
    tree = create_tree(Root, game)
    states = get_states(tree)
    new_list = []
    for state in states:
        if state.player == 0 and state.is_terminal() is False:
            new_list.append(state)
    for state in states:
        if state.player == 1 and state.is_terminal() is False:
            new_list.append(state)
    return new_list

def state_to_obs(state):
    if state.player == 0:
        obs = ObservationForKiller.from_game_state(state)
        return obs
    if state.player == 1:
        obs = ObservationForDetective.from_game_state(state)
        return obs

# states = get_all_states(MurderGameModel(1, 4, 4))
# print(len(states))


obs_list=[]
def get_all_obs(game):
    states = get_all_states(game)
    for state in states:
        if state_to_obs(state) not in obs_list:
            obs_list.append(state_to_obs(state))
    return obs_list


k_obs_list=[]
d_obs_list=[]
def get_kd_obs(game):
    obs_list=get_all_obs(game)
    for obs in obs_list:
        if obs.player==0:
            k_obs_list.append(str(obs))
        if obs.player==1:
            d_obs_list.append(str(obs))

    return k_obs_list,d_obs_list

#
# k,d = get_kd_obs(MurderGameModel(1, 4, 4))
# print(k)
#
# for i in k:
#     print(i)


#
#
# def get_all_states(game):
#     Root = TreeNode('Root')
#     tree = create_tree(Root, game)
#     states = get_states(tree)
#     obs_list = [state_to_obs(state) for state in states]
#     new_list = []
#     for i in obs_list:
#         if i not in new_list:
#             new_list.append(i)
#     return new_list











# def get_all_paths(root, path, paths):
#     if root:
#         path.append(root.state)
#         for i in root.children:
#             get_all_paths(i, path, paths)
#         if type(root.state) == float:
#             add = copy.deepcopy(path)
#             paths.append(add)
#         path.pop()
#         return True


# def get_states(path):
#     k_obs = []
#     d_obs = []
#     for state in path:
#         if state.player == 0:
#             k_obs.append(ObservationForKiller.from_game_state(state))
#         if state.player == 1:
#             d_obs.append(ObservationForDetective.from_game_state(state))
#     k_obs_all = [0] * len(k_obs)
#     d_obs_all = [0] * len(d_obs)
#     for i in range(len(k_obs)):
#         k_obs_all[i] = k_obs[:i + 1]
#     for i in range(len(d_obs)):
#         d_obs_all[i] = d_obs[:i + 1]
#     return k_obs_all, d_obs_all


# get all information states
# def get_all_history(game):
#     root = TreeNode('Root')
#     tree = create_tree(root, MurderGameModel(1, 3, 3))
#     paths = []
#     get_all_paths(tree, [], paths)
#     _k_states = []
#     _d_states = []
#     for i in paths:
#         i.pop(0)
#         i.pop()
#         k_obs_all, d_obs_all = get_states(i)
#         _k_states += k_obs_all
#         _d_states += d_obs_all
#     k_states = []
#     [k_states.append(x) for x in _k_states if x not in k_states]
#     d_states = []
#     [d_states.append(x) for x in _d_states if x not in d_states]
#
#     states = k_states + d_states
#     # print(states)
#     return states



