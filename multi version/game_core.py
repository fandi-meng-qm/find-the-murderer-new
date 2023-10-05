from __future__ import annotations
import itertools
import copy
import math
import numpy as np
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import partial
from typing import NamedTuple, Set, Tuple, List

import pyspiel


# Using namedtuple to represent people,
# but in Python version 3.9 NamedTuple is no longer a class, which needs to be noted.
class Person(Tuple):
    location: Tuple[int, int]


def get_init_people(m_grid, n_grid, n_people):
    locations = [(x, y) for x in range(m_grid) for y in range(n_grid)]
    p = list(itertools.combinations(list(range(m_grid * n_grid)), n_people))
    init_people = []
    for i in range(len(p)):
        state = []
        for j in p[i]:
            state.append(Person(locations[j]))
        init_people.append(state)
    # print(init_people)
    return init_people


def step_kill(state, victim: Person):
    new_state = copy.deepcopy(state)
    new_state.alive.remove(victim)
    new_state.dead.append(victim)
    new_state.points -= state.cost_list[state.people.index(victim)]
    new_state.player = 1
    new_state.step += 1

    return new_state


def step_accuse(state, suspect: Person):
    new_state = copy.deepcopy(state)
    new_state.accused.append(suspect)
    new_state.player = 0
    new_state.step += 1

    return new_state


@dataclass
class MurderGameState:
    people: List[Person]
    alive: List[Person]
    dead: List[Person]
    accused: List[Person]
    killer: Person
    cost_list: List[int]
    points: int
    step: int
    player: int
    m_grid: int
    n_grid: int
    n_people: int

    def is_terminal(self):
        if self.player >= 0:
            if self.killer in set(self.accused) or self.legal_actions() is None:
                return True
            else:
                return False
        else:
            return False

    def player_return(self):
        return -len(self.alive) / len(self.people)

    def returns(self):
        return -len(self.alive) / len(self.people), len(self.alive) / len(self.people)

    def current_player(self):
        return self.player

    def is_simultaneous_node(self):  # real signature unknown; restored from __doc__
        """ is_simultaneous_node(self: pyspiel.State) -> bool """
        return False

    def is_chance_node(self):
        return self.player == -1

    def get_init_states(self):
        init_people = get_init_people(self.m_grid, self.n_grid, self.n_people)
        init_states = []
        for i in init_people:
            for j in i:
                cost_list = gen_cost_list(i, j)
                init_states.append(MurderGameState(people=i, alive=copy.copy(i), dead=list(), accused=list(), killer=j,
                                                   cost_list=cost_list, points=math.ceil(sum(cost_list) / 2) + 3,
                                                   step=-1, player=0,
                                                   m_grid=self.m_grid, n_grid=self.n_grid, n_people=self.n_people))
        return init_states

    def chance_outcomes(self):
        initial_states = self.get_init_states()
        chance_outcomes = [(i, 1 / len(initial_states)) for i in range(len(initial_states))]
        return chance_outcomes

    def child(self, action):
        if self.player == -1:
            init_states = MurderGameState.get_init_states(self)
            child_state = init_states[action]
            child_state.step += 1
            return child_state
        if self.player == 0:
            if action is None:
                child_state = copy.deepcopy(self)
                child_state.player = 1
                child_state.step += 1
                return child_state
            else:
                child_state = step_kill(self, self.people[action])
                return child_state

        if self.player == 1:
            child_state = step_accuse(self, self.people[action])
            child_state.step += 1
            return child_state

    def information_state_string(self, player_id):
        if player_id == 0:
            # print(player_id)
            # print(self)
            s = str(ObservationForKiller.from_game_state(self))
            return s
        # print(player_id)
        if player_id == 1:
            # print(player_id)
            s = str(ObservationForDetective.from_game_state(self))
            return s

    def legal_actions(self):

        if self.player == 0:
            people, victims, points, cost_list = KillerInterface.get_actions(ObservationForKiller.from_game_state(self))
            if victims is None or victims == []:
                return None
            else:
                actions = list(self.people.index(i) for i in victims)
                # print(actions)
                return actions

        if self.player == 1:
            suspects = DetectiveInterface.get_actions(ObservationForDetective.from_game_state(self))
            actions = list(self.people.index(i) for i in suspects)
            return actions

    def obs_str(self):
        if self.player == 0:
            obs = ObservationForKiller.from_game_state(self)
            return str(obs)
        if self.player == 1:
            obs = ObservationForDetective.from_game_state(self)
            return str(obs)


# The board of the game is an m * n rectangle, people are randomly distributed on the board
def get_people(m_grid: int, n_grid: int, n_people: int):
    locations = [(x, y) for x in range(m_grid) for y in range(n_grid)]
    random.shuffle(locations)
    people = [Person(locations[i]) for i in range(n_people)]
    return people


# def get_init_states(init_people):
#     init_states = []
#     for i in init_people:
#         for j in i:
#             init_states.append([{'people':i},{'alive':[]},{'dead':[]},{'killer':j}])
#
#     return init_states
# init_people = get_init_people(2,2,3)
# get_init_states(init_people)

def gen_distance_table(people):
    dist_table = np.zeros((len(people), len(people)))
    for i in range(len(people)):
        for j in range(len(people)):
            dist_table[i][j] = \
                ((people[i][0] - people[j][0]) ** 2 +
                 (people[i][1] - people[j][1]) ** 2) ** 0.5
    return dist_table


def gen_cost_list(people, killer):
    cost_list = len(people) * [0]
    dist_list = gen_distance_table(people)[people.index(killer)]
    _range = np.max(dist_list) - np.min(dist_list)
    for i in range(len(people)):
        cost_list[i] = math.ceil(((dist_list[i] - np.min(dist_list)) / _range) * len(people))
    return cost_list


class MurderGameModel:
    def __init__(self, m_grid: int, n_grid: int, n_people: int):
        self.m_grid = m_grid
        self.n_grid = n_grid
        self.n_people = n_people

        people = get_people(m_grid, n_grid, n_people)
        killer = random.choice(people)
        cost_list = gen_cost_list(people, killer)
        self.players = [0, 1]

        self.state = MurderGameState(people=people, alive=copy.copy(people), dead=list(), accused=list(), killer=killer,
                                     cost_list=cost_list, points=math.ceil(sum(cost_list) / 2) + 3, step=-1, player=-1,
                                     m_grid=m_grid,
                                     n_grid=n_grid, n_people=n_people)
        self.max_turns = 100

    def num_players(self):
        return len(self.players)

    def is_chance_node(self):
        return self.state.player == -1

    def get_init_states(self):
        init_people = get_init_people(self.m_grid, self.n_grid, self.n_people)
        init_states = []
        for i in init_people:
            for j in i:
                cost_list = gen_cost_list(i, j)
                init_states.append(MurderGameState(people=i, alive=copy.copy(i), dead=list(), accused=list(), killer=j,
                                                   cost_list=cost_list, points=math.ceil(sum(cost_list) / 2) + 3,
                                                   step=-1, player=0, m_grid=self.m_grid, n_grid=self.n_grid,
                                                   n_people=self.n_grid))

        return init_states

    # def new_initial_states(self):
    #     return

    def step_kill(self, victim: Person):

        self.state.alive.remove(victim)
        self.state.dead.append(victim)
        self.state.points -= self.state.cost_list[self.state.people.index(victim)]

    def step_accuse(self, suspect: Person):
        self.state.accused.append(suspect)

    def is_terminal(self):
        if self.state.killer in set(self.state.accused):
            score = len(self.state.alive) / len(self.state.people)
            return score
        else:
            return None

    def get_type(self):
        pass

    def new_initial_state(self, *args, **kwargs):  # real signature unknown; restored from __doc__

        return MurderGameState(people=[], alive=copy.copy([]), dead=list(), accused=list(), killer=None,
                               cost_list=[], points=0, step=-1, player=-1,
                               m_grid=self.m_grid, n_grid=self.n_grid, n_people=self.n_people)

    def copy_state(self) -> MurderGameModel:
        return copy.deepcopy(self)


@dataclass
class ObservationForKiller:
    people: List[Person]
    alive: List[Person]
    accused: List[Person]
    killer: Person
    points: int
    cost_list: []
    player: 0

    @classmethod
    def from_game_state(cls, state: MurderGameState):
        return ObservationForKiller(state.people, state.alive, state.accused, state.killer, state.points,
                                    state.cost_list, 0)


@dataclass
class ObservationForDetective:
    people: List[Person]
    alive: List[Person]
    dead: List[Person]
    accused: List[Person]
    player: 1

    @classmethod
    def from_game_state(cls, state: MurderGameState):
        return ObservationForDetective(state.people, state.alive, state.dead, state.accused, 1)


class KillerInterface:

    def get_actions(self: ObservationForKiller):
        people = self.people
        all_victims = copy.copy(self.alive)
        all_victims.remove(self.killer)
        points = self.points
        cost_list = self.cost_list

        if points <= 0 or all_victims == []:
            victims = None
        else:
            victims = []
            for i in all_victims:
                if (points - cost_list[people.index(i)]) >= 0:
                    victims.append(i)

        return people, victims, points, cost_list

        # victim_index = np.random.choice(list(range(len(self.people))), 1, replace=False, p=killer_policy)
        # cost = cost_list[int(victim_index)]
        # victim = self.people[int(victim_index)]
        #
        # return victim, cost


class DetectiveInterface:
    def get_actions(self: ObservationForDetective):
        suspects = list(set(self.alive).difference(set(self.accused)))

        return suspects
