from abc import abstractmethod, ABC
from dataclasses import dataclass
import copy
import random
from typing import Set, List
import numpy as np
from game_core import Person, MurderGameState


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
        return ObservationForKiller(state.people, state.alive, state.accused, state.killer, state.points, state.cost_list, 0)


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

        if points <=0 or all_victims==[]:
            victims = None
        else:
            victims=[]
            for i in all_victims:
                if (points - cost_list[people.index(i)]) >= 0:
                    victims.append(i)
            if len(victims)==0:
                victims = None
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

        # suspect_index = np.random.choice(list(range(len(self.people))), 1, replace=False, p=detective_policy)
        # suspect = self.people[int(suspect_index)]
        # return suspect
