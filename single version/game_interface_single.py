from abc import abstractmethod, ABC
from dataclasses import dataclass
import random
from typing import Set, List
import numpy as np
from game_core import Person, MurderGameState


@dataclass
class ObservationForKiller:
    people: Set[Person]
    alive: Set[Person]
    killer: Person

    @classmethod
    def from_game_state(cls, state: MurderGameState):
        return ObservationForKiller(state.people, state.alive, state.killer)


@dataclass
class ObservationForDetective:
    people: Set[Person]
    alive: Set[Person]
    dead: Set[Person]
    accused: Set[Person]

    @classmethod
    def from_game_state(cls, state: MurderGameState):
        return ObservationForDetective(state.people,state.alive, state.dead, state.accused)


class KillerInterface:
    def get_action(self: ObservationForKiller, killer_policy: List[float]) -> Person:
        victim_id = np.random.choice(list(range(len(self.people))), 1, replace=False, p=killer_policy)
        # the alive is the killer's possible actions
        for person in self.alive:
            if person.id == victim_id:
                victim = person
        return victim


class DetectiveInterface:
    def get_action(self: ObservationForDetective, detective_policy: List[float]) -> Person:
        suspect_id = np.random.choice(list(range(len(self.people))), 1, replace=False, p=detective_policy)
        # detective's possible actions is  alive - accused
        for person in self.alive - self.accused:
            if person.id == suspect_id:
                suspect = person
        return suspect

