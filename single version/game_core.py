from __future__ import annotations

import copy
import dataclasses
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import partial
from typing import NamedTuple, Set, Tuple, List


# Using namedtuple to represent people,
# but in Python version 3.9 NamedTuple is no longer a class, which needs to be noted.
class Person(NamedTuple):
    id: int
    location: Tuple[int, int]


@dataclass
class MurderGameState:
    people: Set[Person]
    alive: Set[Person]
    dead: Set[Person]
    accused: Set[Person]
    killer: Person
    move_no: int = 0


# The board of the game is an m * n rectangle, people are randomly distributed on the board
def get_people(m_grid: int, n_grid: int, n_people: int):
    locations = [(x, y) for x in range(m_grid) for y in range(n_grid)]
    random.shuffle(locations)
    people = [Person(i, locations[i]) for i in range(n_people)]
    return people


class MurderGameModel:
    def __init__(self, m_grid: int, n_grid: int, n_people: int):
        self.m_grid = m_grid
        self.n_grid = n_grid
        self.n_people = n_people
        people = get_people(m_grid, n_grid, n_people)
        killer = random.choice(people)
        self.state = MurderGameState(people=set(people), alive=set(people), dead=set(), accused=set(), killer=killer)
        self.max_turns = 100

    def step_kill(self, victim: Person):
        self.state.alive.discard(victim)
        self.state.dead.add(victim)

    def step_accuse(self, suspect: Person):
        self.state.accused.add(suspect)

    def is_terminal(self):
        if {self.state.killer} & self.state.accused == set():
            return None
        else:
            score = len(self.state.alive) / len(self.state.people)
            return score

    # In this case, the detective cannot repeat the charges or accuse the deceased.
    def n_actions(self) -> int:
        if self.killer_turn():
            return len(self.state.alive - {self.state.killer})
        else:
            return len(self.state.alive - self.state.accused)

    def killer_turn(self):
        # alternate  between killer and detective moves, starting with killer
        return 0 == self.state.move_no % 2

    def act(self, action: Person) -> None:
        if self.killer_turn():
            self.step_kill(action)
        else:
            self.step_accuse(action)
        self.state.move_no += 1

    def copy_state(self) -> MurderGameModel:
        return copy.deepcopy(self)

    def get_actions(self):
        if self.killer_turn():
            return list(self.state.alive)
        else:
            return list(self.state.alive - self.state.accused)
