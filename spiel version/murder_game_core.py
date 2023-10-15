from __future__ import annotations
import copy
import itertools
import math
from typing import List, NamedTuple, Tuple
from dataclasses import dataclass
import pyspiel

import numpy as np

import enum


class MurderParams:
    m_grid: int = 1
    n_grid: int = 3
    n_people: int = 3


_NUM_PLAYERS = 2
# _DECK = frozenset([0, 1, 2])
_GAME_TYPE = pyspiel.GameType(
    short_name="python_Murder_Mystery",
    long_name="Python Murder Mystery",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False)


class MurderPlayer(enum.IntEnum):
    KILLER = 0
    DETECTIVE = 1


class MurderGame(pyspiel.Game):
    def __init__(self, params=None, game_params: MurderParams = None) -> None:
        game_params = game_params or MurderParams()
        self.game_params = game_params
        n_actions = game_params.n_people
        game_info = pyspiel.GameInfo(
            num_distinct_actions=n_actions,
            max_chance_outcomes=math.comb(game_params.m_grid * game_params.n_grid, game_params.n_people) *
                                game_params.n_people,
            num_players=2,
            min_utility=-1,
            max_utility=1,
            utility_sum=0.0,
            max_game_length=1000
        )
        super().__init__(_GAME_TYPE, game_info, params or dict())

    def new_initial_state(self, game_params=None) -> MurderState:
        """Returns a state corresponding to the start of a game."""
        return MurderState(self, self.game_params)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return MurderObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            self.game_params)


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


def gen_distance_table(people) -> np.array:
    dist_table = np.zeros((len(people), len(people)))
    for i in range(len(people)):
        for j in range(len(people)):
            dist_table[i][j] = \
                ((people[i][0] - people[j][0]) ** 2 +
                 (people[i][1] - people[j][1]) ** 2) ** 0.5
    return dist_table


def gen_cost_list(people, killer) -> list:
    cost_list = len(people) * [0]
    dist_list = gen_distance_table(people)[people.index(killer)]
    _range = np.max(dist_list) - np.min(dist_list)
    for i in range(len(people)):
        cost_list[i] = math.ceil(((dist_list[i] - np.min(dist_list)) / _range) * len(people))
    return cost_list


def get_init_states(params) -> list:
    init_people = get_init_people(params.m_grid, params.n_grid, params.n_people)
    init_states = []
    for i in init_people:
        for j in i:
            cost_list = gen_cost_list(i, j)
            game = MurderGame(game_params=params)
            state = MurderState(game)
            # state.params= params
            # print(state.params)
            state.people = i
            state.alive = i
            state.killer = j
            state.dead = []
            state.accused = []
            state.cost_list = cost_list
            state.points = math.ceil(sum(cost_list) / 2) + 3
            state.step = 1
            init_states.append(state)
    return init_states


class MurderState(pyspiel.State):
    """A python version of the state."""

    def __init__(self, game, params: MurderParams = None):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.params = params
        self.people = list()
        self.alive = list()
        self.dead = list()
        self.accused = list()
        self.killer = Person()
        self.cost_list = list()
        self.points = math.ceil(sum(self.cost_list) / 2) + 3
        self.step = 0

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        # in the current version only the first move is a chance move - the one which decides who is the killer
        elif self.step == 0:
            return pyspiel.PlayerId.CHANCE
        else:
            return self.to_play()

    def to_play(self) -> int:
        return 1 - (self.step % 2)

    def _legal_actions(self, player) -> List[int] or None:
        """Returns a list of legal actions"""
        # check this is not the chance player, for some reason that is handled separately
        assert player >= 0
        if self.current_player() == MurderPlayer.KILLER:
            people, victims, points, cost_list = KillerInterface.get_actions(ObservationForKiller.from_game_state(self))
            actions = list(self.people.index(i) for i in victims)
                # print(actions)
            return actions

        if self.current_player() == MurderPlayer.DETECTIVE:
            suspects = DetectiveInterface.get_actions(ObservationForDetective.from_game_state(self))
            actions = list(self.people.index(i) for i in suspects)
            return actions

    def is_chance_node(self):
        if self.step ==0:
            return True

    def chance_outcomes(self):
        assert self.step == 0
        initial_states = get_init_states(self.params)
        chance_outcomes = [(i, 1 / len(initial_states)) for i in range(len(initial_states))]
        return chance_outcomes

    def _kill_action(self, victim: Person) -> None:
        assert self.current_player() == MurderPlayer.KILLER
        self.alive.remove(victim)
        self.dead.append(victim)
        self.points -= self.cost_list[self.people.index(victim)]

    def _accuse_action(self, suspect: Person) -> None:
        assert self.current_player() == MurderPlayer.DETECTIVE
        self.accused.append(suspect)

    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            assert self.step == 0
            self.init_state = action

        else:
            if self.current_player() == MurderPlayer.KILLER:
                if action is None:
                    pass
                else:
                    self._kill_action(self.people[action])
            else:
                self._accuse_action(self.people[action])
            self.step += 1

    def n_actions(self) -> int:
        if self.step == 0:
            return len(get_init_states(self.params))
        else:
            return len(self._legal_actions(self.current_player()))

    def clone(self):
        cp = super().clone()
        return cp

    def is_terminal(self):
        people, victims, points, cost_list = KillerInterface.get_actions(ObservationForKiller.from_game_state(self))
        if victims is None or victims == []:
            return True
        elif self.killer in set(self.accused):
            return True
        else:
            return False

    def score(self) -> float:
        if not self.is_terminal():
            return 0
        else:
            return len(self.alive) / len(self.people)

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        if not self.is_terminal():
            return [0, 0]
        else:
            score = self.score()
            return [score, -score]


class MurderMysteryVariationsObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig, params: MurderParams):
        """Initializes an empty observation tensor."""
        if params == None:
            raise ValueError(f"Observation needs params for setup; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation will depend on the player
        # The tensor comprises the following pieces given N players
        # Each set describes the killer identity (N), the alive people (N), the dead people(N), the accused (N)
        # The one-hot coding for killer has N elements, all will be zero if the killer is not assigned yet
        self.params = params
        size = 4 * params.n_people
        shape = (size)
        self.tensor = np.zeros(size, np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def _code_set(self, people: set, n: int) -> List[int]:
        return [1 if x in people else 0 for x in range(n)]

    def _code_killer(self, killer: int, n: int) -> List[int]:
        return [1 if x == killer else 0 for x in range(n)]

    def set_from(self, state: MurderState, player: int):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        obs.fill(0)
        if player == MurderPlayer.DETECTIVE:
            killer_list = self._code_killer(-1, self.params.n_people)
        else:
            killer_list = self._code_killer(state.killer, self.params.n_people)
        alive_list = self._code_set(state.alive, self.params.n_people)
        dead_list = self._code_set(state.alive, self.params.n_people)
        accused_list = self._code_set(state.alive, self.params.n_people)
        all_list = [*killer_list, *alive_list, *dead_list, *accused_list]
        for i, x in enumerate(all_list):
            obs[i] = x
        # print("All list: ", all_list)
        # print("obs: ", obs)

    def string_from(self, state: MurderState, player: int):
        """Observation of `state` from the PoV of `player`, as a string."""
        if player == MurderPlayer.KILLER:
            return f"k_{state.killer}-a_{state.alive}"
        else:
            return f"a_{state.alive}_d{state.dead}_s{state.accused}"


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
    def from_game_state(cls, state: MurderState):
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
    def from_game_state(cls, state: MurderState):
        return ObservationForDetective(state.people, state.alive, state.dead, state.accused, 1)


class KillerInterface:

    def get_actions(self: ObservationForKiller):
        people = self.people
        all_victims = copy.deepcopy(self.alive)
        if self.killer in set(all_victims):
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

