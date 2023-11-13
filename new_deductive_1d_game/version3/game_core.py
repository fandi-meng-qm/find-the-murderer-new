from __future__ import annotations

import itertools
from typing import List, Tuple

import pyspiel

import numpy as np


class MurderParams:
    def __init__(self, m_grid, n_grid, n_people):
        self.m_grid = m_grid
        self.n_grid = 1
        self.n_people = 1


_NUM_PLAYERS = 1
# _DECK = frozenset([0, 1, 2])
_GAME_TYPE = pyspiel.GameType(
    short_name="1d_simple_game",
    long_name="1d simple game",
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


class MurderGame(pyspiel.Game):
    def __init__(self, params=None, game_params: MurderParams = None) -> None:
        game_params = game_params or MurderParams()
        self.game_params = game_params
        n_actions = 2**game_params.m_grid
        game_info = pyspiel.GameInfo(
            num_distinct_actions=n_actions,
            max_chance_outcomes=game_params.m_grid,
            num_players=1,
            min_utility=-1.0,
            max_utility=1.0,
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


class MurderState(pyspiel.State):
    """A python version of the state."""

    def __init__(self, game, params: MurderParams = None):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.params = params
        self.people = tuple()
        self.step = 0
        self.information_state = [1] * self.params.m_grid
        self.init_actions = list(itertools.product([0, 1], repeat=self.params.m_grid))

    def current_player(self) -> pyspiel.PlayerId or int:
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        # in the current version only the first move is a chance move - the one which decides who is the killer
        elif self.step == 0:
            return pyspiel.PlayerId.CHANCE
        else:
            return 0

    def _legal_actions(self, player) -> List[int] or None:
        """Returns a list of legal actions"""
        # check this is not the chance player, for some reason that is handled separately
        assert player >= 0
        actions = list(range(len(self.init_actions)))
        return actions

    def is_chance_node(self) -> bool:
        if self.step == 0:
            return True

    def chance_outcomes(self) -> List[tuple]:
        assert self.step == 0
        chance_outcomes = [(i, 1 / self.params.m_grid) for i in range(self.params.m_grid)]
        return chance_outcomes

    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            assert self.step == 0
            self.people = (action, 0)
            self.step = 1
        else:
            if self.people[0] in [index for index, value in enumerate(self.init_actions[action]) if value == 1]:
                for index in [index for index, value in enumerate(self.init_actions[action]) if value == 0]:
                    self.information_state[index] = 0
            else:
                for index in [index for index, value in enumerate(self.init_actions[action]) if value == 1]:
                    self.information_state[index] = 0
            self.step += 1

    def n_actions(self) -> int:
        if self.step == 0:
            return self.params.m_grid
        else:
            return 2**self.params.m_grid

    def clone(self):
        cp = super().clone()
        return cp

    def is_terminal(self) -> bool:
        if self.step > 0:
            if sum(self.information_state) == 1:
                return True
            else:
                return False
        else:
            return False

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        if not self.is_terminal():
            return 0
        else:
            return [-self.step]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return f"m_grid={self.params.m_grid},n_grid={self.params.n_grid},n_people={self.params.n_people}," \
               f"people={self.people},step={self.step},information_state={self.information_state},"


class MurderObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig, params: MurderParams):
        """Initializes an empty observation tensor."""
        if params == None:
            raise ValueError(f"Observation needs params for setup; passed {params}")
        self.params = params
        size = params.m_grid * params.n_grid
        shape = (params.m_grid, params.n_grid)
        self.tensor = np.zeros(size, np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state: MurderState, player: int):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        obs.fill(0)
        all_array = np.array(state.information_state)
        for i, x in enumerate(all_array):
            obs[i] = x
        # print("All list: ", all_list)
        # print("obs: ", obs)

    def string_from(self, state: MurderState, player: int):
        """Observation of `state` from the PoV of `player`, as a string."""
        return f"information_state{state.information_state}"


pyspiel.register_game(_GAME_TYPE, MurderGame)
