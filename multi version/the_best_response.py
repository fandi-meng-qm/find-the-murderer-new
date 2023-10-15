# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Computes a Best-Response policy.

The goal if this file is to be the main entry-point for BR APIs in Python.

TODO(author2): Also include computation using the more efficient C++
`TabularBestResponse` implementation.
"""

import collections
import itertools

import numpy as np

from open_spiel.python import games  # pylint:disable=unused-import
from open_spiel.python import policy as openspiel_policy
# from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import noisy_policy
from open_spiel.python.algorithms import policy_utils
import pyspiel

from get_all_states import get_all_states


def _memoize_method(key_fn=lambda x: x):
    """Memoize a single-arg instance method using an on-object cache."""

    def memoizer(method):
        cache_name = "cache_" + method.__name__

        def wrap(self, arg):
            key = key_fn(arg)
            cache = vars(self).setdefault(cache_name, {})
            if key not in cache:
                cache[key] = method(self, arg)
            return cache[key]

        return wrap

    return memoizer


def compute_states_and_info_states_if_none(game,
                                           all_states=None,
                                           state_to_information_state=None):
    """Returns all_states and/or state_to_information_state for the game.

  To recompute everything, pass in None for both all_states and
  state_to_information_state. Otherwise, this function will use the passed in
  values to reconstruct either of them.

  Args:
    game: The open_spiel game.
    all_states: The result of calling get_all_states.get_all_states. Cached for
      improved performance.
    state_to_information_state: A dict mapping state.history_str() to
      state.information_state for every state in the game. Cached for improved
      performance.
  """
    if all_states is None:
        all_states = get_all_states.get_all_states(
            game,
            depth_limit=-1,
            include_terminals=False,
            include_chance_states=False)

    if state_to_information_state is None:
        state_to_information_state = {
            state: all_states[state].information_state_string()
            for state in all_states
        }

    return all_states, state_to_information_state


class BestResponsePolicy(openspiel_policy.Policy):
    """Computes the best response to a specified strategy."""

    def __init__(self,
                 game,
                 player_id,
                 policy,
                 root_state=None,
                 cut_threshold=0.0):
        """Initializes the best-response calculation.

    Args:
      game: The game to analyze.
      player_id: The player id of the best-responder.
      policy: A `policy.Policy` object.
      root_state: The state of the game at which to start analysis. If `None`,
        the game root state is used.
      cut_threshold: The probability to cut when calculating the value.
        Increasing this value will trade off accuracy for speed.
    """
        self._num_players = game.num_players()
        self._player_id = player_id
        self._policy = policy
        if root_state is None:
            root_state = game.new_initial_state()
        self._root_state = root_state
        self.infosets = self.info_sets(root_state)

        self._cut_threshold = cut_threshold

    def info_sets(self, state):
        """Returns a dict of infostatekey to list of (state, cf_probability)."""
        infosets = collections.defaultdict(list)
        for s, p in self.decision_nodes(state):
            infosets[s.information_state_string(self._player_id)].append((s, p))
        return dict(infosets)

    def decision_nodes(self, parent_state):
        """Yields a (state, cf_prob) pair for each descendant decision node."""
        if not parent_state.is_terminal():
            if (parent_state.current_player() == self._player_id or
                    parent_state.is_simultaneous_node()):
                yield (parent_state, 1.0)
            if self.transitions(parent_state) is not None:
                for action, p_action in self.transitions(parent_state):
                    for state, p_state in self.decision_nodes(
                            openspiel_policy.child(parent_state, action)):
                        yield (state, p_state * p_action)

    def joint_action_probabilities_counterfactual(self, state):
        """Get list of action, probability tuples for simultaneous node.

    Counterfactual reach probabilities exclude the best-responder's actions,
    the sum of the probabilities is equal to the number of actions of the
    player _player_id.
    Args:
      state: the current state of the game.

    Returns:
      list of action, probability tuples. An action is a tuple of individual
        actions for each player of the game.
    """
        actions_per_player, probs_per_player = (
            openspiel_policy.joint_action_probabilities_aux(state, self._policy))
        probs_per_player[self._player_id] = [
            1.0 for _ in probs_per_player[self._player_id]
        ]
        return [(list(actions), np.prod(probs)) for actions, probs in zip(
            itertools.product(
                *actions_per_player), itertools.product(*probs_per_player))]

    def transitions(self, state):
        """Returns a list of (action, cf_prob) pairs from the specified state."""
        if state.current_player() == self._player_id:
            # Counterfactual reach probabilities exclude the best-responder's actions,
            # hence return probability 1.0 for every action.
            return [(action, 1.0) for action in state.legal_actions()]
        elif state.is_chance_node():
            return state.chance_outcomes()
        elif state.is_simultaneous_node():
            return self.joint_action_probabilities_counterfactual(state)
        else:
            return list(self._policy.action_probabilities(state).items())

    # @_memoize_method(key_fn=lambda state: state.obs_str())
    def value(self, state):
        """Returns the value of the specified state to the best-responder."""
        if state.is_terminal():
            if state.current_player() == self._player_id:
                return -state.player_return()
            else:
                return state.player_return()
        elif (state.current_player() == self._player_id or
              state.is_simultaneous_node()):
            action = self.best_response_action(
                state.information_state_string(self._player_id))
            return self.q_value(state, action)
        else:
            return sum(p * self.q_value(state, a)
                       for a, p in self.transitions(state)
                       if p > self._cut_threshold)

    def q_value(self, state, action):
        """Returns the value of the (state, action) to the best-responder."""
        if state.is_simultaneous_node():

            def q_value_sim(sim_state, sim_actions):
                child = sim_state.clone()
                # change action of _player_id
                sim_actions[self._player_id] = action
                child.apply_actions(sim_actions)
                return self.value(child)

            actions, probabilities = zip(*self.transitions(state))
            return sum(p * q_value_sim(state, a)
                       for a, p in zip(actions, probabilities / sum(probabilities))
                       if p > self._cut_threshold)
        else:
            return self.value(state.child(action))

    # @_memoize_method()
    def best_response_action(self, infostate):
        """Returns the best response for this information state."""

        infoset = self.infosets[infostate]
        # Get actions from the first (state, cf_prob) pair in the infoset list.
        # Return the best action by counterfactual-reach-weighted state-value.

        return max(
            infoset[0][0].legal_actions(),
            key=lambda a: sum(cf_p * self.q_value(s, a) for s, cf_p in infoset))

    def action_probabilities(self, state, player_id=None):
        """Returns the policy for a player in a state.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for whom we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state.
    """
        if player_id is None:
            if state.is_simultaneous_node():
                player_id = self._player_id
            else:
                player_id = state.current_player()
        return {
            self.best_response_action(state.information_state_string(player_id)): 1
        }
