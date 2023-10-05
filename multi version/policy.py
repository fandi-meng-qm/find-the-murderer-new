import itertools
from typing import Iterable

import numpy as np
from get_all_states import get_all_states, state_to_obs, get_all_obs, get_kd_obs
import pyspiel
from game_interface_multi import *


def child(state, action):
    """Returns a child state, handling the simultaneous node case."""
    if isinstance(action, Iterable):
        child_state = state.clone()
        child_state.apply_actions(action)
        return child_state
    else:
        return state.child(action)


def legal_actions(obs):
    if obs.player == 0:
        people, victims, points, cost_list = KillerInterface.get_actions(obs)
        if victims is None:
            return None
        else:
            actions = list(obs.people.index(i) for i in victims)
            return actions

    if obs.player == 1:
        suspects = DetectiveInterface.get_actions(obs)
        actions = list(obs.people.index(i) for i in suspects)
        return actions


def joint_action_probabilities_aux(state, policy):
    """Auxiliary function for joint_action_probabilities.

  Args:
    state: a game state at a simultaneous decision node.
    policy: policy that gives the probability distribution over the legal
      actions for each players.

  Returns:
    actions_per_player: list of list of actions for each player
    probs_per_player: list of list of probabilities do the corresponding action
     in actions_per_player for each player.
  """
    assert state.is_simultaneous_node()
    action_probs_per_player = [
        policy.action_probabilities(state, player)
        for player in range(state.get_game().num_players())
    ]
    actions_per_player = [pi.keys() for pi in action_probs_per_player]
    probs_per_player = [pi.values() for pi in action_probs_per_player]
    return actions_per_player, probs_per_player


def joint_action_probabilities(state, policy):
    """Yields action, probability pairs for a joint policy in simultaneous state.

  Args:
    state: a game state at a simultaneous decision node.
    policy: policy that gives the probability distribution over the legal
      actions for each players.

  Yields:
    (action, probability) pairs. An action is a tuple of individual
      actions for each player of the game. The probability is a single joint
      probability (product of all the individual probabilities).
  """
    actions_per_player, probs_per_player = joint_action_probabilities_aux(
        state, policy)
    for actions, probs in zip(
            itertools.product(*actions_per_player),
            itertools.product(*probs_per_player)):
        yield actions, np.prod(probs)


class Policy:
    """Base class for policies.

  A policy is something that returns a distribution over possible actions
  given a state of the world.

  Attributes:
    game: the game for which this policy applies
    player_ids: list of player ids for which this policy applies; each in the
      interval [0..game.num_players()-1].
  """

    def __init__(self, game, player_ids):
        """Initializes a policy.

    Args:
      game: the game for which this policy applies
      player_ids: list of player ids for which this policy applies; each should
        be in the range 0..game.num_players()-1.
    """
        self.game = game
        self.player_ids = player_ids

    def action_probabilities(self, state, player_id=None):
        """Returns a dictionary {action: prob} for all legal actions.

    IMPORTANT: We assume the following properties hold:
    - All probabilities are >=0 and sum to 1
    - TLDR: Policy implementations should list the (action, prob) for all legal
      actions, but algorithms should not rely on this (yet).
      Details: Before May 2020, only legal actions were present in the mapping,
      but it did not have to be exhaustive: missing actions were considered to
      be associated to a zero probability.
      For example, a deterministic state-poliy was previously {action: 1.0}.
      Given this change of convention is new and hard to enforce, algorithms
      should not rely on the fact that all legal actions should be present.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for whom we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state.
    """
        raise NotImplementedError()

    def __call__(self, state, player_id=None):
        """Turns the policy into a callable.

    Args:
      state: The current state of the game.
      player_id: Optional, the player id for whom we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      Dictionary of action: probability.
    """
        return self.action_probabilities(state, player_id)

    def to_tabular(self, states=None):
        """Returns a new `TabularPolicy` equivalent to this policy.

    Args:
      states: States of the game that will be used for the tabular policy. If
        None, then get_tabular_policy_states() method will be used to generate
        them.

    Returns:
      a TabularPolicy.
    """
        states = states or get_tabular_policy_states(self.game)
        tabular_policy = TabularPolicy(self.game, self.player_ids, states=states)
        for index, state in enumerate(tabular_policy.states):
            tabular_policy.action_probability_array[index, :] = 0
            for action, probability in self.action_probabilities(state).items():
                tabular_policy.action_probability_array[index, action] = probability
        return tabular_policy


class TabularPolicy(Policy):

    def __init__(self,
                 game,
                 players=None,
                 to_string=lambda s: s.history_str(),
                 states=None):
        """Initializes a uniform random policy for all players in the game."""
        players = sorted(players or range(game.num_players()))
        super().__init__(game, players)
        self.game_type = game.get_type()

        # Get all states in the game at which players have to make decisions unless
        # they are explicitly specified.
        # Assemble legal actions for every valid (state, player) pair, keyed by
        # information state string.
        self.states = get_all_states(game)
        obs_list = get_all_obs(game)
        self.state_lookup = {str(obs): obs_list.index(obs) for obs in obs_list}
        k, d = get_kd_obs(game)
        self.states_per_player = [k, d]

        actions = sorted(game.state.people)
        state_action = np.zeros([len(obs_list), len(actions)])
        for i in range(len(obs_list)):
            actions = legal_actions(obs_list[i])
            if actions is not None:
                state_action[i][actions] = 1 / len(actions)
        self.action_probability_array = state_action

        legal_actions_list = []
        state_in_list = self.states

    def _state_key(self, state, player):
        """Returns the key to use to look up this (state, player) pair."""
        obs = state_to_obs(state)
        return str(obs)

    def action_probabilities(self, state, player_id=None):
        """Returns an {action: probability} dict, covering all legal actions."""
        legal_actions = (
            state.legal_actions()
            if player_id is None else state.legal_actions(player_id))
        if not legal_actions:
            return {0: 1.0}
        probability = self.policy_for_key(self._state_key(state, player_id))
        return {action: probability[action] for action in legal_actions}

    def state_index(self, state):
        """Returns the index in the TabularPolicy associated to `state`."""
        return self.state_lookup[self._state_key(state, state.current_player())]

    def policy_for_key(self, key):
        """Returns the policy as a vector given a state key string.

    Args:
      key: A key for the specified state.

    Returns:
      A vector of probabilities, one per action. This is a slice of the
      backing policy array, and so slice or index assignment will update the
      policy. For example:
      ```
      tabular_policy.policy_for_key(s)[:] = [0.1, 0.5, 0.4]
      ```
    """

        return self.action_probability_array[self.state_lookup[key]]

    def to_dict(self):
        """Returns a single dictionary representing the tabular policy.

    Returns:
      A dictionary of string keys to lists of (action, prob) pairs.
    """
        policy_dict = {}
        num_actions = self.action_probability_array.shape[1]
        for infostate_key, index in self.state_lookup.items():
            probs = self.action_probability_array[index]
            actions_and_probs = [(a, probs[a]) for a in range(num_actions)]
            policy_dict[infostate_key] = actions_and_probs
        return policy_dict

    def __copy__(self, copy_action_probability_array=True):
        """Returns a shallow copy of self.

    Most class attributes will be pointers to the copied object's attributes,
    and therefore altering them could lead to unexpected behavioural changes.
    Only action_probability_array is expected to be modified.

    Args:
      copy_action_probability_array: Whether to also include
        action_probability_array in the copy operation.

    Returns:
      Copy.
    """
        result = TabularPolicy.__new__(TabularPolicy)
        result.state_lookup = self.state_lookup
        result.game_type = self.game_type
        result.legal_actions_mask = self.legal_actions_mask
        result.state_in = self.state_in
        result.state_lookup = self.state_lookup
        result.states_per_player = self.states_per_player
        result.states = self.states
        result.game = self.game
        result.player_ids = self.player_ids
        if copy_action_probability_array:
            result.action_probability_array = np.copy(self.action_probability_array)
        return result

    def copy_with_noise(self,
                        alpha=0.0,
                        beta=0.0,
                        random_state=np.random.RandomState()):
        """Returns a copy of this policy perturbed with noise.

    Generates a new random distribution using a softmax on normal random
    variables with temperature beta, and mixes it with the old distribution
    using 1-alpha * old_distribution + alpha * random_distribution.
    Args:
      alpha: Parameter characterizing the mixture amount between new and old
        distributions. Between 0 and 1.
        alpha = 0: keep old table.
        alpha = 1: keep random table.
      beta: Temperature of the softmax. Makes for more extreme policies.
      random_state: A numpy `RandomState` object. If not provided, a shared
        random state will be used.

    Returns:
      Perturbed copy.
    """
        copied_instance = self.__copy__(False)
        probability_array = self.action_probability_array
        noise_mask = random_state.normal(size=probability_array.shape)
        noise_mask = np.exp(beta * noise_mask) * self.legal_actions_mask
        noise_mask = noise_mask / (np.sum(noise_mask, axis=1).reshape(-1, 1))
        copied_instance.action_probability_array = (
                                                           1 - alpha) * probability_array + alpha * noise_mask
        return copied_instance


class UniformRandomPolicy(Policy):
    """Policy where the action distribution is uniform over all legal actions.

  This is computed as needed, so can be used for games where a tabular policy
  would be unfeasibly large, but incurs a legal action computation every time.
  """

    def __init__(self, game):
        """Initializes a uniform random policy for all players in the game."""
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)

    def action_probabilities(self, state, player_id=None):
        """Returns a uniform random policy for a player in a state.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for which we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state. This will contain all legal actions, each with the same
      probability, equal to 1 / num_legal_actions.
    """
        legal_actions = (
            state.legal_actions()
            if player_id is None else state.legal_actions(player_id))
        if not legal_actions:
            return {0: 1.0}
        probability = 1 / len(legal_actions)
        return {action: probability for action in legal_actions}


class FirstActionPolicy(Policy):
    """A policy that always takes the lowest-numbered legal action."""

    def __init__(self, game):
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)

    def action_probabilities(self, state, player_id=None):
        legal_actions = (
            state.legal_actions()
            if player_id is None else state.legal_actions(player_id))
        if not legal_actions:
            return {0: 1.0}
        min_action = min(legal_actions)
        return {
            action: 1.0 if action == min_action else 0.0 for action in legal_actions
        }


def get_tabular_policy_states(game):
    """Returns the states of the game for a tabular policy."""
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD:
        # TODO(perolat): We use s.observation_string(DEFAULT_MFG_PLAYER) here as the
        # number of history is exponential on the depth of the MFG. What we really
        # need is a representation of the state. For many player Mean Field games,
        # the state will be (x0, x1, x2, ..., xn) and the observation_string(0) will
        # output the string of x0. In that case we would need something like
        # str([observation_string(i) for i in range(num_player)])
        to_string = lambda s: s.observation_string(pyspiel.PlayerId.
                                                   DEFAULT_PLAYER_ID)
    else:
        to_string = lambda s: s.history_str()
    return get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False,
        include_mean_field_states=False,
        to_string=to_string)


def tabular_policy_from_callable(game, callable_policy, players=None):
    """Converts a legacy callable policy into a TabularPolicy.

  Recommendation - instead of using this to convert your policy for evaluation
  purposes, work directly with a `TabularPolicy` if possible.
  Second choice - work with a `Policy` class and call `to_tabular` as needed.

  Args:
    game: The game for which we want a TabularPolicy.
    callable_policy: A callable: state -> action probabilities dict or list.
    players: List of players this policy applies to. If `None`, applies to all
      players.

  Returns:
    A TabularPolicy that materializes the callable policy.
  """
    tabular_policy = TabularPolicy(game, players)
    for state_index, state in enumerate(tabular_policy.states):
        action_probabilities = dict(callable_policy(state))
        infostate_policy = [
            action_probabilities.get(action, 0.)
            for action in range(game.num_distinct_actions())
        ]
        tabular_policy.action_probability_array[state_index, :] = infostate_policy
    return tabular_policy


def pyspiel_policy_to_python_policy(game, pyspiel_tabular_policy, players=None):
    """Converts a pyspiel.TabularPolicy to a TabularPolicy.

  Args:
    game: The OpenSpiel game.
    pyspiel_tabular_policy: Pyspiel tabular policy to copy from.
    players: List of integer player ids to copy policy from. For example,
      `players=[0]` will only copy player 0's policy over into the python policy
      (the other player's policies will be undefined). Default value of `None`
      will copy all players' policies.

  Returns:
    python_policy
  """
    policy = TabularPolicy(game, players=players)
    for item in pyspiel_tabular_policy.policy_table().items():
        info_state_str, actions_probs = item
        # If requested, only populate a policy for particular players.
        if players is not None and info_state_str not in policy.state_lookup:
            continue
        state_policy = policy.policy_for_key(info_state_str)
        for action, prob in actions_probs:
            state_policy[action] = prob
    return policy


def python_policy_to_pyspiel_policy(python_tabular_policy):
    """Converts a TabularPolicy to a pyspiel.TabularPolicy."""
    infostates_to_probabilities = dict()
    for infostate, index in python_tabular_policy.state_lookup.items():
        probs = python_tabular_policy.action_probability_array[index]
        legals = python_tabular_policy.legal_actions_mask[index]

        action_probs = []
        for action, (prob, is_legal) in enumerate(zip(probs, legals)):
            if is_legal == 1:
                action_probs.append((action, prob))
        infostates_to_probabilities[infostate] = action_probs
    return pyspiel.TabularPolicy(infostates_to_probabilities)


def python_policies_to_pyspiel_policies(policies):
    """Same conversion as above (list version).

  Args:
    policies: a list of python.TabularPolicy

  Returns:
    a list of pyspiel.TabularPolicy.
  """
    return [python_policy_to_pyspiel_policy(p) for p in policies]


def merge_tabular_policies(tabular_policies, game):
    """Merges n_player policies into single joint policy.

  Missing states are filled with a valid uniform policy.

  Args:
    tabular_policies: List of python TabularPolicy (one for each player).
    game: The game corresponding to the resulting TabularPolicy.

  Returns:
    merged_policy: A TabularPolicy with each player i's policy taken from the
      ith joint_policy.
  """
    if len(tabular_policies) != game.num_players():
        raise ValueError("len(tabular_policies) != num_players: %d != %d" %
                         (len(tabular_policies), game.num_players()))
    merged_policy = TabularPolicy(game)
    for p, p_states in enumerate(merged_policy.states_per_player):
        for p_state in p_states:
            to_index = merged_policy.state_lookup[p_state]
            # Only copy if the state exists, otherwise fall back onto uniform.
            if p_state in tabular_policies[p].state_lookup:
                from_index = tabular_policies[p].state_lookup[p_state]
                merged_policy.action_probability_array[to_index] = (
                    tabular_policies[p].action_probability_array[from_index])
    return merged_policy
