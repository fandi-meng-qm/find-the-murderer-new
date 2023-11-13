import copy
import random
import re
import numpy as np
from open_spiel.python import rl_agent


# input: observations string output: MurderGame and MurderState
def deserialize_state(observations):
    from new_deductive_1d_game.version3.test import game
    state = game.new_initial_state()
    history = observations.split("[State]\nhistory=")[1].split("\n")[0]
    history_list = [int(i) for i in re.split(',|:', history)]
    # history to state
    player_list = history_list[::2]
    action_list = history_list[1::2]
    for i in range(len(player_list)):
        state = state.child(action_list[i])
    return game, state


class ISMCTSAgent(rl_agent.AbstractAgent):
    def __init__(self, player_id, num_actions, bot):
        assert num_actions > 0
        self._player_id = player_id
        self._bot = bot
        self._num_actions = num_actions

    def step(self, time_step, is_evaluation=False):
        # If it is the end of the episode, don't select an action.
        if time_step.last():
            return

        assert "serialized_state" in time_step.observations

        # print(time_step.observations["serialized_state"])
        _, state = deserialize_state(time_step.observations["serialized_state"])
        # Call the MCTS bot's step to get the action.
        probs = np.zeros(self._num_actions)
        action = self._bot.step(state)
        probs[action] = 1.0

        return rl_agent.StepOutput(action=action, probs=probs)


class RandomAgent(rl_agent.AbstractAgent):
    def __init__(self, player_id, num_actions, bot):
        assert num_actions > 0
        self._player_id = player_id
        self._num_actions = num_actions
        self._bot = bot

    def step(self, time_step, is_evaluation=False):
        # If it is the end of the episode, don't select an action.
        if time_step.last():
            return

        assert "serialized_state" in time_step.observations

        # print(time_step.observations["serialized_state"])
        _, state = deserialize_state(time_step.observations["serialized_state"])
        legal_actions = time_step.observations["legal_actions"]
        probs = np.zeros(self._num_actions)
        action = random.choice(legal_actions[0])
        # print(state.init_actions[action])
        probs[action] = 1.0

        return rl_agent.StepOutput(action=action, probs=probs)

class BetterRandomAgent(rl_agent.AbstractAgent):
    def __init__(self, player_id, num_actions, bot):
        assert num_actions > 0
        self._player_id = player_id
        self._num_actions = num_actions
        self._bot = bot

    def step(self, time_step, is_evaluation=False):
        # If it is the end of the episode, don't select an action.
        if time_step.last():
            return

        assert "serialized_state" in time_step.observations

        # print(time_step.observations["serialized_state"])
        _, state = deserialize_state(time_step.observations["serialized_state"])
        legal_actions = time_step.observations["legal_actions"]
        probs = np.zeros(self._num_actions)
        action_list=copy.deepcopy(legal_actions[0])
        for a in action_list:
            if {index for index, value in enumerate(state.init_actions[a]) if value == 1} <= \
                    {index for index, value in enumerate(state.information_state) if value == 0}:
                action_list.remove(a)
        action = random.choice(action_list)
        probs[action] = 1.0

        return rl_agent.StepOutput(action=action, probs=probs)

class OptimalAgent(rl_agent.AbstractAgent):
    def __init__(self, player_id, num_actions, bot):
        assert num_actions > 0
        self._player_id = player_id
        self._num_actions = num_actions
        self._bot = bot

    def step(self, time_step, is_evaluation=False):
        # If it is the end of the episode, don't select an action.
        if time_step.last():
            return

        assert "serialized_state" in time_step.observations

        # print(time_step.observations["serialized_state"])
        _, state = deserialize_state(time_step.observations["serialized_state"])
        legal_actions = time_step.observations["legal_actions"]
        probs = np.zeros(self._num_actions)
        changed_count = 0
        action_binary=copy.deepcopy(state.information_state)
        for i in range(len(state.information_state)):
            if changed_count >= sum(state.information_state)/2:
                break
            if state.information_state[i] == 1:
                action_binary[i] = 0
                changed_count += 1
        action = state.init_actions.index(tuple(action_binary))
        probs[action] = 1.0

        return rl_agent.StepOutput(action=action, probs=probs)
