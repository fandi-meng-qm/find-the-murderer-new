
# Copyright 2022 DeepMind Technologies Limited
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
"""Test the IS-MCTS Agent."""
import re
import numpy as np
from open_spiel.python import rl_environment
import ismcts
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import mcts_agent
from game_core import MurderGame,MurderParams
from open_spiel.python import rl_agent

params = MurderParams(4, 1, 1)
game = MurderGame(game_params=params)

def deserialize_state(observations):
    state = game.new_initial_state()
    history = observations.split("[State]\nhistory=")[1].split("\n")[0]
    history_list = [int(i) for i in re.split(',|:', history)]
    # history to state
    player_list = history_list[::2]
    action_list = history_list[1::2]
    for i in range(len(player_list)):
        if player_list[i] ==-1:
            state=state.child(action_list[i])
        elif player_list[i]==0:
            state=state.child(action_list[i])
    return game, state

class MCTSAgent(rl_agent.AbstractAgent):
  """MCTS agent class.

  Important note: this agent requires the environment to provide the full state
  in its TimeStep objects. Hence, the environment must be created with the
  use_full_state flag set to True, and the state must be serializable.
  """

  def __init__(self, player_id, num_actions, mcts_bot, name="mcts_agent"):
    assert num_actions > 0
    self._player_id = player_id
    self._mcts_bot = mcts_bot
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    assert "serialized_state" in time_step.observations

    # print(time_step.observations["serialized_state"])
    _, state = deserialize_state(time_step.observations["serialized_state"])
    # print(state)
    # Call the MCTS bot's step to get the action.
    probs = np.zeros(self._num_actions)
    action = self._mcts_bot.step(state)
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)


def test():
    env = rl_environment.Environment(game, include_full_state=True)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]
    # Create the MCTS bot. Both agents can share the same bot in this case since
    # there is no state kept between searches. See mcts.py for more info about
    # the arguments.
    ismcts_bot = ismcts.ISMCTSBot(
        game=env.game,
        uct_c=1.5,
        max_simulations=200,
        evaluator=mcts.RandomRolloutEvaluator())

    agents = [
        MCTSAgent(
            player_id=idx, num_actions=num_actions, mcts_bot=ismcts_bot)
        for idx in range(num_players)
    ]

    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = agents[player_id].step(time_step)
      print(agent_output)
      time_step = env.step([agent_output.action])
      print(time_step)
    for agent in agents:
      agent.step(time_step)

test()



# def run_ismcts():
#     params = MurderParams(8, 1, 1)
#     game = MurderGame(game_params=params)
#     state = game.new_initial_state()
#     while state.is_terminal() is not True:
#         action = ismcts_agent(state,1000)
#         state._apply_action(action)
#     print(state.returns())




