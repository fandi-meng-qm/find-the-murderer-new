import re
from typing import Tuple

from absl.testing import absltest
from open_spiel.python import rl_environment
# from open_spiel.python.algorithms import ismcts
import ismcts
from open_spiel.python.algorithms import mcts
from murder_game_core import MurderGame, MurderParams, get_init_states
import pyspiel
import numpy as np
from open_spiel.python import rl_agent


params = MurderParams(1, 5, 5)
game = MurderGame(game_params=params)

def deserialize_state(observations):
    state = game.new_initial_state()
    history = observations.split("[State]\nhistory=")[1].split("\n")[0]
    history_list = [int(i) for i in re.split(',|:', history)]
    # history to state
    player_list = history_list[::2]
    action_list = history_list[1::2]
    for i in range(len(player_list)):
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
        max_simulations=100,
        evaluator=mcts.RandomRolloutEvaluator())

    agents = [
        MCTSAgent(
            player_id=idx, num_actions=num_actions, mcts_bot=ismcts_bot)
        for idx in range(num_players)
    ]

    time_step = env.reset()
    while not time_step.last():
        # and time_step.observations["current_player"] is not:
      player_id = time_step.observations["current_player"]
      print(player_id)
      print(time_step.observations["legal_actions"])
      agent_output = agents[player_id].step(time_step)
      print(agent_output)
      time_step = env.step([agent_output.action])
      print(time_step)
    for agent in agents:
      agent.step(time_step)

test()
