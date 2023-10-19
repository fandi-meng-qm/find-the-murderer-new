
from absl.testing import absltest
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import ismcts
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import mcts_agent
from murder_game_core import MurderGame, MurderParams
import pyspiel
import numpy as np
from open_spiel.python import rl_agent

# def serialize_state(state):
#     # 根据您的游戏来定义序列化方法
#     serialized_data = {
#         "player": state.current_player(),
#         # ... 其他所需的状态属性
#     }
#     return serialized_data


def deserialize_state(observations):
    params = MurderParams(1, 5, 5)
    game = MurderGame(game_params=params)
    state = game.new_initial_state()
    history = observations.split("[State]\nhistory=")[1].split("\n")[0]
    history_list = [int(i) for i in history.split(':')]
    # history to state
    player_list = history_list[::2]
    action_list = history_list[1::2]
    for i in range(len(player_list)):
        if player_list[i] ==-1:
            state=state.child(action_list[i])
        elif player_list[i]==0:
            state=state.child(action_list[i])
        elif player_list[i]==1:
            state = state.child(action_list[i])
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
    # h=time_step.observations["serialized_state"].split("[State]\nhistory=")[1].split("\n")[0]
    # history_list = [int(i) for i in h.split(':')]
    # print(history_list[-2])
    # print(pyspiel.deserialize_game_and_state(time_step.observations["serialized_state"]))

    _, state = deserialize_state(time_step.observations["serialized_state"])
    # Call the MCTS bot's step to get the action.
    probs = np.zeros(self._num_actions)
    action = self._mcts_bot.step(state)
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)


class MCTSAgentTest(absltest.TestCase):

  def test_episode(self):
    params = MurderParams(1, 5, 5)
    game = MurderGame(game_params=params)
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
      player_id = time_step.observations["current_player"]
      agent_output = agents[player_id].step(time_step)
      time_step = env.step([agent_output.action])
    for agent in agents:
      agent.step(time_step)


if __name__ == "__main__":
  absltest.main()