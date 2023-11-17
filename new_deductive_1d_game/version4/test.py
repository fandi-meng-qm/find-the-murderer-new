from __future__ import annotations
from new_deductive_1d_game.version4.game_core import MurderGame, MurderParams
from open_spiel.python import rl_environment
from new_deductive_1d_game.version4.agents import ISMCTSAgent, RandomAgent, OptimalAgent,BetterRandomAgent
from new_deductive_1d_game import ismcts
from open_spiel.python.algorithms import mcts

params = MurderParams(4, 1, 1)
game = MurderGame(game_params=params)
env = rl_environment.Environment(game, include_full_state=True)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]
ismcts_bot = ismcts.ISMCTSBot(
    game=env.game,
    uct_c=300,
    max_simulations=2,
    evaluator=mcts.RandomRolloutEvaluator())


def test(agent) -> float:
    bot = ismcts_bot if agent == ISMCTSAgent else None
    agents = [
        agent(
            player_id=idx, num_actions=num_actions, bot=bot)
        for idx in range(num_players)
    ]

    time_step = env.reset()
    step_count = 0
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        # print(agent_output.action)
        time_step = env.step([agent_output.action])
        # print(time_step)
        step_count += 1
    for agent in agents:
        agent.step(time_step)
    # print(step_count)
    return step_count


def test_games(n, agent):
    reward_list = []
    for i in range(n):
        reward_list.append(test(agent))
    average_reward = sum(reward_list) / len(reward_list)
    print(f"The average number of steps for {n} times of gameplay with {agent} is {average_reward}.")

if __name__ == "__main__":
    # test_games(100, RandomAgent)
    # test_games(100, BetterRandomAgent)
    test_games(1000, ISMCTSAgent)
    # test_games(100,OptimalAgent)

# 6.422,9.803,6.415,5.844,5.231,4.919,4.131,3.776,3.76