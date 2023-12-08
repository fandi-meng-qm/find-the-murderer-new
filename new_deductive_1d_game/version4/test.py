from __future__ import annotations
from game_core import MurderGame, MurderParams
from open_spiel.python import rl_environment
from new_deductive_1d_game.version4.agents import ISMCTSAgent, RandomAgent, OptimalAgent,BetterRandomAgent
import ismcts
from open_spiel.python.algorithms import mcts
import time

params = MurderParams(4, 1, 1)
game = MurderGame(game_params=params)
env = rl_environment.Environment(game, include_full_state=True)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]



def test(agent,n) -> float:
    ismcts_bot = ismcts.ISMCTSBot(
        game=env.game,
        uct_c=1.5,
        max_simulations=n,
        evaluator=mcts.RandomRolloutEvaluator())

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

def test_simulations(n):
    reward_list = [-test(RandomAgent,None)]
    for i in range(2,n):
        reward_list.append(-test(ISMCTSAgent, i))
    return reward_list

def test_games(n, agent):
    reward_list = []
    for i in range(n):
        reward_list.append(test(agent,None))
    average_reward = sum(reward_list) / len(reward_list)
    print(f"The average number of steps for {n} times of gameplay with {agent} is {average_reward}.")

if __name__ == "__main__":
    start_time = time.time()
    # test_games(100, RandomAgent)
    # test_games(100, BetterRandomAgent)
    # test_games(20, ISMCTSAgent)
    # test_games(100,OptimalAgent)
    end_time = time.time()
    print("time cost: {:.2f}s".format(end_time - start_time))

# 6.422,9.803,6.415,5.844,5.231,4.919,4.131,3.776,3.76