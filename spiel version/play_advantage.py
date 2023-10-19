import logging
import pickle
from typing import List
from pyspiel import TabularPolicy

from test_utilities import policy_as_list, sample
from murder_game_core import MurderGame, MurderParams
from open_spiel.python import policy as policy_lib
from all_tabular_policies import uniform_random_policy, cfr_policy, \
    reinforce_policy, greedy_policy

# set up a logger so that we can easily turn on/off turn by turn printing
game_logger = logging.getLogger(__name__)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(c_format)

game_logger.addHandler(c_handler)


# this should work for any openspiel game, but I've been lazy and put this type for now
def run_game(game: MurderGame, players: List[TabularPolicy]) -> List[float]:
    state = game.new_initial_state()
    # print(state.params)
    while not state.is_terminal():
        game_logger.info(f"\nstep: {state.step}")
        game_logger.info(f"State: {state}")
        if state.is_chance_node():
            action_probs = state.chance_outcomes()
        else:
            player_policy = players[state.current_player()]
            action_probs = policy_as_list(player_policy, state)
        action = sample(action_probs)
        game_logger.info(f"Player {state.current_player()} took action {action}")
        state = state.child(action)
        game_logger.info(f"New state -> : {state}")
    game_logger.info(f"\nFinal state: {state}")
    return state.returns()


def run_games(n_games, player_policies) -> None:
    game_logger.info(game.game_params)
    total = 0
    for i in range(n_games):
        # zero sum game, just take return for first player
        result = run_game(game, player_policies)[0]
        game_logger.debug(f"{i}: \t {result}")
        total += result
    print(f"Average: {total / n_games:1}")


cfr_killer_random_detective = [cfr_policy, uniform_random_policy]
random_killer_cfr_detective = [uniform_random_policy, cfr_policy]

reinforce_killer_random_detective = [reinforce_policy, uniform_random_policy]
random_killer_reinforce_detective = [uniform_random_policy,reinforce_policy]

greedy_killer_random_detective = [greedy_policy, uniform_random_policy]
random_killer_greedy_detective = [uniform_random_policy,greedy_policy]

params = MurderParams(1, 5, 5)
game = MurderGame(game_params=params)

if __name__ == '__main__':
    # (un)comment following to turn (on/off) logging
    # game_logger.setLevel(logging.INFO)
    game_logger.info("Test message")

    run_games(1000, cfr_killer_random_detective)
    run_games(1000, random_killer_cfr_detective)
    run_games(1000, reinforce_killer_random_detective)
    run_games(1000, random_killer_reinforce_detective)
    run_games(1000, greedy_killer_random_detective)
    run_games(1000, random_killer_greedy_detective)