from game_interface_single import *
from fixed_killer import *
from game_core import MurderGameModel
from detective import random_detective



def game(model, killer_policy, detective_policy) -> float:


    while model.is_terminal() is None:
        # get killer's observation
        killer_obs = ObservationForKiller.from_game_state(model.state)
        # get killer's policy(list) based on the observation and the chosen kill policy
        kill_policy = killer_policy(killer_obs.people, killer_obs.alive, killer_obs.killer)
        # get killer's action, the input is the possible actions and the policy list(probability)
        kill_action = KillerInterface.get_action(killer_obs, kill_policy)
        # update the killer's action to the game model
        MurderGameModel.step_kill(model, kill_action)

        # the detective step is the same structure
        detective_obs = ObservationForDetective.from_game_state(model.state)
        accuse_policy = detective_policy(detective_obs.people, detective_obs.alive, detective_obs.accused)
        accuse_action = DetectiveInterface.get_action(detective_obs, accuse_policy)
        MurderGameModel.step_accuse(model, accuse_action)

    if model.is_terminal is not None:
        score = model.is_terminal()
        print(score)
        return score


# killer_policy:
# detective_policy

game(MurderGameModel(3,1,3), fixed_killer, random_detective)

