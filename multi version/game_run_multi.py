from game_interface_multi import *
from game_core import MurderGameModel
from random_detective import random_detective
from random_killer import random_killer


def game(model, killer_policy, detective_policy) -> float:
    model.state.step += 1
    while model.is_terminal() is None and model.state.step >= 0:
        model.state.player = model.state.step % 2
        # get killer's observation
        killer_obs = ObservationForKiller.from_game_state(model.state)
        # get killer's legal actions
        people, victims, points, cost_list = KillerInterface.get_actions(killer_obs)
        # get killer's action
        if victims is None or victims == []:
            pass
        else:
            victim = killer_policy(people, victims, points, cost_list)
            MurderGameModel.step_kill(model, victim)
        model.state.step += 1

        model.state.player = model.state.step % 2
        # get detective's observation
        detective_obs = ObservationForDetective.from_game_state(model.state)
        # get detective's legal actions
        suspects = DetectiveInterface.get_actions(detective_obs)
        # get detective's action
        suspect = detective_policy(suspects)
        MurderGameModel.step_accuse(model, suspect)
        model.state.step += 1

    if model.is_terminal is not None:
        detective_score = model.is_terminal()
        killer_score = - model.is_terminal()
        print(detective_score, killer_score)
        return detective_score, killer_score


# killer_policy: random_killer,
# detective_policy: random_detective,

if __name__ == "__main__":
    game(MurderGameModel(1, 5, 5), random_killer, random_detective)
