import pickle,dill
from test_utilities import get_cfr_policy
from murder_game_core import MurderParams, MurderGame



def save_cfr_policy():
    params = MurderParams(1,5,5)
    game = MurderGame(game_params=params)
    cfr_policy,eval_steps,eval_nash_conv = get_cfr_policy(game, 129)
    cfr_policy=cfr_policy.action_probability_array
    with open("cfr_policy.pkl", "wb") as file:
        pickle.dump(cfr_policy, file)




save_cfr_policy()

