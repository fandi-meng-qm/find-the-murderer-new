from game_core import MurderGame, MurderParams
from typing import List, Optional, Dict, Tuple
import numpy as np


class InfoSetTabularPolicy:
    def __init__(self, game) -> None:
        self.game = game
        self.policy_dict = dict()

    def uniform_random(self) -> Dict[tuple, List[float]]:
        init_state = self.game.new_initial_state()
        for _ in self.game.all_info_states:
            self.policy_dict[_] = [1 / len(init_state.init_actions)] * len(init_state.init_actions)
        return self.policy_dict

    def prob_random(self) -> Dict[tuple, List[float]]:
        init_state = self.game.new_initial_state()
        for _ in self.game.all_info_states:
            self.policy_dict[_] = np.random.rand(len(init_state.init_actions)).tolist()
        return self.policy_dict



