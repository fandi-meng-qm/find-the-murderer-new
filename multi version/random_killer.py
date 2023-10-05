import array, math
import random

import numpy as np
from game_interface_multi import *
from typing import Set, List


def random_killer(people, victims, points, cost_list) -> id:
    victim = random.choice(victims)
    return victim


