import numpy as np
from typing import List

np.random.seed(0)


def random_detective(people, alive, accused) -> List[float]:
    random_policy = [0] * len(people)
    for person in people:
        if person in alive - accused:
            random_policy[person.id] = 1 / (len(alive - accused))
    return random_policy





