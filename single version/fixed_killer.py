import array
import numpy as np
from game_core import MurderGameState


# Because the mechanism of the game is mainly related to distance,
# it is necessary to have a table that includes all distances,
# and then calculate the probability based on this table.
# # The number of rows and columns of this table are both n_people
def gen_distance_table(people) -> array:
    dist_table = np.zeros((len(people), len(people)))
    for i in people:
        for j in people:
            dist_table[i.id][j.id] = \
                ((i.location[0] - j.location[0]) ** 2 +
                 (i.location[1] - j.location[1]) ** 2) ** 0.5
    return dist_table


def near_killer(people, alive, killer) -> list:
    recip_list = [0] * len(people)
    dead = people - alive
    for person in people:
        if person in {killer}:
            recip_list[person[0]] = 0
        if person in alive - {killer}:
            recip_list[person[0]] = 1 / gen_distance_table(people)[killer.id, ...][person[0]]
        if person in dead:
            recip_list[person[0]] = 0
    near_policy = [i / sum(recip_list) for i in recip_list]
    return near_policy


def far_killer(people, alive, killer) -> list:
    far_policy = [0] * len(people)
    dead = people - alive
    for person in people:
        if person in {killer}:
            far_policy[person[0]] = 0
        if person in alive:
            far_policy[person[0]] = gen_distance_table(people)[killer.id, ...][person[0]]
        if person in dead:
            far_policy[person[0]] = 0
    far_policy = [i / sum(far_policy) for i in far_policy]
    return far_policy


# use a dictionary to include all kinds of killer, this version only near and far
killers = {'near_killer': near_killer,
           'far_killer': far_killer
           }


def choose_killer(killer_kinds):
    killer_kind = np.random.choice(['near_killer', 'far_killer'], size=1, replace=False, p=(1, 0))
    kill_policy = killer_kinds[killer_kind[0]]
    return kill_policy


fixed_killer = choose_killer(killers)





