import math
import numpy as np

# class MurderParams:
#     def __init__(self, m_grid: int, n_grid: int, n_people: int):
#         self.m_grid = 1
#         self.n_grid = 5
#         self.n_people = 5
#
#
# def get_init_people(m_grid, n_grid, n_people):
#     locations = [(x, y) for x in range(m_grid) for y in range(n_grid)]
#     p = list(itertools.combinations(list(range(m_grid * n_grid)), n_people))
#     init_people = []
#     for i in range(len(p)):
#         state = []
#         for j in p[i]:
#             state.append(Person(locations[j]))
#         init_people.append(state)
#     # print(init_people)
#     return init_people
#
#
# def gen_distance_table(people) -> np.array:
#     dist_table = np.zeros((len(people), len(people)))
#     for i in range(len(people)):
#         for j in range(len(people)):
#             dist_table[i][j] = \
#                 ((people[i][0] - people[j][0]) ** 2 +
#                  (people[i][1] - people[j][1]) ** 2) ** 0.5
#     return dist_table
#
#
# def gen_cost_list(people, killer) -> list:
#     cost_list = len(people) * [0]
#     dist_list = gen_distance_table(people)[people.index(killer)]
#     _range = np.max(dist_list) - np.min(dist_list)
#     for i in range(len(people)):
#         cost_list[i] = math.ceil(((dist_list[i] - np.min(dist_list)) / _range) * len(people))
#     return cost_list
#
#
# def get_init_states(params) -> list:
#     init_people = get_init_people(params.m_grid, params.n_grid, params.n_people)
#     init_states = []
#     for i in init_people:
#         for j in i:
#             cost_list = gen_cost_list(i, j)
#             game = MurderGame(params)
#             state = MurderState(game)
#             # state.params= params
#             print(state.params)
#             init_states.append(state)
#     return init_states


# get_init_states(MurderParams(1,5,5))

print(math.comb(5, 5))
