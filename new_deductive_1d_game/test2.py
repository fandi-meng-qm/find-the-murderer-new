import itertools
import math
import random

print(math.comb(18, 3))
print(6*6*9)

set1={0, 1, 2, 3, 4, 5}
set2={6,7,8,9,10,11}
set3={12,13,14,15,16,17,18,19,20}
actions = [set(item) for item in list(itertools.product(set1,set2,set3))]

print(actions)
print(len(actions))

my_set = set(range(18))


combinations = itertools.combinations(my_set, 3)
list=[]
for combo in combinations:
    list.append(set(combo))
print(len(list))

def random_choose(cards):
    card = random.choice(cards)
    print(card)
    return card

random_choose({0,1,2})

