from test_utilities import get_cfr_policy, total_advantage, print_policy, advantage_as_policy_player
from murder_game_core import MurderGame, MurderParams
from all_tabular_policies import uniform_random_policy, cfr_policy, \
    reinforce_policy, greedy_policy

params = MurderParams(1, 5, 5)
game = MurderGame(game_params=params)




def test_advantage(policy1,policy2):
    result = total_advantage(game, policy1, policy2)
    return result


# -0.5560078827880276,detective's advantage with policy: 0.5902574056255119
# test_advantage(uniform_random_policy,uniform_random_policy)
test_advantage(cfr_policy,uniform_random_policy)
test_advantage(reinforce_policy,uniform_random_policy)
test_advantage(greedy_policy,uniform_random_policy)
# test_advantage(greedy_policy,cfr_policy)

# if __name__ == '__main__':
#     params = MurderParams(1,5,5)
#     game = MurderGame(game_params=params)
#     # create a uniform random policy
#     uniform_random_policy = policy_lib.TabularPolicy(game)
#     cfr_policy,eval_steps,eval_nash_conv = get_cfr_policy(game, 129)
#     # show nash converge
#     fig, ax = plt.subplots()
#     ax.set_title("NashConv by CFR Iteration")
#     ax.plot(eval_steps, eval_nash_conv)
#     fig.show()
#     #  print cfr policy
#     print_policy(cfr_policy)
#     result = total_advantage(game, cfr_policy, uniform_random_policy)
#     print_policy(cfr_policy)
#     print(f"Advantage = {result}")
