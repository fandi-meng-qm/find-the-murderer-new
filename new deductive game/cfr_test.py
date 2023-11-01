from game_core import MurderGame,MurderParams


params = MurderParams(4, 1, 1)
game = MurderGame(game_params=params)

if __name__ == '__main__':
    # create a uniform random policy
    uniform_random_policy = policy_lib.TabularPolicy(game)
    cfr_policy,eval_steps,eval_nash_conv = get_cfr_policy(game, 129)
    # show nash converge
    fig, ax = plt.subplots()
    ax.set_title("NashConv by CFR Iteration")
    ax.plot(eval_steps, eval_nash_conv)
    fig.show()
    #  print cfr policy
    print_policy(cfr_policy)
    result = total_advantage(game, cfr_policy, uniform_random_policy)
    print_policy(cfr_policy)
    print(f"Advantage = {result}")