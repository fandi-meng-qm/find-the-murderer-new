from typing import List
import matplotlib.pyplot as plt
import numpy as np
from game_core import MurderGame, MurderParams
from reinforce import reinforce
from new_deductive_1d_game.version4.evolutionary import one_plus_one_es
from test import test_simulations
from agents import ISMCTSAgent

def plot_data(data: List[List[float]], title: str) -> None:
    for dat in data:
        plt.plot(dat)
    plt.xlabel("number of itertions")
    plt.ylabel("score")
    plt.title(title)
    plt.show()


def plot_series_with_shaded_error_bars(data: List[List[float]], title: str, series_label: str = 'series',
                                       show: bool = True, x_label: str = 'max_simulations', y_label: str = 'score') -> None:

    data = np.array(data)
    n = len(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) / np.sqrt(n)
    # plt.plot(mean, label=series_label)
    x_axis = np.arange(2, 2 + len(mean))
    plt.plot(x_axis, mean, label=series_label)
    plt.fill_between(x_axis, mean - std, mean + std, alpha=0.5)
    # plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    if show: plt.show()


def run_error_bars(n):
    data = []
    # for i in range(20):
    #     curr_policy, return_list = one_plus_one_es(game, max_iterations=1000, mutation_rate=0.9, sigma=0.2)
    #     # averaged_lst = [np.mean(return_list[i:i + 100]) for i in range(0, len(return_list), 100)]
    #     # policy, return_list = reinforce(400, game, 0.001)
    #     data.append(return_list)
    #
    # plot_series_with_shaded_error_bars(data, "1+1ES/REINFORCE on 1-D Simple Game, n_units =20", series_label="1+1 ES", show=False)
    #
    # data_2 = []
    # for i in range(20):
    #     policy2, return_list2 = reinforce(100, game, 0.001)
    #     data_2.append(return_list2)
    # plot_series_with_shaded_error_bars(data_2, "1+1ES/REINFORCE on 1-D Simple Game, n_units =20", series_label="REINFORCE", show=True)

    data_3 = []
    for i in range(n):
        return_list3 = test_simulations(100)
        data_3.append(return_list3)
    plot_series_with_shaded_error_bars(data_3, "ISMCTS on 1-D Simple Game, n_units =20",
                                       series_label="ISMCTS", show=True)


if __name__ == "__main__":
    params = MurderParams(8, 1, 1)
    game = MurderGame(game_params=params)
    run_error_bars(10)
