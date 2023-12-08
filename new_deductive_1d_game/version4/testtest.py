from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_data(data: List[List[float]], title: str) -> None:
   for dat in data:
       plt.plot(dat)
   plt.xlabel("t")
   plt.ylabel("Error")
   plt.title(title)
   plt.show()


def plot_series_with_shaded_error_bars(data: List[List[float]], title: str, series_label: str = 'series', show: bool = True, x_label: str = 't', y_label: str = 'score') -> None:
   data = np.array(data)
   n = len(data)
   mean = np.mean(data, axis=0)
   std = np.std(data, axis=0) / np.sqrt(n)
   plt.plot(mean, label = series_label)
   plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5)
   plt.xlabel(x_label)
   plt.ylabel(y_label)
   plt.legend()
   plt.title(title)
   if show: plt.show()

def run_error_bars():
   data = []
   for i in range(10):
       data.append(np.random.normal(0, 1, 100))

   print(data)
   plot_series_with_shaded_error_bars(data, "Test", series_label="series 1", show=False)
   data_2 = []
   for i in range(10):
       data_2.append(np.random.normal(0.5, 0.5, 100))
   plot_series_with_shaded_error_bars(data_2, "Test 2", series_label="series 2", show=True)


if __name__ == "__main__":
   run_error_bars()

