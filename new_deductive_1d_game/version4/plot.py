from matplotlib import pyplot as plt



def show_ismct_converge(the_list):
  x = list(range(2,11))
  plt.plot(x, the_list)

  plt.xlabel('numuber of max_simulations')
  plt.ylabel('the average number of steps (1000 times)')

  plt.title("ISMCTS agent's performance with the increase of the max_simulations")

  plt.show()

the_list = [6.422,9.803,6.415,5.844,5.231,4.919,4.131,3.776,3.76]

show_ismct_converge(the_list)
