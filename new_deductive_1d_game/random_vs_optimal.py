import math
from matplotlib import pyplot as plt


memo={(1,0): 0}

def random_cut(i,n):
    if (n,i) in memo:
        return memo[(n,i)]
    # No need to cut if there is only one cell
    if n == 1:
        return 0
    # Calculate the expected number of cuts for every possible split point
    total = 0
    for k in range(1, n):  # Iterate over all possible split points
        # Calculate the expected number of cuts for the left and right side after a split
        # and add one for the current split 01234567 1234567
        left_cuts = random_cut(i,k) if i < k else 0
        right_cuts = random_cut(i-k,n - k) if i >= k else 0
        total += (left_cuts + right_cuts + 1)  # Add the cut for the current decision

    # Store the result in memo to use later
    memo[(n,i)] = total / (n - 1)  # Average number of cuts
    # print(memo)
    return memo[(n,i)]

def random_count(n):
    count=0
    for i in range(0,n):
        count += random_cut(i,n)
    return count/n

def optimal_cut(n):
    # print(math.log(n, 2))
    return math.log(n, 2)



size_list=[]
random_list=[]
optimal_list=[]

for i in range(2,129):
    if i & (i - 1) == 0:
        size_list.append(i)
        random_list.append(random_count(i))
        optimal_list.append(optimal_cut(i))

# print(random_list)
plt.plot(size_list, random_list, label='Random Policy')
plt.plot(size_list, optimal_list, label='Optimal Policy')
for y in range(0, 12, 2):
    plt.axhline(y=y, color='grey', linestyle='--', linewidth=0.5)

plt.xlabel('Number of grids')
plt.ylabel('Expected number of cuts')

plt.title('Compare the expected number of cuts of random policy and optimal policy')
plt.legend()
plt.show()



