from random import random
import numpy as np

k = 20
time_steps = 10000
eps = 0.1
a = 0.1
ps = np.random.normal(0.5, 0.2, k)
totreward = 0

qt = np.zeros(k)

def evaluater(i):
    return np.random.normal(ps[i], 0.2)


for i in range(time_steps):
    if random() > eps:
        action = np.argmax(qt)
    else:
        action = np.random.randint(0, k)
    reward = evaluater(action)
    # m, n = qt[action]
    qt[action] = qt[action] + a * (reward - qt[action])
    # qt[action][0] = (m*n + reward)/(n+1)
    # qt[action][1] += 1
    totreward += reward

    # Random Walk feature
    ps_change = np.random.normal(0, 0.01, k)
    ps = ps + ps_change

p_error = np.sum(np.absolute((qt - ps)/ ps)) / k


print(f"Reward was {totreward} with epsilon of {eps}.\n"
      f"Average % difference in true vs learned values: {round(p_error * 100, 3)}%")

# It is apparent from the % differences obtained with/without using the random walks
# that stationary bandits suffer immensely from this problem.

print(qt, ps)