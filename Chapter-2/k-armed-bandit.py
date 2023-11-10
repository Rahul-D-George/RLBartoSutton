from random import random
import numpy as np


def evaluater(ind, probs):
    return np.random.normal(probs[ind], 0.2)


best = None
tope = 100
avg = 0

for _ in range(100):
    k = 20
    time_steps = 10000
    eps = 0.13
    a = 0.1

    ps = np.random.normal(0.5, 0.2, k)
    totreward = 0

    qt = np.zeros(k)  # (k, 2)
    for i in range(time_steps):
        if random() > eps:
            action = np.argmax(qt)  # qt[:, 0]
        else:
            action = np.random.randint(0, k)
        reward = evaluater(action, ps)
        # m, n = qt[action]
        qt[action] = qt[action] + a * (reward - qt[action])
        # qt[action][0] = (m*n + reward)/(n+1)
        # qt[action][1] += 1
        totreward += reward

        # Random Walk feature
        # ps_change = np.random.normal(0, 0.01, k)
        # ps = ps + ps_change

    p_error = 100 * np.sum(np.absolute((qt - ps) / ps)) / k

    avg += p_error

    if p_error < tope:
        tope = p_error
        best = (ps, qt)

print(f"Best scenario had %-diff {round(tope, 3)}%.\n"
      f"Ultimate differences were:\nPS:\n {best[0]}\nQT:\n{best[1]}"
      f"\nAverage scenario had %-diff {round(avg / 100, 3)}%")

# It is apparent from the % differences obtained with/without using the random walks
# that stationary bandits suffer immensely from this problem.
