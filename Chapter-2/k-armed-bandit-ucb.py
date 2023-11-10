import numpy as np


def evaluater(ind, probs):
    return np.random.normal(probs[ind], 0.2)


best = None
tope = 1000000
avg = 0

for _ in range(1):
    k = 20
    time_steps = 100000
    a = 0.1
    c = 2

    ps = np.random.normal(0.5, 0.2, k)
    totreward = 0

    qt = np.zeros((k, 2))

    visited = np.arange(k).tolist()

    for i in range(time_steps):

        action = 0
        if len(visited) != 0:
            action = visited.pop()
            explore = 0
        else:
            explore = c * np.sqrt(np.log(i) / qt[:, 1])
            action = np.argmax(qt[:, 0] + explore)

        reward = evaluater(action, ps)

        m, n = qt[action]
        qt[action][0] = qt[action][0] + a * (reward - qt[action][0])
        qt[action][1] += 1

        totreward += reward

        # Random Walk feature
        if i % 100 == 0:
            ps = ps + np.random.normal(0, 0.01, k)

    p_error = 100 * np.sum(np.absolute((qt[:, 0] - ps) / ps)) / k

    avg += p_error

    if p_error < tope:
        tope = p_error
        best = (ps, qt)

print(f"Best scenario had %-diff {round(tope, 3)}%.\n"
      f"Ultimate differences were:\nPS:\n {best[0]}\nQT:\n{best[1][:, 0]}"
      f"\nAverage scenario had %-diff {round(avg / 1, 3)}%")