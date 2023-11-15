import numpy as np


def evaluater(ind, stateprobs):
    return np.random.normal(stateprobs[ind], 0.2)


best = 0
reward = 0
avg = 0
iterations = 100
bestconf = []

for _ in range(iterations):
    k = 6
    time_steps = 1000
    eps = 0.13
    alpha = 0.1

    ps = np.random.normal(0.5, 0.2, k)
    totreward = 0

    ht = np.zeros(k)

    avg_reward = 0

    for i in range(time_steps):

        probs = np.array([(np.exp(a) / (np.sum(np.exp(ht)))) for a in ht])
        action = np.random.choice(len(probs), p=probs)

        reward = evaluater(action, ps)
        avg_reward = ((avg_reward * i) + reward) / (i + 1)

        for a, hta in enumerate(ht):
            if a == action:
                ht[a] = hta + alpha * (reward - avg_reward) * (1 - probs[a])
            else:
                ht[a] = hta - alpha * (reward - avg_reward) * probs[a]

        #  Random Walk feature
        ps_change = np.random.normal(0, 0.01, k)
        ps = ps + ps_change

        totreward += reward

    avg += totreward

    if best < totreward:
        best = totreward
        bestconf = (ht, ps)

print(f"Best scenario had {best}.\n"
      f"\nAverage scenario had {round(avg / iterations, 3)}"
      f"\nScenario had ht: {bestconf[0]}\nProbs: {bestconf[1]}")
