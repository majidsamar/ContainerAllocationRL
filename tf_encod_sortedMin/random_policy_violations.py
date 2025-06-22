import numpy as np
import matplotlib.pyplot as plt

BAYS = 9
ROWS = 3
TIERS = 5

fill_ratio = 0.9
num_containers = round(fill_ratio * BAYS * ROWS * TIERS)

# empirically compute the probability of violations with  a random policy
num_runs = 10_00


yard = np.zeros((BAYS, ROWS, TIERS))

def random_policy():
    while True:
        b = np.random.randint(0, BAYS)
        r = np.random.randint(0, ROWS)

        empties = np.argwhere(yard[b, r] == 0.)

        if len(empties) > 0:
            t = empties[0]

            return b, r, t



all_violations = []
total_violations = 0

for i in range(num_runs):
    yard = np.zeros((BAYS, ROWS, TIERS))
    violations = []
    for _ in range(num_containers):

        action = random_policy()

        dwell_time = np.random.random()

        stack = yard[action[:2]]
        violated = np.logical_and(
            stack - dwell_time < 0,
            stack != 0.
        )

        n_violated = np.sum(violated)
        total_violations += n_violated

        yard[action] = dwell_time

        violations.append(n_violated)

    all_violations.append(np.cumsum(violations))


print("Mean KPI", total_violations / (num_runs * num_containers))

all_kpi = np.array(all_violations) / num_containers
mean_violations = np.mean(all_kpi, axis=0)
std_violations = np.std(all_kpi, axis=0)
fill_ratios = (np.arange(num_containers) / (BAYS * ROWS * TIERS))

plt.errorbar(x=fill_ratios, y=mean_violations, yerr=std_violations)
plt.ylabel("KPI")
plt.xlabel("Yard Fill Ratio")
plt.show()