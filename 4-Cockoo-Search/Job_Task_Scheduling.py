import random
import math

print("Natali Pawar")
print("1WN24CS169")

# 1. Define tasks and heterogeneous processors
tasks = [4, 6, 3, 7, 5, 2, 8, 4]  # Task workloads

speed = [1, 2, 3]       # Processor speeds
power = [2, 5, 9]        # Power consumption per time unit
deadline = 12

N = 25
pa = 0.25
MAX_ITER = 300
step_size = 0.3
random.seed(42)

num_tasks = len(tasks)
num_processors = len(speed)

# 2. Decode nest into processor assignments
def decode(nest):
    return [
        max(0, min(num_processors - 1, round(x)))
        for x in nest
    ]

# 3. Calculate energy and completion time
def evaluate(nest):
    assignment = decode(nest)
    load = [0.0] * num_processors
    energy = 0.0

    for i, p in enumerate(assignment):
        time = tasks[i] / speed[p]
        load[p] += time
        energy += time * power[p]

    makespan = max(load)

    # Penalize schedules that violate the deadline
    penalty = 1000 * max(0, makespan - deadline)
    return energy + penalty, energy, makespan, assignment, load

# 4. Initialize nests
nests = [
    [random.uniform(0, num_processors - 1)
     for _ in range(num_tasks)]
    for _ in range(N)
]

best = min(nests, key=lambda x: evaluate(x)[0])
best_score = evaluate(best)[0]

# 5. Cuckoo Search
for iteration in range(MAX_ITER):

    for i in range(N):
        u = random.gauss(0, 1)
        v = random.gauss(0, 1)
        levy = u / (abs(v) ** (1 / 1.5) + 1e-10)

        new = [
            max(0, min(num_processors - 1,
                nests[i][j] + step_size * levy))
            for j in range(num_tasks)
        ]

        j = random.randrange(N)

        if evaluate(new)[0] < evaluate(nests[j])[0]:
            nests[j] = new

    # Abandon the worst nests
    ranked = sorted(nests, key=lambda x: evaluate(x)[0])
    keep = max(1, int(N * (1 - pa)))
    nests = ranked[:keep]

    while len(nests) < N:
        nests.append([
            random.uniform(0, num_processors - 1)
            for _ in range(num_tasks)
        ])

    current = min(nests, key=lambda x: evaluate(x)[0])

    if evaluate(current)[0] < best_score:
        best = current
        best_score = evaluate(best)[0]

# 6. Display the best schedule
score, energy, makespan, assignment, load = evaluate(best)

print("Best Task Schedule:")
for i, p in enumerate(assignment):
    print(f"Task {i + 1} -> Processor {p + 1}")

print("\nProcessor completion times:", [round(x, 2) for x in load])
print("Total Energy Consumption:", round(energy, 2))
print("Makespan:", round(makespan, 2))
print("Deadline:", deadline)
print("Deadline satisfied:", makespan <= deadline)
