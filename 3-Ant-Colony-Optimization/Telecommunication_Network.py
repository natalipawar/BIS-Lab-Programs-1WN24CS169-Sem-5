import random

# Network: nodes and link costs
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'C': 1, 'D': 7, 'E': 5},
    'C': {'A': 4, 'B': 1, 'D': 3, 'E': 2},
    'D': {'B': 7, 'C': 3, 'F': 2},
    'E': {'B': 5, 'C': 2, 'F': 3},
    'F': {'D': 2, 'E': 3}
}

random.seed(7)
ants, iterations = 20, 50
alpha, beta, rho, Q = 1, 2, 0.4, 100

pheromone = {
    u: {v: 1.0 for v in graph[u]} for u in graph
}

best_path = None
best_cost = float('inf')

for _ in range(iterations):
    routes = []

    for _ in range(ants):
        path = ['A']
        visited = {'A'}

        while path[-1] != 'F':
            current = path[-1]
            choices = [
                v for v in graph[current]
                if v not in visited
            ]

            if not choices:
                break

            weights = [
                pheromone[current][v] ** alpha
                * (1 / graph[current][v]) ** beta
                for v in choices
            ]

            nxt = random.choices(
                choices, weights=weights, k=1
            )[0]

            path.append(nxt)
            visited.add(nxt)

        if path[-1] == 'F':
            cost = sum(
                graph[a][b]
                for a, b in zip(path, path[1:])
            )
            routes.append((path, cost))

            if cost < best_cost:
                best_path = path[:]
                best_cost = cost

    # Pheromone evaporation
    for u in pheromone:
        for v in pheromone[u]:
            pheromone[u][v] *= (1 - rho)

    # Pheromone deposit
    for path, cost in routes:
        for a, b in zip(path, path[1:]):
            pheromone[a][b] += Q / cost
            pheromone[b][a] += Q / cost

print("Best Route:", best_path)
print("Minimum Cost:", best_cost)

