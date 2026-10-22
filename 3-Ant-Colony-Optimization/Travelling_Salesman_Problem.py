
import random
import math

# 1. Define cities using coordinates
cities = {
    "A": (0, 0),
    "B": (2, 3),
    "C": (5, 2),
    "D": (6, 6),
    "E": (1, 5)
}

N = len(cities)

# Calculate distance between cities
distance = [[0] * N for _ in range(N)]
names = list(cities.keys())

for i in range(N):
    for j in range(N):
        x1, y1 = cities[names[i]]
        x2, y2 = cities[names[j]]
        distance[i][j] = math.sqrt(
            (x2 - x1)**2 + (y2 - y1)**2
        )

# 2. Initialize parameters
num_ants = N
alpha = 1
beta = 2
rho = 0.5
Q = 100
iterations = 100

pheromone = [[1.0] * N for _ in range(N)]

best_route = None
best_distance = float("inf")

# 3. Construct solutions
for iteration in range(iterations):
    all_routes = []
    all_distances = []

    for ant in range(num_ants):
        start = random.randrange(N)
        route = [start]
        unvisited = set(range(N))
        unvisited.remove(start)

        while unvisited:
            current = route[-1]
            choices = list(unvisited)

            weights = [
                (pheromone[current][j] ** alpha) *
                ((1 / distance[current][j]) ** beta)
                for j in choices
            ]

            next_city = random.choices(
                choices, weights=weights, k=1
            )[0]

            route.append(next_city)
            unvisited.remove(next_city)

        # Return to the starting city
        route.append(start)

        total = sum(
            distance[route[i]][route[i + 1]]
            for i in range(N)
        )

        all_routes.append(route)
        all_distances.append(total)

        if total < best_distance:
            best_distance = total
            best_route = route.copy()

    # 4. Update pheromones: evaporation
    for i in range(N):
        for j in range(N):
            pheromone[i][j] *= (1 - rho)

    # Deposit more pheromone on shorter routes
    for route, total in zip(all_routes, all_distances):
        for i in range(N):
            a = route[i]
            b = route[i + 1]
            pheromone[a][b] += Q / total
            pheromone[b][a] += Q / total

# 6. Output the best solution
print("Best Route:",
      " -> ".join(names[i] for i in best_route))
print("Shortest Distance:", round(best_distance, 2))
