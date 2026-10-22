import numpy as np

np.random.seed(42)

# Illustrative monthly returns of 4 stocks
returns = np.array([
    [0.04, 0.02, 0.05, 0.01],
    [0.02, 0.03, 0.01, 0.02],
    [-0.03, 0.01, -0.02, 0.00],
    [0.05, 0.04, 0.03, 0.02],
    [0.01, -0.02, 0.04, 0.01],
    [0.03, 0.02, 0.02, 0.03],
    [-0.01, 0.01, -0.03, 0.02],
    [0.04, 0.03, 0.05, 0.01],
    [0.02, 0.00, 0.01, 0.02],
    [0.03, 0.04, 0.02, 0.01],
    [-0.02, 0.02, 0.00, 0.01],
    [0.05, 0.01, 0.04, 0.03]
])

# Annualized estimates
mean_returns = returns.mean(axis=0) * 12
cov_matrix = np.cov(returns, rowvar=False) * 12
risk_free_rate = 0.03

N = 40
GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# Generate valid portfolio weights
def normalize(weights):
    weights = np.maximum(weights, 0)
    if weights.sum() == 0:
        weights = np.ones(len(weights))
    return weights / weights.sum()

def portfolio_metrics(weights):
    ret = np.dot(weights, mean_returns)
    variance = weights @ cov_matrix @ weights
    risk = np.sqrt(max(variance, 0))
    sharpe = (ret - risk_free_rate) / (risk + 1e-12)
    return ret, variance, risk, sharpe

# GA fitness for each objective
def fitness(weights, objective):
    ret, variance, risk, sharpe = portfolio_metrics(weights)

    if objective == "GMV":
        return -variance       # Minimize variance
    else:
        return sharpe          # Maximize Sharpe ratio

def run_ga(objective):
    population = np.array([
        normalize(np.random.random(4)) for _ in range(N)
    ])

    best = None
    best_fitness = -float("inf")

    for generation in range(GENERATIONS):
        scores = np.array([
            fitness(ind, objective) for ind in population
        ])

        idx = np.argmax(scores)

        if scores[idx] > best_fitness:
            best_fitness = scores[idx]
            best = population[idx].copy()

        # Keep the best individual (elitism)
        new_population = [best.copy()]

        while len(new_population) < N:
            # Selection: tournament selection
            def select_parent():
                candidates = np.random.choice(N, 3, replace=False)
                winner = candidates[np.argmax(scores[candidates])]
                return population[winner].copy()

            p1 = select_parent()
            p2 = select_parent()

            # Crossover
            if np.random.random() < CROSSOVER_RATE:
                a = np.random.random()
                child = a * p1 + (1 - a) * p2
            else:
                child = p1.copy()

            # Mutation
            for i in range(len(child)):
                if np.random.random() < MUTATION_RATE:
                    child[i] += np.random.normal(0, 0.1)

            new_population.append(normalize(child))

        population = np.array(new_population)

    return best

# Objective 1: Global Minimum Variance
gmv = run_ga("GMV")

# Objective 2: Maximum Sharpe Ratio
msr = run_ga("MSR")

for name, weights in [
    ("GMV Portfolio", gmv),
    ("Maximum Sharpe Portfolio", msr)
]:
    ret, variance, risk, sharpe = portfolio_metrics(weights)

    print("\n", name)
    print("Stock Weights (%):", np.round(weights * 100, 2))
    print("Expected Annual Return (%):", round(ret * 100, 2))
    print("Annual Risk (%):", round(risk * 100, 2))
    print("Sharpe Ratio:", round(sharpe, 4))
