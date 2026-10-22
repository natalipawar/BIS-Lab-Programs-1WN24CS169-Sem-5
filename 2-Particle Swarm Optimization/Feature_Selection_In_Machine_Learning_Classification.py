
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split into training, validation and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# PSO parameters
N = 15
ITER = 30
w = 0.7
c1 = 1.5
c2 = 1.5
num_features = X.shape[1]

# Initialize particles
position = np.random.randint(0, 2, (N, num_features))
velocity = np.random.uniform(-1, 1, (N, num_features))

pbest = position.copy()
pbest_score = np.full(N, float("inf"))

gbest = position[0].copy()
gbest_score = float("inf")

# Evaluate feature subset
def fitness(mask):
    if np.sum(mask) == 0:
        return float("inf")

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train[:, mask == 1], y_train)

    predictions = model.predict(X_val[:, mask == 1])
    accuracy = accuracy_score(y_val, predictions)

    error = 1 - accuracy
    feature_ratio = np.sum(mask) / num_features

    # Minimize classification error and feature count
    return error + 0.1 * feature_ratio

# Binary PSO
for iteration in range(ITER):
    for i in range(N):
        if np.sum(position[i]) == 0:
            position[i, np.random.randint(num_features)] = 1

        score = fitness(position[i])

        if score < pbest_score[i]:
            pbest_score[i] = score
            pbest[i] = position[i].copy()

        if score < gbest_score:
            gbest_score = score
            gbest = position[i].copy()

    for i in range(N):
        r1 = np.random.rand(num_features)
        r2 = np.random.rand(num_features)

        velocity[i] = (
            w * velocity[i]
            + c1 * r1 * (pbest[i] - position[i])
            + c2 * r2 * (gbest - position[i])
        )

        probability = 1 / (1 + np.exp(-np.clip(velocity[i], -10, 10)))
        position[i] = (np.random.rand(num_features) < probability).astype(int)

# Evaluate selected features on test data
selected = np.where(gbest == 1)[0]

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train[:, selected], y_train)

predictions = model.predict(X_test[:, selected])
accuracy = accuracy_score(y_test, predictions)

print("Total features:", num_features)
print("Selected features:", len(selected))
print("Selected feature names:", feature_names[selected])
print("Test classification accuracy:", round(accuracy * 100, 2), "%")
