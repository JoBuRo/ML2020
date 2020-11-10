import numpy as np
import openml


def euclidean_distance(
        example_1: np.ndarray,
        example_2: np.ndarray,
) -> np.float:

    sum_squared_distance = 0
    nr_features = example_1.shape[0]
    for feature_index in range(0, nr_features):
        sum_squared_distance += np.power(example_1[feature_index] - example_2[feature_index], 2)

    return np.sqrt(sum_squared_distance)


def knn(
    query_example: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 5,
) -> np.int:
    neighbors = []
    for i in range(len(x_train)):
        neighbors += [(y_train[i], euclidean_distance(x_train[i], query_example))]

    neighbors.sort(key=lambda x: x[1])
    return sum([x[0] for x in neighbors[:k]]) > k / 2

# Exercise 1
# For exercise 1, we are going to use the diabetes dataset
task = openml.tasks.get_task(267)
train_indices, test_indices = task.get_train_test_split_indices()
dataset = task.get_dataset()
X, y, categorical_indicator, _ = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute,
)

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

NR_NEIGHBORS = 5

nr_correct_pred = 0
for example, label in zip(X_test, y_test):
    y_pred = knn(example, X_train, y_train, k=NR_NEIGHBORS)
    if y_pred == label:
        nr_correct_pred += 1

accuracy = nr_correct_pred / X_test.shape[0]
print(f'The accuracy of the k-nearest neighbor algorithm with k = {NR_NEIGHBORS} is {accuracy}')
