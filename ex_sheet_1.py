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
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 5,
) -> np.int:

    # TODO Write the main algorithm here
    num_train_data, num_features = X_train.shape
    distances = np.empty([num_train_data])
    
    for i in range (0,num_train_data):
        for j in range (0, num_features):
            euc_distance = query_example - X_train[i,:] 
            distances[i] = np.sqrt(sum(np.square(euc_distance),1))

    sorted_index = np.argsort(distances)
    sorted_distances = distances[sorted_index] 
    
    predictions = y_train[sorted_index] 
    if sum(predictions[0:k]) > 2:
        return 1
    else:
        return 0

    
    

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
    print("predicted: ", y_pred, "label: ", label)
    if y_pred == label:
        nr_correct_pred += 1

accuracy = nr_correct_pred / X_test.shape[0]
print(f'The accuracy of the k-nearest neighbor algorithm with k = {NR_NEIGHBORS} is {accuracy}')
