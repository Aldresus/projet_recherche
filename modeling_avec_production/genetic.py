import random
import numpy as np

import pandas as pd

data = pd.read_csv('/Users/hugochampy/Documents/le_code_la/Optimization/sampled_dataset.csv')
data.head()

data_train = data[data['building_id'] != 8]
data_test = data[data['building_id'] == 8]

data_train.shape, data_test.shape


target_column = 'production'

x_train = data_train.drop(target_column, axis=1)
y_train = data_train[target_column].values.reshape(-1, 1)

x_test = data_test.drop(target_column, axis=1)
y_test = data_test[target_column].values.reshape(-1, 1)


import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


x_scaler = MinMaxScaler(feature_range=(0, 1))
x_scaler.fit(x_train)

x_train_scaled = x_scaler.transform(x_train)
x_test_scaled = x_scaler.transform(x_test)


def get_windows(x, y, window_size):
    x_windows, y_windows = [], []

    for i in range(len(x) - window_size):
        x_window = x[i:i+window_size]
        y_window = y[i:i+window_size]

        x_window = np.hstack((x_window, y_window))

        x_windows.append(x_window)
        y_windows.append(y[i+window_size])

    return np.array(x_windows), np.array(y_windows)


x_train_windows, y_train_windows = get_windows(x_train_scaled, y_train, 10)
x_test_windows, y_test_windows = get_windows(x_test_scaled, y_test, 10)


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

import tensorflow as tf
tf.random.set_seed(42)





import time
import logging
from codecarbon import EmissionsTracker


# tracker = EmissionsTracker(
#     project_name="3CABTP",
#     co2_signal_api_token="9RkoBO6iipmoq",
#     log_level=logging.INFO,
#     output_file="lstm.csv",
#     output_dir='/Users/hugochampy/Documents/le_code_la/Optimization/emissions/',
#     save_to_file=True,
#     measure_power_secs=10
# )

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print(f"MSE: {mse}")
    return  mse


import random
import math
import numpy as np

# Define the range of hyperparameters to search
PARAM_RANGES = {
    'param1': (0.1, 1.0),
    'param2': (10, 100),
    'param3': (0.001, 0.1)
}

# Define the population size and number of generations
POPULATION_SIZE = 50
NUM_GENERATIONS = 100

# Define the mutation rate and crossover rate
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# Define the fitness function (replace with your actual evaluation function)
def fitness_function(params):
    """
    Evaluate the fitness of an individual (set of hyperparameters)
    by training and evaluating your model.
    Return a fitness score (higher is better).
    """
    print(f"params {params}")
    # Unpack the parameters
    lstm1_units, lstm1_activation, dropout1_rate, lstm2_units, lstm2_activation, dropout2_rate, dense1_units, dense1_activation, dense2_units, dense2_activation, optimizer_learning_rate, epochs, batch_size = params

    # Implement your model training and evaluation here
    # ...

    #place holder for the performance metric return a random number
    model = Sequential([
    LSTM(lstm1_units, activation=lstm1_activation, input_shape=(
        x_train_windows.shape[1:]), return_sequences=True),
    #lstm params :
    #     units
    #     activation
    Dropout(dropout1_rate),
    #params
    #     rate



    LSTM(lstm2_units, activation=lstm2_activation, return_sequences=False),
    Dropout(dropout2_rate),
    Dense(dense1_units, activation=dense1_activation),
    #params
    #     units
    #     activation

    Dense(dense2_units, activation=dense2_activation)
])

    optimizer = Adam(learning_rate=optimizer_learning_rate)
    # params
    #       learning_rate
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    # tracker.start()
    try:

        start_time = time.time()
        history = model.fit(x=x_train_windows,
                            y=y_train_windows,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2,
                            shuffle=False,
                            verbose=0)
        training_duration = time.time() - start_time
    finally:
        print("finito pipo")
        # tracker.stop()



    y_pred_train = model.predict(x_train_windows)
    y_pred_test = model.predict(x_test_windows)

    print(evaluate_model(y_train_windows, y_pred_train))
    performance_metric = evaluate_model(y_test_windows, y_pred_test)


    # Return the performance metric (e.g., accuracy, loss) to be minimized
    return performance_metric

# Initialize the population
population = []
for _ in range(POPULATION_SIZE):
    individual = {param: random.uniform(*PARAM_RANGES[param]) for param in PARAM_RANGES}
    population.append(individual)

# Genetic algorithm loop
for generation in range(NUM_GENERATIONS):
    # Evaluate the fitness of each individual
    fitness_scores = [fitness_function(individual) for individual in population]

    # Select parents for the next generation
    parents = []
    for _ in range(POPULATION_SIZE // 2):
        parent1 = random.choices(population, weights=fitness_scores, k=1)[0]
        parent2 = random.choices(population, weights=fitness_scores, k=1)[0]
        parents.append(parent1)
        parents.append(parent2)

    # Create the next generation
    next_generation = []
    for i in range(0, POPULATION_SIZE, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]

        # Crossover
        if random.random() < CROSSOVER_RATE:
            child1 = {}
            child2 = {}
            for param in PARAM_RANGES:
                if random.random() < 0.5:
                    child1[param] = parent1[param]
                    child2[param] = parent2[param]
                else:
                    child1[param] = parent2[param]
                    child2[param] = parent1[param]
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        # Mutation
        for child in [child1, child2]:
            for param in PARAM_RANGES:
                if random.random() < MUTATION_RATE:
                    child[param] = random.uniform(*PARAM_RANGES[param])

        next_generation.append(child1)
        next_generation.append(child2)

    # Replace the old population with the new generation
    population = next_generation

# Find the best individual from the final population
best_individual = max(population, key=fitness_function)
print("Best hyperparameters found:", best_individual)