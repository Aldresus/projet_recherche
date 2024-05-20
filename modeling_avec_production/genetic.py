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

# Define the parameter ranges
LSTM1_units = list(range(1, 1000, 10))

LSTM1_activation = ['tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'linear']
DROPOUT1_rate = list(np.arange(0, 1, 0.01))
LSTM2_units = list(range(1, 1000, 10))
LSTM2_activation = ['tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'linear']
DROPOUT2_rate = list(np.arange(0, 1, 0.01))
DENSE1_units = list(range(1, 1000, 10))
DENSE1_activation = ['tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'linear']
DENSE2_units = [1]
DENSE2_activation = ['linear']
OPTIMIZER_learning_rate = list(np.arange(0, 0.001, 0.0001))
EPOCHS = [10]
#EPOCHS = list(range(200, 400, 10))
BATCH_SIZE = list(range(32, 128, 8))

# Define the objective function (replace with your actual objective function)
def objective_function(params):
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
                            verbose=2)
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

# Define the genetic algorithm functions
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = [
            random.choice(LSTM1_units),
            random.choice(LSTM1_activation),
            random.choice(DROPOUT1_rate),
            random.choice(LSTM2_units),
            random.choice(LSTM2_activation),
            random.choice(DROPOUT2_rate),
            random.choice(DENSE1_units),
            random.choice(DENSE1_activation),
            random.choice(DENSE2_units),
            random.choice(DENSE2_activation),
            random.choice(OPTIMIZER_learning_rate),
            random.choice(EPOCHS),
            random.choice(BATCH_SIZE)
        ]
        population.append(individual)
    return population

def evaluate_fitness(individual):
    fitness = objective_function(individual)
    return fitness

def selection(population, fitness_scores):
    # Perform tournament selection
    tournament_size = 5
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament]
        winner_index = tournament[tournament_fitness.index(min(tournament_fitness))]
        selected_parents.append(population[winner_index])
    return selected_parents

def crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1] if i+1 < len(parents) else parents[0]
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)
    return offspring

def mutation(offspring, mutation_rate):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            param_index = random.randint(0, len(offspring[i]) - 1)
            if param_index == 0:
                offspring[i][param_index] = random.choice(LSTM1_units)
            elif param_index == 1:
                offspring[i][param_index] = random.choice(LSTM1_activation)
            elif param_index == 2:
                offspring[i][param_index] = random.choice(DROPOUT1_rate)
            elif param_index == 3:
                offspring[i][param_index] = random.choice(LSTM2_units)
            elif param_index == 4:
                offspring[i][param_index] = random.choice(LSTM2_activation)
            elif param_index == 5:
                offspring[i][param_index] = random.choice(DROPOUT2_rate)
            elif param_index == 6:
                offspring[i][param_index] = random.choice(DENSE1_units)
            elif param_index == 7:
                offspring[i][param_index] = random.choice(DENSE1_activation)
            elif param_index == 8:
                offspring[i][param_index] = random.choice(DENSE2_units)
            elif param_index == 9:
                offspring[i][param_index] = random.choice(DENSE2_activation)
            elif param_index == 10:
                offspring[i][param_index] = random.choice(OPTIMIZER_learning_rate)
            elif param_index == 11:
                offspring[i][param_index] = random.choice(EPOCHS)
            elif param_index == 12:
                offspring[i][param_index] = random.choice(BATCH_SIZE)
    return offspring

def genetic_algorithm(pop_size, crossover_rate, mutation_rate, generations):
    population = initialize_population(pop_size)
    best_solution = None
    best_fitness = float('inf')

    for generation in range(generations):
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        
        if min(fitness_scores) < best_fitness:
            best_index = fitness_scores.index(min(fitness_scores))
            best_solution = population[best_index]
            best_fitness = min(fitness_scores)
        
        parents = selection(population, fitness_scores)
        offspring = crossover(parents, crossover_rate)
        offspring = mutation(offspring, mutation_rate)
        population = offspring

        print(f"Generation {generation+1}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness

# Run the genetic algorithm
pop_size = 50
crossover_rate = 0.8
mutation_rate = 0.1
generations = 100

best_params, best_fitness = genetic_algorithm(pop_size, crossover_rate, mutation_rate, generations)
print("Best parameters:", best_params)
print("Best fitness:", best_fitness)
