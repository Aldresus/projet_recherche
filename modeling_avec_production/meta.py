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
EPOCHS = list(range(200, 400, 10))
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

# Define the simulated annealing function
def simulated_annealing(initial_temp, final_temp, alpha, max_iterations):
    if initial_temp < final_temp:
        raise ValueError("Initial temperature must be greater than final temperature.")
    # Initialize the current solution randomly
    current_solution = [
        128, "tanh", 0.2,
        64, "tanh", 0.2,
        64, "relu", 1, "linear",
        0.001, 300, 128
    ]
    current_cost = objective_function(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    temp = initial_temp

    for iteration in range(max_iterations):
        if temp < final_temp:
            break
        
        # Generate a neighboring solution
        neighbor = list(current_solution)
        param_index = random.randint(0, len(current_solution) - 1)
        if param_index == 0:
            neighbor[param_index] = random.choice(LSTM1_units)
        elif param_index == 1:
            neighbor[param_index] = random.choice(LSTM1_activation)
        elif param_index == 2:
            neighbor[param_index] = random.choice(DROPOUT1_rate)
        elif param_index == 3:
            neighbor[param_index] = random.choice(LSTM2_units)
        elif param_index == 4:
            neighbor[param_index] = random.choice(LSTM2_activation)
        elif param_index == 5:
            neighbor[param_index] = random.choice(DROPOUT2_rate)
        elif param_index == 6:
            neighbor[param_index] = random.choice(DENSE1_units)
        elif param_index == 7:
            neighbor[param_index] = random.choice(DENSE1_activation)
        elif param_index == 8:
            neighbor[param_index] = random.choice(DENSE2_units)
        elif param_index == 9:
            neighbor[param_index] = random.choice(DENSE2_activation)
        elif param_index == 10:
            neighbor[param_index] = random.choice(OPTIMIZER_learning_rate)
        elif param_index == 11:
            neighbor[param_index] = random.choice(EPOCHS)
        elif param_index == 12:
            neighbor[param_index] = random.choice(BATCH_SIZE)

        neighbor_cost = objective_function(neighbor)

        # Acceptance criterion
        cost_diff = neighbor_cost - current_cost
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
            current_solution = neighbor
            current_cost = neighbor_cost

            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost

        # Update temperature
        temp = temp * alpha
        print(f"temp {temp}, cost {current_cost}, best_cost {best_cost}")

    return best_solution, best_cost

# Run the simulated annealing algorithm
initial_temp = 20  # Initial temperature
final_temp = 2  # Final temperature
alpha = 0.85  # Temperature decay rate
max_iterations = 10000  # Maximum number of iterations

best_params, best_cost = simulated_annealing(initial_temp, final_temp, alpha, max_iterations)
print("Best parameters:", best_params)
print("Best cost:", best_cost)
