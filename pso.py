# %%
import random
import matplotlib.pyplot as plt
import logging
import time
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


# data = pd.read_csv('../full_dataset.csv')
# data.drop('date', axis=1, inplace=True)

data = pd.read_csv(
    '/home/quentin/Desktop/Optimization/projet/projet_recherche/sampled_dataset.csv')

data.head()

# %%
data_train = data[data['building_id'] != 8]
data_test = data[data['building_id'] == 8]

data_train.shape, data_test.shape

# %%
target_column = 'production'

x_train = data_train.drop(target_column, axis=1)
y_train = data_train[target_column].values.reshape(-1, 1)

x_test = data_test.drop(target_column, axis=1)
y_test = data_test[target_column].values.reshape(-1, 1)

# %%


# %%
x_scaler = MinMaxScaler(feature_range=(0, 1))
x_scaler.fit(x_train)

x_train_scaled = x_scaler.transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

# %%


def get_windows(x, y, window_size):
    x_windows, y_windows = [], []

    for i in range(len(x) - window_size):
        x_window = x[i:i+window_size]
        y_window = y[i:i+window_size]

        x_window = np.hstack((x_window, y_window))

        x_windows.append(x_window)
        y_windows.append(y[i+window_size])

    return np.array(x_windows), np.array(y_windows)


# %%
x_train_windows, y_train_windows = get_windows(x_train_scaled, y_train, 10)
x_test_windows, y_test_windows = get_windows(x_test_scaled, y_test, 10)

# %%

tf.random.set_seed(42)

# %%
model = Sequential([
    LSTM(128, activation='tanh', input_shape=(
        x_train_windows.shape[1:]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),

    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_absolute_error')

model.summary()

# %%
# from codecarbon import EmissionsTracker


# tracker = EmissionsTracker(
#     project_name="3CABTP",
#     co2_signal_api_token="9RkoBO6iipmoq",
#     log_level=logging.INFO,
#     output_file="lstm.csv",
#     output_dir='../emissions/',
#     save_to_file=True,
#     measure_power_secs=10
# )

# %%
def start_training():
    start_time = time.time()
    history = model.fit(x=x_train_windows,
                        y=y_train_windows,
                        epochs=300,
                        batch_size=128,
                        validation_split=0.2,
                        shuffle=False)
    training_duration = time.time() - start_time

    return history, training_duration

# %%
# tracker.start()
# try:
#     history, training_duration = start_training()
# finally:
#     tracker.stop()

# %%
# import matplotlib.pyplot as plt

# start = 10
# plt.plot(history.history['loss'][start:], label='train')
# plt.plot(history.history['val_loss'][start:], label='validation')

# plt.legend()
# plt.show()

# %%


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'mape': mape
    }


# %%
y_pred_train = model.predict(x_train_windows)
y_pred_test = model.predict(x_test_windows)

print(evaluate_model(y_train_windows, y_pred_train))
print(evaluate_model(y_test_windows, y_pred_test))

# %%
x_test_windows.shape

# %%
baseline_predictions = []

for x in x_test_windows:
    baseline_predictions.append(x[-1][-1])

baseline_predictions = np.array(baseline_predictions).reshape(-1, 1)
print(evaluate_model(y_test_windows, baseline_predictions))

# %%
# model.save('../models/lstm_model.keras')

# %%

plt.figure(figsize=(20, 10))

start = 0
end = -1

plt.plot(y_test[start:end], '-^', label='true')
plt.plot(y_pred_test[start:end], '-*', label='predicted')

plt.legend()
plt.grid(True)
plt.show()

# %%
# Hyperparameter optimisation


# Exemple de fonction d'évaluation
# Cette fonction doit être remplacée par le processus d'évaluation de votre modèle

def evaluate_score(hyperparameters):
    # Ici, vous évaluerez votre modèle en utilisant les hyperparamètres
    # et retournerez une métrique de performance, comme l'accuracy
    # Pour cet exemple, on retourne une valeur aléatoire (à remplacer)
    model = Sequential([
        LSTM(hyperparameters[0], activation=hyperparameters[1], input_shape=(
            x_train_windows.shape[1:]), return_sequences=True),
        Dropout(hyperparameters[2]),
        LSTM(hyperparameters[3], activation=hyperparameters[4],
             return_sequences=False),
        Dropout(hyperparameters[5]),
        Dense(hyperparameters[6], activation=hyperparameters[7]),

        Dense(1, activation='linear')
    ])

    optimizer = Adam(learning_rate=hyperparameters[8])
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    model.fit(x=x_train_windows,
              y=y_train_windows,
              epochs=hyperparameters[9],
              batch_size=hyperparameters[10],
              validation_split=0.2,
              shuffle=False)

    y_pred_test = model.predict(x_test_windows)

    print("y_true shape:", y_test_windows.shape)
    print("y_pred shape:", y_pred_test.shape)

    evaluation = evaluate_model(y_test_windows, y_pred_test)

    return evaluation['mse']

# Fonction pour convertir la représentation numérique en chaîne de caractères pour l'hyperparamètre catégoriel


def correct_hyperparameters_value(hyperparameters_value):
    corrected_hyperparameters_value = hyperparameters_value.copy()
    categories = ['tanh', 'relu', 'sigmoid', 'hard_sigmoid', 'linear']

    corrected_hyperparameters_value[0] = int(
        np.round(hyperparameters_value[0]))
    corrected_hyperparameters_value[1] = categories[int(
        hyperparameters_value[1])]
    corrected_hyperparameters_value[3] = int(
        np.round(hyperparameters_value[3]))
    corrected_hyperparameters_value[4] = categories[int(
        hyperparameters_value[4])]
    corrected_hyperparameters_value[6] = int(
        np.round(hyperparameters_value[6]))
    corrected_hyperparameters_value[7] = categories[int(
        hyperparameters_value[7])]
    corrected_hyperparameters_value[9] = int(
        np.round(hyperparameters_value[9]))
    corrected_hyperparameters_value[10] = int(
        np.round(hyperparameters_value[10]))
    print(corrected_hyperparameters_value)

    return corrected_hyperparameters_value

# PSO


class Particle:
    def __init__(self, bounds):
        self.position = np.array(
            [random.uniform(bound[0], bound[1]) for bound in bounds])
        self.velocity = np.array([random.uniform(-1, 1) for _ in bounds])
        self.best_position = self.position.copy()
        self.best_score = -float('inf')

    def update_velocity(self, global_best_position):
        w = 0.5  # inertie
        c1 = 0.8  # cognition (particule)
        c2 = 0.9  # social (essaim)

        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            cognitive_velocity = c1 * r1 * \
                (self.best_position[i] - self.position[i])
            social_velocity = c2 * r2 * \
                (global_best_position[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + \
                cognitive_velocity + social_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        # Appliquer les contraintes de l'espace de recherche
        for i in range(len(self.position)):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]


def pso(num_particles, bounds, num_iterations):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_score = float('inf')
    global_best_position = None

    for _ in range(num_iterations):
        particles_count = 1
        for particle in particles:
            hyperparameters = correct_hyperparameters_value(
                particle.position.copy().tolist())  # Coorection des valeurs des hyperparamètres

            score = evaluate_score(hyperparameters)
            print("iteration : ", _, " / particle_count : ", particles_count)
            print("hyperparameters : ", hyperparameters)
            print("mse : ", score)

            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()

            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()

            particles_count += 1

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position(bounds)

    return global_best_position, global_best_score


# Bounds de chaque hyperparamètre
bounds = [(16, 256),     # units LSTM 1
          (0, 4),        # activation LSTM 1
          (0, 0.99),        # rate dropout1
          (16, 256),     # units LSTM 2
          (0, 4),        # activation LSTM 2
          (0, 0.99),        # rate dropout 2
          (16, 256),     # units DENSE 1
          (0, 4),        # activation DENSE 1
          (0, 0.001),    # learning rate OPTIMIZER
          (200, 400),    # epoch number
          (32, 128)]     # batch size

num_particles = 10
num_iterations = 10

best_position, best_score = pso(num_particles, bounds, num_iterations)
converted_best_position = correct_hyperparameters_value(
    best_position.copy().tolist())
print(f"Meilleure position: {
      converted_best_position}, Meilleur score: {best_score}")
