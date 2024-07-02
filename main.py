import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import time


# Function to load the dataset from a file
def load_data(file_path):
    df = pd.read_csv(file_path, header=None, sep=r'\s+')
    df.columns = ['rate1', 'rate2']  # Assign column names
    return df


# Function to create features and targets from the dataset
def prepare_features_and_targets(df, lag):
    X, y = [], []
    for i in range(lag, len(df)):
        X.append(df.iloc[i - lag:i, 0].values)  # Collect past values as features
        y.append(df.iloc[i, 1])  # Collect current value as target
    return np.array(X), np.array(y)


# Function to train and evaluate the Ridge Regression model
def train_and_evaluate_ridge(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge()

    # Measure training time
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Measure prediction time
    start_time = time.time()
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    prediction_time = time.time() - start_time

    # Calculate RMSE for training and testing data
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return train_rmse, test_rmse, y_train_pred, y_test_pred, training_time, prediction_time


# Load the dataset
file_path = 'gas_datasets.txt'
data = load_data(file_path)

# Split the dataset into training and testing sets
split = len(data) // 2
train_set = data.iloc[:split]
test_set = data.iloc[split:]

# Number of past values to consider for features
lag = 5

# Create training and testing features and targets
X_train, y_train = prepare_features_and_targets(train_set, lag)
X_test, y_test = prepare_features_and_targets(test_set, lag)

# Train and evaluate the Ridge Regression model
train_rmse, test_rmse, y_train_pred, y_test_pred, training_time, prediction_time = train_and_evaluate_ridge(X_train,
                                                                                                            y_train,
                                                                                                            X_test,
                                                                                                            y_test)

# Print the results
print(
    f"Ridge Regression: Train RMSE = {train_rmse}, Test RMSE = {test_rmse}, Training Time = {training_time}s, Prediction Time = {prediction_time}s")

# Plot predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(train_set.index[lag:], y_train, 'o-', label='Actual (Train)', markersize=5, color='blue')
plt.plot(test_set.index[lag:], y_test, 'd-.', label='Actual (Test)', markersize=5, color='red')
plt.plot(train_set.index[lag:], y_train_pred, label='Predicted (Train)', linestyle='--', color='green')
plt.plot(test_set.index[lag:], y_test_pred, label='Predicted (Test)', linestyle='--', color='purple')

plt.xlabel('Time', fontsize=12)
plt.ylabel('Rate', fontsize=12)
plt.title('Predictions vs Actual Values (Ridge Regression)', fontsize=15)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
