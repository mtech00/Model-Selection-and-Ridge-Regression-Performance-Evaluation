import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import time
import timeit


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


# Function to train and evaluate models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Measure training time
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Measure prediction time
    prediction_time = timeit.timeit(lambda: model.predict(X_test_scaled), number=1)

    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

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

# Define the models to be evaluated
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                             random_state=42),
    'KNeighbors Regressor': KNeighborsRegressor(),
    'SVR': SVR(kernel='rbf')
}

results = []
predictions = {}  # To store the predictions for plotting

# Train and evaluate each model
for model_name, model in models.items():
    # Train and evaluate the model
    train_rmse, test_rmse, y_train_pred, y_test_pred, training_time, prediction_time = train_and_evaluate_model(model,
                                                                                                                X_train,
                                                                                                                y_train,
                                                                                                                X_test,
                                                                                                                y_test)
    # Store the results
    results.append({
        'Model': model_name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Training Time (s)': training_time,
        'Prediction Time (s)': prediction_time
    })
    # Store the predictions for plotting
    predictions[model_name] = {
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }
    # Print the results
    print(
        f"{model_name}: Train RMSE = {train_rmse}, Test RMSE = {test_rmse}, Training Time = {training_time}s, Prediction Time = {prediction_time}s")

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Normalize the values for comparison
results_df['Normalized Test RMSE'] = results_df['Test RMSE'] / results_df['Test RMSE'].max()
results_df['Normalized Training Time'] = results_df['Training Time (s)'] / results_df['Training Time (s)'].max()
results_df['Normalized Prediction Time'] = results_df['Prediction Time (s)'] / results_df['Prediction Time (s)'].max()

# Calculate the total score as the sum of normalized values
results_df['Total Score'] = results_df['Normalized Test RMSE'] + results_df['Normalized Training Time'] + results_df[
    'Normalized Prediction Time']

# Sort the DataFrame by Total Score to find the best model
sorted_results_df = results_df.sort_values(by='Total Score')

# Display the sorted DataFrame
print("\nModel Performance Sorted by Total Score:")
print(sorted_results_df)

# Determine the model with the lowest Total Score
best_model_row = sorted_results_df.loc[sorted_results_df['Total Score'].idxmin()]
best_model_name = best_model_row['Model']
print(f"\nBest Model (based on Total Score): {best_model_name}")

# Display the best model's results
print(f"\nBest Model Results:\n{best_model_row}")

# Plot predictions vs actual values for the best model
plt.figure(figsize=(14, 7))
plt.plot(train_set.index[lag:], y_train, 'o-', label='Actual (Train)', markersize=5, color='blue')
plt.plot(test_set.index[lag:], y_test, 'd-.', label='Actual (Test)', markersize=5, color='red')

best_model_predictions = predictions[best_model_name]
plt.plot(train_set.index[lag:], best_model_predictions['y_train_pred'], label=f'Predicted ({best_model_name} - Train)',
         linestyle='--')
plt.plot(test_set.index[lag:], best_model_predictions['y_test_pred'], label=f'Predicted ({best_model_name} - Test)',
         linestyle='--')

plt.xlabel('Time', fontsize=12)
plt.ylabel('Rate', fontsize=12)
plt.title(f'Predictions vs Actual Values for {best_model_name}', fontsize=15)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
