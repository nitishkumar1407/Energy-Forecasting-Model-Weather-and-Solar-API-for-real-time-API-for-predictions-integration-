# energy_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# Load Dataset
data = pd.read_csv(r"/Users/nitishkumar/Desktop/forecasting SIH/Solar Model/adjusted_city_scale_energy_dataset_cleaned.csv")
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

hourly_cols = [col for col in data.columns if "-" in col]
melted = data.melt(
    id_vars=['City', 'Year', 'Month', 'Date', 'Temperature (°C)'],
    value_vars=hourly_cols,
    var_name='Hour',
    value_name='Consumption_kWh'
)

# Extract date features
melted['month'] = melted['Date'].dt.month
melted['day'] = melted['Date'].dt.day
melted['dayofweek'] = melted['Date'].dt.dayofweek
melted['dayofyear'] = melted['Date'].dt.dayofyear
melted['hour'] = melted['Hour'].str.split(" ").str[0].replace({'12':'0'}).astype(int)

# One-hot encode City
city_dummies = pd.get_dummies(melted['City'], prefix='City', drop_first=False)

# Features & Target
X = pd.concat([
    melted[['month', 'day', 'dayofweek', 'dayofyear', 'hour', 'Temperature (°C)']],
    city_dummies
], axis=1)
y = melted['Consumption_kWh']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
xgb = XGBRegressor(n_estimators=50, n_jobs=-1, random_state=42)
meta_model = make_pipeline(StandardScaler(), LinearRegression())

stack_model = StackingRegressor(
    estimators=[('rf', rf), ('xgb', xgb)],
    final_estimator=meta_model
)

# Train
stack_model.fit(X_train, y_train)

# Evaluate
y_pred = stack_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Save model
joblib.dump(stack_model, "model.pkl")
print("Model saved as model.pkl")