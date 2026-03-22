# ⚡ Energy Forecasting Using Machine Learning with Real-Time Weather & Solar API Integration

This project predicts **city-level daily energy consumption** using machine learning and **real-time weather and solar radiation data**. The system integrates external APIs to fetch live environmental data and uses an ensemble ML model to forecast electricity demand.

Users can input a **city and date**, and the system automatically retrieves **temperature and solar data from APIs** to generate accurate **energy consumption predictions in kilowatt-hours (kWh)**.

The project demonstrates an **end-to-end ML pipeline**, including model training, real-time API integration, backend deployment, and interactive visualization.

---

# 🚀 Key Components

The system includes:

* Machine Learning pipeline using a stacked ensemble model
* Real-time Weather API integration for temperature data
* Solar radiation API integration for solar intensity
* Flask backend API to serve predictions
* Streamlit frontend dashboard for interactive forecasting
* SQLite database to store prediction logs
* Model retraining capability for continuous improvement

---

# 📁 Project Structure

```
Energy-Forecasting/
│
├── ML_model.py
│   └── Model training pipeline and saving trained model (model.pkl)
│
├── backend.py
│   └── Flask API that receives inputs and returns energy predictions
│
├── app.py
│   └── Streamlit web application for user interaction
│
├── model.pkl
│   └── Trained stacked regression model used for predictions
│
├── energy_prediction.db
│   └── SQLite database storing prediction history
│
├── datewise_predicted_consumption_temperature.csv
│   └── Training dataset containing historical energy and temperature data
│
├── requirements.txt
│   └── Python dependencies
│
└── .gitignore
```

---

# 📌 Tech Stack

* Python
* Scikit-learn
* XGBoost
* Flask
* Streamlit
* SQLite
* Weather APIs
* Solar APIs

# 🧠 Machine Learning Model

The system uses a **Stacking Regressor** to combine multiple machine learning models for improved forecasting performance.

### Base Models

* RandomForestRegressor
* XGBRegressor (XGBoost)

### Meta Model

* LinearRegression wrapped with StandardScaler using a pipeline

This hybrid approach allows the system to capture both:

* Non-linear relationships (via Random Forest & XGBoost)
* Linear patterns (via Linear Regression)

The final trained model is saved as:

```
model.pkl
```

and is used for **real-time predictions via the backend API**.

---

# 🌤 Real-Time Data Integration

The system integrates external APIs to improve prediction accuracy.

### Weather API

Provides real-time environmental data such as:

* Temperature
* Humidity
* Weather conditions

### Solar API

Provides solar energy related metrics such as:

* Solar radiation
* Solar intensity
* Cloud coverage impact

These environmental factors are used as **input features for the ML model**, enabling more realistic energy consumption forecasts.

---

# 📊 Dataset

The training dataset spans from **2019 to 2024** and includes:

* City-wise energy consumption
* Daily temperature values
* Dates (used for feature engineering)
* City identifiers

From the date column, additional features are automatically generated:

* Month
* Day
* Day of week
* Seasonal patterns

> Note: The dataset was synthetically generated to simulate real-world energy demand patterns across cities in **Madhya Pradesh, India**.

---

# ✨ Features

✔ Ensemble ML Model (Stacked Regressor)
✔ Real-time Weather API Integration
✔ Solar Radiation API Integration
✔ Automatic feature engineering from date
✔ One-hot encoding for city-based forecasting
✔ Flask REST API for predictions
✔ Streamlit interactive dashboard
✔ SQLite database logging prediction history
✔ Multi-city energy demand forecasting
✔ Model evaluation using **MAE, RMSE, and R²**

---

# 🖥 User Interface

The application includes an **interactive Streamlit dashboard** where users can:

* Select a city
* Choose a date
* Fetch real-time weather and solar data
* Generate energy consumption predictions instantly

### Screenshot

```
images/Dashboard.jpeg
```

---

# 🗃 Database

The system stores prediction results in a **SQLite database**.

Stored information includes:

* City
* Date
* Temperature
* Solar radiation
* Predicted energy consumption
* Optional actual consumption for model evaluation

Database file:

```
energy_prediction.db
```

This enables **historical analysis and model improvement**.

---

# 🔁 Model Retraining

The system supports **model retraining** to improve forecasting performance as more data becomes available.

Retraining can be performed by running:

```bash
python ML_model.py
```

This will:

1. Load the updated dataset
2. Train the stacking regression model
3. Evaluate performance metrics
4. Save the updated model as:

```
model.pkl
```

The backend API will automatically use the updated model for future predictions.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/nitishkumar1407/Energy-Forecasting-Model-Weather-and-Solar-API-for-real-time-API-for-predictions-integration-.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have **Python 3.7+ installed**.

---

# ▶️ Running the Application

### Start Backend API

```bash
python backend.py
```

### Launch Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 🔮 Future Enhancements

* Live deployment using Docker or cloud platforms
* Integration with power grid / DISCOM datasets
* Advanced time-series models (LSTM / Prophet)
* Real-time energy demand dashboards
* Automated daily model retraining pipeline


