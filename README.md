# ⚡ Energy Forecasting Using Machine Learning

This project predicts **city-level daily energy consumption** using machine learning with **real-time weather and solar API data**. The system fetches environmental data and generates energy consumption predictions in **kilowatt-hours (kWh)**.

It includes a **machine learning model, Flask backend API, and Streamlit dashboard** for interactive forecasting.

---

# 🚀 Key Features

* Stacked ML model for energy prediction
* Real-time **Weather API** integration
* **Solar radiation API** integration
* Flask backend for prediction service
* Streamlit interactive dashboard
* SQLite database to store prediction history
* Model retraining capability

---

# 📁 Project Structure

```
Energy-Forecasting/
│
├── ML_model.py          # Model training and saving (model.pkl)
├── backend.py           # Flask prediction API
├── app.py               # Streamlit dashboard
├── model.pkl            # Trained ML model
├── energy_prediction.db # SQLite database
├── dataset.csv          # Training dataset
├── requirements.txt
└── .gitignore
```

---

# 🛠 Tech Stack

* Python
* Scikit-learn
* XGBoost
* Flask
* Streamlit
* SQLite
* Weather API
* Solar API
# 🧠 Machine Learning Model

The project uses a **Stacking Regressor** combining multiple models.

**Base Models**

* RandomForestRegressor
* XGBoost Regressor

**Meta Model**

* Linear Regression with StandardScaler

The trained model is saved as:

```
model.pkl
```

---

# 🌤 Real-Time Data Integration

Environmental data is fetched from external APIs:

**Weather API**

* Temperature
* Weather conditions

**Solar API**

* Solar radiation
* Solar intensity

These features improve energy demand prediction accuracy.

---

# 📊 Dataset

The dataset contains historical data from **2019–2024**, including:

* City-wise energy consumption
* Daily temperature
* Date features

Additional time features such as **month and day-of-week** are automatically generated.

---

# 🖥 User Interface

A **Streamlit dashboard** allows users to:

* Select city
* Choose date
* Fetch real-time weather data
* Predict energy consumption instantly

---

# 🗃 Database

Predictions are stored in a **SQLite database**:

```
energy_prediction.db
```

Stored fields include city, date, temperature, solar data, and predicted consumption.

---

# 🔁 Model Retraining

To retrain the model with updated data:

```
python ML_model.py
```

This retrains the model and updates:

```
model.pkl
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/nitishkumar1407/Energy-Forecasting-Model-Weather-and-Solar-API-for-real-time-API-for-predictions-integration-.git
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the backend API:

```
python backend.py
```

Run the Streamlit dashboard:

```
streamlit run app.py
```

