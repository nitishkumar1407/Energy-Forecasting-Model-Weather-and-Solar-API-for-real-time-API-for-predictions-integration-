import streamlit as st
import requests
from datetime import date, time
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .custom-title { font-size: 40px; color: #c7ced9; font-weight: bold; text-align: center; margin-bottom: 20px; }
    .logo{ font-size: 31px; font-weight: bold; color: white; }
    .navbar { display: flex; justify-content: space-between; align-items: center; background-color: #2E86C1; padding: 0.5rem 2rem; border-radius: 5px; position: sticky; top: 0; z-index: 999; }
    .navbar .tabs { display: flex; gap: 1.5rem; }
    .navbar .tabs a { color: white; font-weight: bold; text-decoration: none; font-size: 16px; }
    .navbar .tabs a:hover { color: #F39C12; }
    .kpi-card { background-color: #f5f5f5; border-radius: 10px; padding: 1rem; text-align: center; margin: 5px; }
    .kpi-value { font-size: 28px; font-weight: bold; color: #2E86C1; }
    .kpi-title { font-size: 14px; color: #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="custom-title">Sustainable/Renewable Energy Forecasting</div>',
    unsafe_allow_html=True,
)

# ---------------- NAVBAR ----------------
st.markdown(
    """
    <div class="navbar">
        <div class="logo"> Forecast Rangers</div>
        <div class="tabs">
            <a href="#">Predict Energy</a>
            <a href="#">Historical Analysis</a>
            <a href="#">Renewable Sources</a>
            <a href="#">AI Supports</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- PREDICTION SECTION ----------------
# left column: logo/image, right column: prediction inputs & result
col1, col2 = st.columns([2, 3])

with col1:
    img_path = Path("/Users/nitishkumar/Desktop/forecasting SIH/Solar Model/image/FR.png")
    if img_path.exists():
        # replaced deprecated use_column_width with use_container_width
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning("⚠️ Logo image not found. Please check path.")

with col2:
    # left-align the title inside the right column
    st.markdown(
        "<h2 style='text-align: left; color:#2E86C1; margin-bottom: 0.25rem;'>Predict Energy Consumption</h2>",
        unsafe_allow_html=True,
    )

    city = st.selectbox(
        "Select City",
        [
            "Bhopal",
            "Indore",
            "Gwalior",
            "Jabalpur",
            "Ujjain",
            "Sagar",
            "Rewa",
            "Satna",
            "Ratlam",
            "Dewas",
            "Khargone",
            "Singrauli",
            "Khandwa",
            "Chhindwara",
            "Sehore",
        ],
    )

    selected_date = st.date_input("Select Date", date.today())
    selected_time = st.time_input("Select Time", time(hour=12, minute=0))

    if st.button("Predict Energy Consumption"):
        payload = {
            "city": city,
            "date": selected_date.strftime("%Y-%m-%d"),
            "time": selected_time.strftime("%H:%M"),
        }

        try:
            response = requests.post("http://127.0.0.1:5001/predict", json=payload, timeout=5)

            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    st.error("Backend did not return valid JSON.")
                    st.text(response.text)
                    st.stop()

                st.success(f"Prediction Successful for {city} at {selected_time.strftime('%H:%M')}")

                # Safely extract numeric values (default 0)
                predicted_consumption = float(result.get("Total_Consumption_kWh") or 0)
                solar_contribution = float(result.get("Solar_Contribution_kWh") or 0)
                net_demand = float(result.get("Net_Demand_kWh") or 0)

                # Show KPIs using columns
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                    st.markdown(
                        f"<div class='kpi-card'><div class='kpi-title'>{city} — Predicted Consumption</div><div class='kpi-value'>{predicted_consumption:.2f} kWh</div></div>",
                        unsafe_allow_html=True,
                    )
                with kpi2:
                    st.markdown(
                        f"<div class='kpi-card'><div class='kpi-title'>{city} — Solar Contribution</div><div class='kpi-value'>{solar_contribution:.2f} kWh</div></div>",
                        unsafe_allow_html=True,
                    )
                with kpi3:
                    st.markdown(
                        f"<div class='kpi-card'><div class='kpi-title'>{city} — Net Demand</div><div class='kpi-value'>{net_demand:.2f} kWh</div></div>",
                        unsafe_allow_html=True,
                    )

                # Hourly chart simulation for 24h
                hours = np.arange(0, 24)
                hourly_consumption = [predicted_consumption / 24.0] * 24
                # put solar contribution at the selected hour (you can change logic as needed)
                hourly_solar = [solar_contribution if h == selected_time.hour else 0 for h in hours]
                hourly_net = [max(0, c - s) for c, s in zip(hourly_consumption, hourly_solar)]

                df_hourly = pd.DataFrame(
                    {
                        "Hour": [f"{h}:00" for h in hours],
                        "Predicted Consumption": hourly_consumption,
                        "Solar Contribution": hourly_solar,
                        "Net Demand": hourly_net,
                    }
                )

                fig = px.bar(
                    df_hourly,
                    x="Hour",
                    y=["Predicted Consumption", "Solar Contribution", "Net Demand"],
                    barmode="group",
                    title=f"{city} Hourly Consumption vs Solar on {selected_date.strftime('%Y-%m-%d')}",
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"Backend returned error {response.status_code}")
                st.text(response.text)

        except requests.exceptions.RequestException as e:
            st.warning(f"Could not connect to backend: {e}")
