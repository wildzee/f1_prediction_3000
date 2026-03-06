import streamlit as st
import pandas as pd
import plotly.express as px
import time

from src.constants import load_calendar, TEAM_COLORS, CLEAN_AIR_PACE
from src.data_loader import (
    get_historical_race_data, get_2026_preseason_data, get_driver_grid_2026,
    get_circuit_baseline, load_session_results, get_live_practice_data, get_live_qualifying_data
)
from src.features import engineer_features
from src.model import train_model, predict_race, get_feature_importances
from src.weather import get_weather_forecast

st.set_page_config(page_title="Race Predictor", page_icon="🏁", layout="wide")

st.markdown("""
<style>
    h1, h2, h3 { color: #E8002D; }
    .stButton>button { background-color: #E8002D; color: white; border-radius: 4px; font-weight: bold; }
    .stButton>button:hover { background-color: #FF1A46; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🏁 F1 Prediction 3000: Race Predictor")

# Load Calendar
@st.cache_data
def get_cal():
    return load_calendar()

calendar = get_cal()
race_names = [f"Round {r['round']}: {r['name']}" for r in calendar]

# --- CONTROLS ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Select Race")
    selected_race_str = st.selectbox("Grand Prix", race_names)
    
    selected_idx = race_names.index(selected_race_str)
    race_info = calendar[selected_idx]
    
    country_name = race_info['name'].replace(" Grand Prix", "")
    fastf1_race_map = {
        "Madrid": "Spain", "Australian": "Australia", "Chinese": "China",
        "Japanese": "Japan", "Saudi Arabian": "Saudi Arabia", "Canadian": "Canada",
        "Spanish": "Spain", "British": "Great Britain", "Hungarian": "Hungary",
        "Belgian": "Belgium", "Dutch": "Netherlands", "Italian": "Italy",
        "Mexican": "Mexico", "Brazilian": "Brazil",
    }
    historical_race_target = fastf1_race_map.get(country_name, country_name)
    
    st.markdown(f"**Date:** {race_info['date']}")
    st.markdown(f"**Circuit:** {historical_race_target}")

with col2:
    st.subheader("🌤️ Weather Forecast")
    weather = get_weather_forecast(race_info['lat'], race_info['lon'], race_info['date'])
    m1, m2, m3 = st.columns(3)
    m1.metric("Temperature", f"{weather['temp']}°C")
    m2.metric("Rain Probability", f"{weather['pop']*100:.0f}%")
    m3.metric("Conditions", weather.get('description', 'N/A').title())

st.divider()

# --- DATA AVAILABILITY CHECK ---
# Priority: 1. Live Qualifying  2. Live Practice  3. Testing  4. Nothing → block
with st.spinner("Checking FastF1 for live session data..."):
    # 1. Try real qualifying from FastF1
    live_q = get_live_qualifying_data(2026, historical_race_target)
    
    # 2. Try predicted qualifying saved from the Qualifying Predictor page
    saved_q = load_session_results(race_info['round'], "Q")
    
    # 3. Try live practice from FastF1
    live_p, p_session = get_live_practice_data(2026, historical_race_target)
    
    # 4. Testing data
    preseason = get_2026_preseason_data()

# Determine exactly which source to use based on strict hierarchy
quali_input = None

if not live_q.empty:
    data_source = "qualifying"
    data_source_label = "✅ Real Qualifying Data (FastF1)"
    st.success("**Data Source: Real Qualifying** — Best possible prediction accuracy.")
    quali_input = live_q.copy()
elif not saved_q.empty:
    data_source = "qualifying_saved"
    data_source_label = "✅ Predicted Qualifying Data (Saved)"
    st.success("**Data Source: Predicted Qualifying** — using the qualifying order you predicted and saved on the other page.")
    quali_input = saved_q.rename(columns={"QTime (s)": "QualifyingTime (s)"})
elif not live_p.empty:
    data_source = "practice"
    data_source_label = f"⚠️ Practice Data ({p_session})"
    st.warning(f"**Data Source: {p_session}** — Qualifying data not available. Using {p_session} lap times as qualifying estimates.")
    quali_input = live_p.rename(columns={"PracticeTime (s)": "QualifyingTime (s)"})
    # Practice is typically 1 second slower than qualifying on average
    quali_input["QualifyingTime (s)"] = quali_input["QualifyingTime (s)"] - 1.0
elif not preseason.empty:
    data_source = "testing"
    data_source_label = "📋 Pre-Season Testing Data"
    st.info("**Data Source: Pre-Season Testing** — No live data available yet for this weekend. Using pre-season baselines.")
else:
    data_source = None
    st.error("❌ **Cannot predict** — No data available at all.")

st.caption(f"Data priority enforced: Real Qualifying → Predicted Qualifying → Practice → Testing")

# --- PREDICTION ENGINE ---
if data_source is not None:
    run_btn = st.button("🚀 Run Prediction Engine", use_container_width=True)
else:
    run_btn = False

if run_btn:
    with st.spinner(f"Loading 2025 {historical_race_target} historical data..."):
        start_time = time.time()
        
        hist_data, raw_laps = get_historical_race_data(2025, historical_race_target, "R")
        circuit_baseline = get_circuit_baseline(2025, historical_race_target)
        grid_2026 = get_driver_grid_2026()
        
        # Engineer features
        df_full, X_features = engineer_features(
            hist_data, preseason, grid_2026, weather,
            qualifying_data=quali_input,
            circuit_baseline=circuit_baseline
        )
    
    with st.spinner("Training XGBoost model..."):
        if not hist_data.empty and not raw_laps.empty:
            y_target = raw_laps.groupby("Driver")["LapTime (s)"].mean()
            valid_mask = df_full["DriverCode"].isin(y_target.index)
            training_df = df_full[valid_mask].copy()
            
            if len(training_df) >= 3:
                y_train = y_target.reindex(training_df["DriverCode"]).values
                X_train = X_features.loc[training_df.index]
                model, mae = train_model(X_train, y_train)
                preds = predict_race(model, X_features)
            else:
                model = None
                mae = 0.0
                preds = df_full["CleanAirRacePace (s)"] - (df_full["TeamPerformanceScore"] * 1.5)
        else:
            model = None
            mae = 0.0
            preds = df_full["CleanAirRacePace (s)"] - (df_full["TeamPerformanceScore"] * 1.5)
        
        df_full["PredictedLapTime (s)"] = preds
        df_full.sort_values(
            by=["PredictedLapTime (s)", "QualifyingTime (s)"],
            ascending=True, inplace=True
        )
        df_full.reset_index(drop=True, inplace=True)
        df_full.index = df_full.index + 1
        gen_time = time.time() - start_time
    
    # --- RESULTS ---
    st.success(f"Predictions generated in {gen_time:.2f} seconds!")
    
    if model is not None:
        st.caption(f"XGBoost MAE: **{mae:.3f} seconds**")
    
    # --- DATA SOURCES PANEL ---
    with st.expander("📋 Data Sources Used in This Prediction", expanded=True):
        src_col1, src_col2 = st.columns(2)
        
        with src_col1:
            st.markdown("**🏎️ Lap Time Input (Grid Order)**")
            if data_source == "qualifying":
                st.success("✅ Real Qualifying (FastF1 live)")
            elif data_source == "qualifying_saved":
                st.info("📁 Predicted Qualifying (saved from Qualifying Predictor)")
            elif data_source == "practice":
                st.warning(f"⚠️ {p_session} Practice laps (qualifying not yet available)")
            elif data_source == "testing":
                st.info("📋 Pre-Season Testing baselines")
            
            st.markdown("**📡 Historical Race Data**")
            if not hist_data.empty:
                st.success(f"✅ 2025 {historical_race_target} GP race laps (FastF1)")
            else:
                st.warning("⚠️ No historical data — using team pace estimates only")
        
        with src_col2:
            st.markdown("**🌤️ Weather**")
            rain_pct = weather.get('pop', 0) * 100
            temp = weather.get('temp', 22)
            desc = weather.get('description', 'N/A').title()
            if weather.get('description', '') == 'Unknown - using historical average' or rain_pct == 0 and temp == 22:
                st.info(f"📋 Default (no forecast available) — assuming dry, 22°C")
            else:
                condition_icon = "🌧️" if rain_pct >= 50 else "⛅" if rain_pct >= 20 else "☀️"
                st.success(f"{condition_icon} Live forecast: {desc}, {temp}°C, {rain_pct:.0f}% rain")
            
            st.markdown("**🤖 Prediction Model**")
            if model is not None:
                st.success(f"✅ XGBoost trained on 2025 {historical_race_target} laps (MAE: {mae:.3f}s)")
            else:
                st.warning("⚠️ Fallback: team pace formula (not enough training data)")
            
            st.markdown("**🧪 Pre-Season Testing**")
            if not preseason.empty:
                st.success(f"✅ 2026 Barcelona + Bahrain testing fastest laps")
            else:
                st.warning("⚠️ No testing data loaded")
    
    st.divider()
    
    col_pod, col_table = st.columns([1, 2])
    
    with col_pod:
        st.subheader("🏆 Podium Prediction")
        medals = ["🥇", "🥈", "🥉"]
        for i in range(min(3, len(df_full))):
            p = df_full.iloc[i]
            tc = TEAM_COLORS.get(p['Team'], '#FFFFFF')
            st.markdown(f"""
            <div style="border-left: 4px solid {tc}; padding: 8px 12px; margin-bottom: 12px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                <h3 style="margin:0; color: {tc};">{medals[i]} P{i+1}: {p['DriverName']}</h3>
                <p style="margin:0; color: #999;">{p['Team']} · {p['PredictedLapTime (s)']:.3f}s</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_table:
        st.subheader("Full Grid Prediction")
        display_df = df_full[["DriverCode", "DriverName", "Team", "QualifyingTime (s)", "PredictedLapTime (s)"]].copy()
        display_df.insert(0, "Pos", range(1, len(display_df) + 1))
        display_df["Gap"] = display_df["PredictedLapTime (s)"] - display_df["PredictedLapTime (s)"].iloc[0]
        
        def color_row(row):
            color = TEAM_COLORS.get(row['Team'], "#FFFFFF")
            return [f'border-left: 4px solid {color}'] * len(row)
        
        st.dataframe(
            display_df.style.apply(color_row, axis=1).format({
                "QualifyingTime (s)": "{:.3f}",
                "PredictedLapTime (s)": "{:.3f}",
                "Gap": "+{:.3f}"
            }),
            use_container_width=True, height=500
        )
    
    st.divider()
    
    # Analytics
    st.subheader("📊 Model Analytics")
    c1, c2 = st.columns(2)
    
    with c1:
        fig = px.bar(
            df_full, y="DriverCode", x="PredictedLapTime (s)",
            color="Team", color_discrete_map=TEAM_COLORS,
            orientation='h', title="Predicted Lap Pace by Driver"
        )
        fig.update_layout(yaxis={'categoryorder': 'total descending'}, template="plotly_dark")
        fig.update_xaxes(range=[df_full["PredictedLapTime (s)"].min() - 0.5, df_full["PredictedLapTime (s)"].max() + 0.5])
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        if model is not None:
            importances = get_feature_importances(model, X_features.columns.tolist())
            fig2 = px.bar(importances, x="Importance", y="Feature", orientation='h',
                          title="XGBoost Feature Importance", color_discrete_sequence=["#E8002D"])
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
