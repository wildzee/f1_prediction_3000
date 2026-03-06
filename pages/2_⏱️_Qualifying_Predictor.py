import streamlit as st
import pandas as pd
import plotly.express as px
import time

from src.constants import load_calendar, TEAM_COLORS, CLEAN_AIR_PACE
from src.data_loader import (
    get_historical_qualifying_data, get_2026_preseason_data,
    get_driver_grid_2026, save_session_results, get_live_practice_data
)
from src.weather import get_weather_forecast

st.set_page_config(page_title="Qualifying Predictor", page_icon="⏱️", layout="wide")

st.markdown("""
<style>
    h1, h2, h3 { color: #E8002D; }
    .stButton>button { background-color: #E8002D; color: white; border-radius: 4px; font-weight: bold; }
    .stButton>button:hover { background-color: #FF1A46; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("⏱️ F1 Prediction 3000: Qualifying Predictor")
st.markdown("Predict qualifying order automatically using **live practice session data** and **historical performance**.")

# Load Calendar
calendar = load_calendar()
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

with col2:
    st.subheader("🌤️ Weather Forecast")
    weather = get_weather_forecast(race_info['lat'], race_info['lon'], race_info['date'])
    m1, m2, m3 = st.columns(3)
    m1.metric("Temperature", f"{weather['temp']}°C")
    m2.metric("Rain Probability", f"{weather['pop']*100:.0f}%")
    m3.metric("Conditions", weather.get('description', 'N/A').title())

st.divider()

# --- DATA AVAILABILITY CHECK ---
with st.spinner("Checking FastF1 for live 2026 session data..."):
    practice_df, session_used = get_live_practice_data(2026, historical_race_target)
    hist_quali_check = get_historical_qualifying_data(2025, historical_race_target)

has_practice = not practice_df.empty

# --- DATA STATUS CARD (always visible before prediction) ---
st.subheader("📡 Data Inputs")
dc1, dc2, dc3 = st.columns(3)

with dc1:
    st.markdown("**🏎️ Live Practice Session**")
    if has_practice:
        laps_count = len(practice_df)
        st.success(f"✅ **{session_used}** ({laps_count} drivers loaded)")
    else:
        st.error("❌ No practice data yet")
        st.caption("Available after FP1/FP2/FP3")

with dc2:
    st.markdown("**📡 2025 Historical Qualifying**")
    if not hist_quali_check.empty:
        st.success(f"✅ {historical_race_target} 2025 Q data ({len(hist_quali_check)} drivers)")
    else:
        st.warning(f"⚠️ No 2025 {historical_race_target} data")
        st.caption("Practice-only estimation will be used")

with dc3:
    st.markdown("**🌤️ Weather for Race Day**")
    rain_pct = weather.get('pop', 0) * 100
    temp = weather.get('temp', 22)
    desc = weather.get('description', 'Unknown - using historical average')
    if desc == 'Unknown - using historical average':
        st.info("📋 No forecast (>5 days out)")
        st.caption("Dry conditions assumed")
    else:
        icon = "🌧️" if rain_pct >= 50 else "⛅" if rain_pct >= 20 else "☀️"
        st.success(f"{icon} {desc.title()}, {temp}°C, {rain_pct:.0f}% rain")
        if rain_pct >= 75:
            st.caption("⚠️ Wet performance factors will be applied")

st.divider()

if has_practice:
    run_btn = st.button("🚀 Predict Qualifying Order", use_container_width=True)
else:
    st.error(f"❌ **Cannot predict qualifying** — No practice session data available yet for the 2026 {historical_race_target} Grand Prix.")
    st.info("FastF1 will automatically fetch the data when FP1/FP2/FP3 sessions happen.")
    run_btn = False

# --- QUALIFYING PREDICTION ---
if run_btn and has_practice:
    with st.spinner("Building qualifying prediction from practice and historical data..."):
        start_time = time.time()
        
        grid = get_driver_grid_2026()
        
        # Merge practice data
        df = grid.merge(practice_df[["DriverCode", "PracticeTime (s)"]], on="DriverCode", how="left")
        
        # Load historical qualifying data (2025)
        hist_quali = get_historical_qualifying_data(2025, historical_race_target)
        if not hist_quali.empty:
            hist_quali = hist_quali.rename(columns={"Driver": "DriverCode", "QualifyingTime (s)": "HistQualiTime (s)"})
            df = df.merge(hist_quali[["DriverCode", "HistQualiTime (s)"]], on="DriverCode", how="left")
            circuit_median = hist_quali["HistQualiTime (s)"].median()
            hist_available = True
        else:
            hist_available = False
        
        # Team performance scores
        team_points = {
            "McLaren": 800, "Ferrari": 650, "Red Bull": 550, "Mercedes": 500,
            "Aston Martin": 120, "Williams": 100, "Alpine": 80, "Racing Bulls": 70,
            "Haas": 60, "Audi": 40, "Cadillac": 30
        }
        max_pts = max(team_points.values())
        team_scores = {t: p / max_pts for t, p in team_points.items()}
        df["TeamScore"] = df["Team"].map(team_scores)
        
        # --- FEATURE ENGINEERING FOR QUALIFYING ---
        # 1. Base estimation: typically 1-2s faster than practice
        df["EstimatedQuali (s)"] = df["PracticeTime (s)"] - 1.5 - (df["TeamScore"] * 0.5)
        
        # 2. Incorporate historical data if available
        if hist_available:
            mask = df["HistQualiTime (s)"].notna() & df["PracticeTime (s)"].notna()
            # If historical is very different from practice (weather, track changes), trust practice more
            # Blend: 80% Live Practice Estimation, 20% Historical Performance
            df.loc[mask, "EstimatedQuali (s)"] = (df.loc[mask, "EstimatedQuali (s)"] * 0.8) + (df.loc[mask, "HistQualiTime (s)"] * 0.2)
        
        # Weather adjustment
        rain_prob = weather.get("pop", 0.0)
        if rain_prob >= 0.75:
            from src.constants import WET_PERFORMANCE
            df["WetFactor"] = df["DriverCode"].map(WET_PERFORMANCE)
            df["EstimatedQuali (s)"] *= df["WetFactor"]
        
        # Handle drivers with missing practice data
        df_with_times = df[df["PracticeTime (s)"].notna()].copy()
        df_without_times = df[df["PracticeTime (s)"].isna()].copy()
        
        df_with_times.sort_values("EstimatedQuali (s)", ascending=True, inplace=True)
        
        if not df_without_times.empty:
            # Drivers without practice data get put at the back
            df_without_times["EstimatedQuali (s)"] = df_with_times["EstimatedQuali (s)"].max() + 1.0
        
        df = pd.concat([df_with_times, df_without_times], ignore_index=True)
        df.index = df.index + 1
        
        gen_time = time.time() - start_time
    
    st.success(f"Qualifying predicted in {gen_time:.2f} seconds! (Blended {session_used} + Historical Data)")
    
    # --- DATA SOURCES PANEL ---
    with st.expander("📋 Data Sources Used in This Prediction", expanded=True):
        src_col1, src_col2 = st.columns(2)
        
        with src_col1:
            st.markdown("**🏎️ Live Practice Data**")
            st.success(f"✅ {session_used} — {historical_race_target} GP 2026 (FastF1, filtered to flying laps)")
            
            st.markdown("**📡 Historical Qualifying Data**")
            if hist_available:
                st.success(f"✅ 2025 {historical_race_target} GP qualifying (FastF1) — blended at 20%")
            else:
                st.warning(f"⚠️ No 2025 {historical_race_target} qualifying data found — using practice only")
        
        with src_col2:
            st.markdown("**🌤️ Weather**")
            rain_pct = weather.get('pop', 0) * 100
            temp = weather.get('temp', 22)
            desc = weather.get('description', 'N/A').title()
            if weather.get('description', '') == 'Unknown - using historical average':
                st.info("📋 Default (no forecast available) — dry conditions assumed")
            else:
                condition_icon = "🌧️" if rain_pct >= 50 else "⛅" if rain_pct >= 20 else "☀️"
                wet_note = " — **wet performance factors applied**" if rain_pct >= 75 else ""
                st.success(f"{condition_icon} Live forecast: {desc}, {temp}°C, {rain_pct:.0f}% rain{wet_note}")
            
            st.markdown("**⚙️ Estimation Method**")
            st.info("Practice time − 1.5s − (team performance bonus) → blended with historical qualifying at 80/20")
    
    st.divider()
    
    # Save button
    col_save, _ = st.columns([1, 2])
    with col_save:
        if st.button("💾 Save as Qualifying Data → feeds Race Predictor"):
            quali_times = {row["DriverCode"]: row["EstimatedQuali (s)"] 
                          for _, row in df.iterrows() if pd.notna(row["EstimatedQuali (s)"])}
            save_session_results(race_info['round'], "Q", quali_times)
            st.success("✅ Saved! The Race Predictor will now use this qualifying data.")
    
    st.divider()
    
    # --- RESULTS ---
    col_pole, col_grid = st.columns([1, 2])
    
    with col_pole:
        st.subheader("🏆 Pole Position Prediction")
        medals = ["🥇", "🥈", "🥉"]
        for i in range(min(3, len(df))):
            p = df.iloc[i]
            tc = TEAM_COLORS.get(p['Team'], '#FFFFFF')
            st.markdown(f"""
            <div style="border-left: 4px solid {tc}; padding: 8px 12px; margin-bottom: 12px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                <h3 style="margin:0; color: {tc};">{medals[i]} P{i+1}: {p['DriverName']}</h3>
                <p style="margin:0; color: #999;">{p['Team']} · {p['EstimatedQuali (s)']:.3f}s</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_grid:
        st.subheader("Full Qualifying Order")
        display_df = df[["DriverCode", "DriverName", "Team", "PracticeTime (s)", "EstimatedQuali (s)"]].copy()
        if hist_available:
            display_df.insert(4, "HistQualiTime (s)", df["HistQualiTime (s)"])
        
        display_df.insert(0, "Pos", range(1, len(display_df) + 1))
        display_df["Gap"] = display_df["EstimatedQuali (s)"] - display_df["EstimatedQuali (s)"].iloc[0]
        
        def color_row(row):
            color = TEAM_COLORS.get(row['Team'], "#FFFFFF")
            return [f'border-left: 4px solid {color}'] * len(row)
        
        st.dataframe(
            display_df.style.apply(color_row, axis=1).format({
                "PracticeTime (s)": "{:.3f}",
                "HistQualiTime (s)": "{:.3f}",
                "EstimatedQuali (s)": "{:.3f}",
                "Gap": "+{:.3f}"
            }),
            use_container_width=True, height=500
        )
    
    st.divider()
    
    # Chart
    fig = px.bar(
        df, y="DriverCode", x="EstimatedQuali (s)",
        color="Team", color_discrete_map=TEAM_COLORS,
        orientation='h', title=f"Predicted Qualifying Pace (from {session_used} + History)"
    )
    fig.update_layout(yaxis={'categoryorder': 'total descending'}, template="plotly_dark")
    fig.update_xaxes(range=[df["EstimatedQuali (s)"].min() - 0.5, df["EstimatedQuali (s)"].max() + 0.5])
    st.plotly_chart(fig, use_container_width=True)
