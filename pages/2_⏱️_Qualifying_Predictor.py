import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

from src.constants import load_calendar, TEAM_COLORS, CLEAN_AIR_PACE
from src.data_loader import (
    get_historical_qualifying_data, get_2026_preseason_data,
    get_driver_grid_2026, save_session_results, get_all_practice_data
)
from src.weather import get_weather_forecast

# ── Cached data loaders ───────────────────────────────────────────────────────
# All heavy FastF1 calls are cached so slider/checkbox interactions
# don't re-trigger session loads and cause segfaults.
@st.cache_data(show_spinner=False)
def cached_all_practice(year, race):
    return get_all_practice_data(year, race)

@st.cache_data(show_spinner=False)
def cached_hist_quali(year, race):
    return get_historical_qualifying_data(year, race)

@st.cache_data(show_spinner=False)
def cached_preseason():
    return get_2026_preseason_data()

@st.cache_data(show_spinner=False)
def cached_weather(lat, lon, date):
    return get_weather_forecast(lat, lon, date)
# ─────────────────────────────────────────────────────────────────────────────

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
    weather = cached_weather(race_info['lat'], race_info['lon'], race_info['date'])
    m1, m2, m3 = st.columns(3)
    m1.metric("Temperature", f"{weather['temp']}°C")
    m2.metric("Rain Probability", f"{weather['pop']*100:.0f}%")
    m3.metric("Conditions", weather.get('description', 'N/A').title())

st.divider()

# --- DATA AVAILABILITY CHECK ---
with st.spinner("Checking FastF1 for live 2026 session data (FP1 + FP2 + FP3)..."):
    practice_df, sessions_loaded = cached_all_practice(2026, historical_race_target)
    hist_quali_check = cached_hist_quali(2025, historical_race_target)
    preseason_check = cached_preseason()

has_practice = not practice_df.empty
sessions_label = " + ".join(sessions_loaded) if sessions_loaded else "None"

# --- DATA STATUS CARD (always visible before prediction) ---
st.subheader("📡 Data Inputs")
dc1, dc2, dc3, dc4 = st.columns(4)

with dc1:
    st.markdown("**🏎️ Live Practice Sessions**")
    if has_practice:
        laps_count = len(practice_df)
        st.success(f"✅ **{sessions_label}** ({laps_count} drivers, best lap across all sessions)")
    else:
        st.error("❌ No practice data yet")
        st.caption("Available after FP1/FP2/FP3")

with dc2:
    st.markdown("**📡 2025 Historical Qualifying**")
    if not hist_quali_check.empty:
        st.success(f"✅ {historical_race_target} 2025 Q ({len(hist_quali_check)} drivers)")
    else:
        st.warning(f"⚠️ No 2025 {historical_race_target} data")
        st.caption("Practice-only estimation used")

with dc3:
    st.markdown("**🧪 Pre-Season Testing**")
    if not preseason_check.empty:
        st.success(f"✅ 2026 BCN + BHR ({len(preseason_check)} drivers)")
    else:
        st.warning("⚠️ No testing data loaded")

with dc4:
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

# --- BLEND SETTINGS ---
with st.expander("⚙️ Blend Settings — Customise Data Source Weights", expanded=False):
    st.caption("Choose which data sources to include and how much to trust each one. Weights are automatically normalised to 100%.")
    
    bs_col1, bs_col2 = st.columns(2)
    
    with bs_col1:
        use_fp = st.checkbox("🏎️ Practice Sessions (FP1/FP2/FP3)", value=has_practice, disabled=not has_practice)
        w_fp = st.slider("FP Weight", 0, 100, 40, step=5, disabled=not use_fp, key="w_fp")
        
        use_hist = st.checkbox("📡 2025 Historical Qualifying", value=not hist_quali_check.empty, disabled=hist_quali_check.empty)
        w_hist = st.slider("Historical Qualifying Weight", 0, 100, 30, step=5, disabled=not use_hist, key="w_hist")
    
    with bs_col2:
        use_testing = st.checkbox("🧪 Pre-Season Testing", value=not preseason_check.empty, disabled=preseason_check.empty)
        w_testing = st.slider("Testing Weight", 0, 100, 15, step=5, disabled=not use_testing, key="w_testing")
        
        use_team_est = st.checkbox("📐 Team-Circuit Estimate", value=True)
        w_team_est = st.slider("Team-Circuit Estimate Weight", 0, 100, 15, step=5, disabled=not use_team_est, key="w_team_est")
    
    # Calculate total and show normalisation info
    raw_total = (w_fp if use_fp else 0) + (w_hist if use_hist else 0) + \
                (w_testing if use_testing else 0) + (w_team_est if use_team_est else 0)
    
    if raw_total == 0:
        st.error("⚠️ At least one source must be selected!")
        run_btn = False
    else:
        # Normalised weights
        nw_fp      = (w_fp      / raw_total) if use_fp      else 0.0
        nw_hist    = (w_hist    / raw_total) if use_hist    else 0.0
        nw_testing = (w_testing / raw_total) if use_testing else 0.0
        nw_team    = (w_team_est / raw_total) if use_team_est else 0.0
        
        pct_parts = []
        if use_fp:       pct_parts.append(f"FP: **{nw_fp*100:.0f}%**")
        if use_hist:     pct_parts.append(f"Hist Q: **{nw_hist*100:.0f}%**")
        if use_testing:  pct_parts.append(f"Testing: **{nw_testing*100:.0f}%**")
        if use_team_est: pct_parts.append(f"Team Est: **{nw_team*100:.0f}%**")
        st.info("📊 Normalised blend: " + " · ".join(pct_parts))

if has_practice and raw_total > 0:
    run_btn = st.button("🚀 Predict Qualifying Order", use_container_width=True)
elif not has_practice:
    st.error(f"❌ **Cannot predict qualifying** — No practice session data available yet for the 2026 {historical_race_target} Grand Prix.")
    st.info("FastF1 will automatically fetch the data when FP1/FP2/FP3 sessions happen.")
    run_btn = False

# --- QUALIFYING PREDICTION ---
if run_btn and has_practice:
    with st.spinner("Building qualifying prediction from all available data sources..."):
        start_time = time.time()
        
        grid = get_driver_grid_2026()
        preseason = get_2026_preseason_data()
        
        # Team performance scores
        team_points = {
            "McLaren": 800, "Ferrari": 650, "Red Bull": 550, "Mercedes": 500,
            "Aston Martin": 120, "Williams": 100, "Alpine": 80, "Racing Bulls": 70,
            "Haas": 60, "Audi": 40, "Cadillac": 30
        }
        max_pts = max(team_points.values())
        team_scores = {t: p / max_pts for t, p in team_points.items()}
        
        df = grid.copy()
        df["TeamScore"] = df["Team"].map(team_scores)
        
        # ── Source 1: Combined FP sessions (best lap across FP1+FP2+FP3) ──
        df = df.merge(practice_df[["DriverCode", "PracticeTime (s)"]], on="DriverCode", how="left")
        # Estimate qualifying pace: practice best − 1.5s − team bonus
        df["FP_EstQuali (s)"] = df["PracticeTime (s)"] - 1.5 - (df["TeamScore"] * 0.5)
        
        # ── Source 2: 2025 Historical Qualifying ──
        hist_quali = get_historical_qualifying_data(2025, historical_race_target)
        if not hist_quali.empty:
            hist_quali = hist_quali.rename(columns={"Driver": "DriverCode", "QualifyingTime (s)": "HistQualiTime (s)"})
            df = df.merge(hist_quali[["DriverCode", "HistQualiTime (s)"]], on="DriverCode", how="left")
            hist_available = True
        else:
            df["HistQualiTime (s)"] = np.nan
            hist_available = False
        
        # ── Source 3: Pre-Season Testing ──
        if not preseason.empty:
            df = df.merge(preseason[["DriverCode", "TestingPace (s)"]], on="DriverCode", how="left")
            testing_available = True
        else:
            df["TestingPace (s)"] = np.nan
            testing_available = False
        
        # ── Source 4: Team-Circuit Estimate (always available as sanity anchor) ──
        # Derives where each driver *should* be based on their team's performance
        # relative to the circuit's fastest historical time.
        # This prevents anomalously fast practice laps from dominating.
        if hist_available:
            p1_circuit_time = df["HistQualiTime (s)"].min()  # fastest 2025 quali time at this circuit
            # Typical qualifying spread: ~3s from best team (1.0) to worst (0.0)
            df["TeamCircuitEst (s)"] = p1_circuit_time + (1 - df["TeamScore"]) * 3.0
            team_est_available = True
        elif not practice_df.empty:
            # Fallback: derive from practice session fastest
            p1_practice = practice_df["PracticeTime (s)"].min()
            df["TeamCircuitEst (s)"] = (p1_practice - 1.5) + (1 - df["TeamScore"]) * 3.0
            team_est_available = True
        else:
            df["TeamCircuitEst (s)"] = np.nan
            team_est_available = False

        # ── Blended Estimation using user-defined weights ──
        def blend_row(row):
            fp_val   = row.get("FP_EstQuali (s)", np.nan)
            hist_val = row.get("HistQualiTime (s)", np.nan)
            test_val = row.get("TestingPace (s)", np.nan)
            team_val = row.get("TeamCircuitEst (s)", np.nan)

            sources, weights = [], []
            if use_fp      and pd.notna(fp_val):   sources.append(fp_val);   weights.append(nw_fp)
            if use_hist    and pd.notna(hist_val): sources.append(hist_val); weights.append(nw_hist)
            if use_testing and pd.notna(test_val): sources.append(test_val); weights.append(nw_testing)
            if use_team_est and pd.notna(team_val): sources.append(team_val); weights.append(nw_team)

            if not sources:
                return np.nan
            total_w = sum(weights)
            return sum(s * w for s, w in zip(sources, weights)) / total_w

        df["EstimatedQuali (s)"] = df.apply(blend_row, axis=1)
        
        # ── Team floor constraint ──
        # A driver's estimate can't be more than 0.3s faster than their team's
        # circuit estimate. This prevents a single lucky qualifying session from
        # overriding the car's realistic performance level (e.g. Stroll P2).
        if team_est_available:
            df["EstimatedQuali (s)"] = df.apply(
                lambda row: max(
                    row["EstimatedQuali (s)"],
                    row["TeamCircuitEst (s)"] - 0.3
                ) if pd.notna(row["EstimatedQuali (s)"]) and pd.notna(row["TeamCircuitEst (s)"])
                else row["EstimatedQuali (s)"],
                axis=1
            )
        
        # ── Weather adjustment (wet race) ──
        rain_prob = weather.get("pop", 0.0)
        if rain_prob >= 0.75:
            from src.constants import WET_PERFORMANCE
            df["WetFactor"] = df["DriverCode"].map(WET_PERFORMANCE)
            df["EstimatedQuali (s)"] *= df["WetFactor"]
        
        # ── Sort: drivers with data first, rest at the back ──
        df_with_times = df[df["EstimatedQuali (s)"].notna()].copy()
        df_without_times = df[df["EstimatedQuali (s)"].isna()].copy()
        
        df_with_times.sort_values("EstimatedQuali (s)", ascending=True, inplace=True)
        
        if not df_without_times.empty:
            df_without_times["EstimatedQuali (s)"] = df_with_times["EstimatedQuali (s)"].max() + 1.0
        
        df = pd.concat([df_with_times, df_without_times], ignore_index=True)
        df.index = df.index + 1
        
        gen_time = time.time() - start_time
    
    st.success(f"Qualifying predicted in {gen_time:.2f} seconds!")
    
    # --- DATA SOURCES PANEL ---
    with st.expander("📋 Data Sources Used in This Prediction", expanded=True):
        src_col1, src_col2 = st.columns(2)
        
        with src_col1:
            st.markdown("**🏎️ Practice Sessions (40% weight)**")
            if sessions_loaded:
                st.success(f"✅ {sessions_label} — best lap per driver across all sessions")
            else:
                st.error("❌ No practice data")
            
            st.markdown("**📡 2025 Historical Qualifying (30% weight)**")
            if hist_available:
                st.success(f"✅ 2025 {historical_race_target} GP qualifying (FastF1)")
            else:
                st.warning(f"⚠️ Not available — weight redistributed to other sources")
        
        with src_col2:
            st.markdown("**🧪 Pre-Season Testing (15% weight)**")
            if testing_available:
                st.success("✅ 2026 Barcelona + Bahrain fastest laps")
            else:
                st.warning("⚠️ Not available — weight redistributed")
            
            st.markdown("**📐 Team-Circuit Estimate (15% weight)**")
            if team_est_available:
                st.success("✅ Derived from circuit baseline + team performance gap (prevents unrealistic outliers)")
            else:
                st.warning("⚠️ Not available")
            
            st.markdown("**🌤️ Weather**")
            rain_pct_d = weather.get('pop', 0) * 100
            temp_d = weather.get('temp', 22)
            desc_d = weather.get('description', 'Unknown - using historical average')
            if desc_d == 'Unknown - using historical average':
                st.info("📋 Default (no forecast) — dry conditions assumed")
            else:
                condition_icon = "🌧️" if rain_pct_d >= 50 else "⛅" if rain_pct_d >= 20 else "☀️"
                wet_note = " — **wet factors applied**" if rain_pct_d >= 75 else ""
                st.success(f"{condition_icon} {desc_d.title()}, {temp_d}°C, {rain_pct_d:.0f}% rain{wet_note}")
    
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
        orientation='h', title=f"Predicted Qualifying Pace (from {sessions_label} + History)"
    )
    fig.update_layout(yaxis={'categoryorder': 'total descending'}, template="plotly_dark")
    fig.update_xaxes(range=[df["EstimatedQuali (s)"].min() - 0.5, df["EstimatedQuali (s)"].max() + 0.5])
    st.plotly_chart(fig, use_container_width=True)
