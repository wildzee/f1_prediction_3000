import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.performance_tracker import get_accuracy_report, get_dynamic_pace
from src.constants import CLEAN_AIR_PACE, TEAM_COLORS, DRIVER_TO_TEAM

st.set_page_config(page_title="Prediction Accuracy", page_icon="📈", layout="wide")

st.markdown("""
<style>
    h1, h2, h3 { color: #E8002D; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Prediction Accuracy Tracker")
st.markdown("Track how well predictions match actual results as the season progresses.")

# --- ACCURACY REPORT ---
st.header("🎯 Prediction vs Actual Results")

report = get_accuracy_report()

if report.empty:
    st.info("📋 **No completed comparisons yet.** After a qualifying or race happens, come back here to see how accurate the predictions were!")
    st.markdown("""
    **How it works:**
    1. Make a prediction on the Qualifying or Race Predictor page
    2. After the actual session happens, the system saves the real results
    3. This page compares predicted vs actual positions and shows accuracy metrics
    """)
else:
    # Summary metrics
    avg_mae = report["MAE_Position"].mean()
    avg_top3 = report["Top3_Correct"].sum()
    total_compared = report["Drivers_Compared"].sum()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Position Error", f"±{avg_mae:.1f} places")
    m2.metric("Top 3 Predictions Correct", f"{avg_top3}")
    m3.metric("Races Compared", f"{len(report)}")
    m4.metric("Total Driver Predictions", f"{total_compared}")
    
    st.divider()
    
    # Per-race table
    st.subheader("📊 Per-Race Breakdown")
    st.dataframe(
        report.style.format({
            "MAE_Position": "{:.2f}",
            "Median_Error": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Accuracy trend chart
    if len(report) > 1:
        fig = px.line(
            report, x="Round", y="MAE_Position",
            markers=True, title="Prediction Accuracy Over Season",
            labels={"MAE_Position": "Mean Absolute Position Error", "Round": "Race Round"}
        )
        fig.update_layout(template="plotly_dark")
        fig.add_hline(y=2, line_dash="dash", line_color="green", 
                      annotation_text="Target: ±2 positions")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- DYNAMIC PACE COMPARISON ---
st.header("🏎️ Dynamic vs Static Driver Pace")
st.markdown("Compares the hardcoded pre-season pace estimates with dynamically updated values from actual race results.")

static_pace = CLEAN_AIR_PACE
dynamic_pace = get_dynamic_pace()

pace_rows = []
for driver, static_val in static_pace.items():
    dynamic_val = dynamic_pace.get(driver, static_val)
    delta = round(dynamic_val - static_val, 3)
    team = DRIVER_TO_TEAM.get(driver, "Unknown")
    pace_rows.append({
        "Driver": driver,
        "Team": team,
        "Static Pace (s)": static_val,
        "Dynamic Pace (s)": dynamic_val,
        "Delta (s)": delta,
        "Status": "🟢 Faster" if delta < -0.1 else "🔴 Slower" if delta > 0.1 else "⚪ Same"
    })

pace_df = pd.DataFrame(pace_rows).sort_values("Dynamic Pace (s)")

# Display table
st.dataframe(pace_df, use_container_width=True, hide_index=True)

# Comparison chart
if not pace_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pace_df["Driver"], x=pace_df["Static Pace (s)"],
        name="Static (Pre-Season)", orientation="h",
        marker_color="rgba(100, 100, 255, 0.6)"
    ))
    fig.add_trace(go.Bar(
        y=pace_df["Driver"], x=pace_df["Dynamic Pace (s)"],
        name="Dynamic (From Results)", orientation="h",
        marker_color="rgba(255, 100, 100, 0.6)"
    ))
    fig.update_layout(
        barmode="group", template="plotly_dark",
        title="Static vs Dynamic Driver Pace Comparison",
        xaxis_title="Lap Time (s)",
        height=600
    )
    min_pace = pace_df[["Static Pace (s)", "Dynamic Pace (s)"]].min().min() - 0.5
    max_pace = pace_df[["Static Pace (s)", "Dynamic Pace (s)"]].max().max() + 0.5
    fig.update_xaxes(range=[min_pace, max_pace])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- MODEL INFO ---
st.header("🧠 Model Architecture")
st.markdown("""
**Current model: XGBoost Gradient Boosting Regressor**

| Parameter | Value | Why |
|---|---|---|
| Learning Rate | 0.05 | Slow learning prevents overfitting |
| Estimators | 500 | More trees with slower learning |
| Max Depth | 4 | Moderate complexity |
| Subsample | 80% | Row sampling for robustness |
| Column Sample | 80% | Feature sampling prevents reliance on single feature |
| Regularisation (L2) | 1.0 | Prevents overly large weights |

**Features used (10):**
1. QualifyingTime — Fastest qualifying lap
2. GridPosition — Starting grid slot
3. RainProbability — 0-1 from Open-Meteo
4. Temperature — Degrees Celsius
5. WindSpeed — Max wind speed (km/h)
6. TeamPerformanceScore — 0-1 team strength
7. CleanAirRacePace — Driver's baseline race pace
8. CircuitType — 0=permanent, 1=semi-street, 2=street
9. DriverExperience — Races at this circuit type
10. DNFRate — Historical DNF probability
""")
