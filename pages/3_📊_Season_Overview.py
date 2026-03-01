import streamlit as st
import pandas as pd
import plotly.express as px

from src.constants import load_calendar, TEAM_COLORS
from src.data_loader import get_2026_preseason_data, get_driver_grid_2026

st.set_page_config(page_title="Season Overview", page_icon="📊", layout="wide")

st.title("📊 F1 Prediction 3000: Season Overview")

# 1. 2026 Calendar
st.header("🗓️ 2026 Race Calendar (24 Rounds)")
calendar = load_calendar()
cal_df = pd.DataFrame(calendar)
# Clean up table for display
cal_df = cal_df.rename(columns={"round": "Round", "name": "Grand Prix", "date": "Date"})

st.dataframe(cal_df[["Round", "Date", "Grand Prix"]], use_container_width=True, hide_index=True)

st.divider()

# 2. Pre-Season Testing
st.header("⏱️ Pre-Season Testing Pace")
st.markdown("Aggregated fastest lap times from Barcelona and Bahrain testing.")

preseason_data = get_2026_preseason_data()
grid_2026 = get_driver_grid_2026()

df = grid_2026.merge(preseason_data, on="DriverCode", how="inner")
df = df.sort_values("TestingPace (s)", ascending=True)

# Plot testing pace colored by team
fig = px.bar(
    df, 
    y="DriverName", 
    x="TestingPace (s)", 
    color="Team",
    color_discrete_map=TEAM_COLORS,
    orientation='h',
    title="2026 Pre-Season Testing Best Lap Pace"
)
fig.update_layout(yaxis={'categoryorder':'total descending'}, template="plotly_dark")

# Zoom in on the variance
min_pace = df["TestingPace (s)"].min() - 0.5
max_pace = df["TestingPace (s)"].max() + 0.5
fig.update_xaxes(range=[min_pace, max_pace])

st.plotly_chart(fig, use_container_width=True)

# 3. Data Tables
st.subheader("Raw Testing Data")
st.dataframe(df, use_container_width=True, hide_index=True)

