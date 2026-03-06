import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from src.constants import CLEAN_AIR_PACE, DRIVER_TO_TEAM, WET_PERFORMANCE

# Circuit type classification (matching training_data_builder.py)
CIRCUIT_TYPES = {
    "Australia": 1, "Bahrain": 0, "Saudi Arabia": 2, "Japan": 0, "China": 0,
    "Miami": 2, "Monaco": 2, "Canada": 1, "Spain": 0, "Austria": 0,
    "Great Britain": 0, "Hungary": 0, "Belgium": 0, "Netherlands": 0,
    "Italy": 0, "Azerbaijan": 2, "Singapore": 2, "United States": 0,
    "Mexico": 0, "Brazil": 0, "Las Vegas": 2, "Qatar": 0, "Abu Dhabi": 0,
    "Madrid": 0, "Imola": 0,
}

# DNF tendency per driver (historical 2022-2025 rate, 0-1)
DNF_RATES = {
    "VER": 0.06, "HAM": 0.05, "LEC": 0.10, "NOR": 0.07, "ALO": 0.08,
    "PIA": 0.06, "RUS": 0.07, "SAI": 0.08, "STR": 0.10, "GAS": 0.09,
    "OCO": 0.11, "BEA": 0.08, "HAD": 0.08, "ANT": 0.08, "COL": 0.10,
    "ALB": 0.09, "LAW": 0.08, "LIN": 0.10, "HUL": 0.10, "BOR": 0.10,
    "BOT": 0.08, "PER": 0.09,
}

# Races per driver (experience at each circuit, 0 for rookies)
EXPERIENCE = {
    "VER": 10, "HAM": 18, "LEC": 7, "NOR": 6, "ALO": 18,
    "PIA": 3, "RUS": 4, "SAI": 6, "STR": 8, "GAS": 7,
    "OCO": 7, "BEA": 1, "HAD": 0, "ANT": 0, "COL": 1,
    "ALB": 4, "LAW": 1, "LIN": 0, "HUL": 12, "BOR": 0,
    "BOT": 12, "PER": 12,
}


def calculate_team_performance_score():
    """
    Normalized team performance score based on estimated 2026 constructor strength.
    Higher = stronger team = predicted to be faster.
    """
    team_points = {
        "McLaren": 800, "Ferrari": 650, "Red Bull": 550, "Mercedes": 500,
        "Aston Martin": 120, "Williams": 100, "Alpine": 80, "Racing Bulls": 70,
        "Haas": 60, "Audi": 40, "Cadillac": 30
    }
    max_points = max(team_points.values())
    return {team: points / max_points for team, points in team_points.items()}


def estimate_qualifying_time(driver_code, circuit_baseline=None):
    """
    Estimate a qualifying time for a driver when real qualifying data isn't available.
    """
    base_pace = CLEAN_AIR_PACE.get(driver_code, 95.0)
    
    if circuit_baseline is not None:
        reference_pace = 93.5
        delta = base_pace - reference_pace
        estimated_quali = circuit_baseline + delta
    else:
        estimated_quali = base_pace - 2.5
    
    return estimated_quali


def engineer_features(historical_data, preseason_data, grid_2026, weather_conditions,
                      qualifying_data=None, circuit_baseline=None, circuit_name=None):
    """
    Build the feature matrix for race prediction.
    
    Enhanced feature set (10 features):
    [0] QualifyingTime — fastest qualifying lap (real or estimated)
    [1] GridPosition — starting grid position
    [2] RainProbability
    [3] Temperature
    [4] WindSpeed — max wind speed (km/h)
    [5] TeamPerformanceScore
    [6] CleanAirRacePace (s)
    [7] CircuitType — 0=permanent, 1=semi-street, 2=street
    [8] DriverExperience — races at this circuit type
    [9] DNFRate — historical DNF probability
    """
    df = grid_2026.copy()
    
    # Merge preseason testing pace (for display, not used as ML feature)
    df = df.merge(preseason_data, on="DriverCode", how="left")
    
    # Merge historical sector times if available
    if not historical_data.empty and "TotalSectorTime (s)" in historical_data.columns:
        df = df.merge(
            historical_data[["Driver", "TotalSectorTime (s)"]],
            left_on="DriverCode", right_on="Driver", how="left"
        )
        df.drop(columns=["Driver"], inplace=True, errors="ignore")
    
    # Map static performance indicators
    df["CleanAirRacePace (s)"] = df["DriverCode"].map(CLEAN_AIR_PACE)
    
    team_performance = calculate_team_performance_score()
    df["TeamPerformanceScore"] = df["Team"].map(team_performance)
    
    # --- Weather ---
    rain_prob = weather_conditions.get("pop", 0.0)
    temperature = weather_conditions.get("temp", 20.0)
    wind_speed = weather_conditions.get("wind_speed", 10.0)
    df["RainProbability"] = rain_prob
    df["Temperature"] = temperature
    df["WindSpeed"] = wind_speed
    
    # --- Circuit Type ---
    ct = 0  # default permanent
    if circuit_name:
        ct = CIRCUIT_TYPES.get(circuit_name, 0)
    df["CircuitType"] = ct
    
    # --- Driver Experience & DNF Rate ---
    df["DriverExperience"] = df["DriverCode"].map(EXPERIENCE).fillna(0)
    df["DNFRate"] = df["DriverCode"].map(DNF_RATES).fillna(0.10)
    
    # --- QUALIFYING TIME (the #1 most important feature) ---
    if qualifying_data is not None and not qualifying_data.empty:
        df = df.merge(qualifying_data[["DriverCode", "QualifyingTime (s)"]],
                      on="DriverCode", how="left")
        mask = df["QualifyingTime (s)"].isna()
        df.loc[mask, "QualifyingTime (s)"] = df.loc[mask, "DriverCode"].apply(
            lambda code: estimate_qualifying_time(code, circuit_baseline)
        )
    else:
        df["QualifyingTime (s)"] = df["DriverCode"].apply(
            lambda code: estimate_qualifying_time(code, circuit_baseline)
        )
    
    # --- GRID POSITION ---
    # If qualifying data exists, derive grid from qualifying order
    df["GridPosition"] = df["QualifyingTime (s)"].rank(method="min").astype(int)
    
    # Wet adjustment on qualifying time
    df["WetFactor"] = df["DriverCode"].map(WET_PERFORMANCE)
    if rain_prob >= 0.75:
        df["QualifyingTime (s)"] = df["QualifyingTime (s)"] * df["WetFactor"]
    
    # --- FEATURE MATRIX (enhanced 10-feature set) ---
    feature_cols = [
        "QualifyingTime (s)",
        "GridPosition",
        "RainProbability",
        "Temperature",
        "WindSpeed",
        "TeamPerformanceScore",
        "CleanAirRacePace (s)",
        "CircuitType",
        "DriverExperience",
        "DNFRate",
    ]
    
    imputer = SimpleImputer(strategy="median")
    X = df[feature_cols].copy()
    X_imputed = imputer.fit_transform(X)
    
    df_features = pd.DataFrame(X_imputed, columns=feature_cols)
    return df, df_features
