import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from src.constants import CLEAN_AIR_PACE, DRIVER_TO_TEAM, WET_PERFORMANCE


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
    Uses CleanAirRacePace as a proxy scaled to the circuit.
    
    For 2026 pre-season predictions without any qualifying data, we use
    the driver's clean air race pace as a rough qualifying estimate.
    Qualifying is typically ~2-3s faster than race pace.
    """
    base_pace = CLEAN_AIR_PACE.get(driver_code, 95.0)
    
    if circuit_baseline is not None:
        # Scale: if the circuit baseline is e.g. 80s (Australia), 
        # scale the relative differences from CleanAirPace
        reference_pace = 93.5  # average of CleanAirPace values
        delta = base_pace - reference_pace  # positive = slower than average
        estimated_quali = circuit_baseline + delta
    else:
        # Without circuit info, use pace directly (quali is ~2-3s faster)
        estimated_quali = base_pace - 2.5
    
    return estimated_quali


def engineer_features(historical_data, preseason_data, grid_2026, weather_conditions,
                      qualifying_data=None, circuit_baseline=None):
    """
    Build the feature matrix matching the original prediction scripts.
    
    Feature order (matching monotone_constraints):
    [0] QualifyingTime — fastest qualifying lap (real or estimated)
    [1] RainProbability
    [2] Temperature
    [3] TeamPerformanceScore
    [4] CleanAirRacePace (s)
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
    
    # Weather
    rain_prob = weather_conditions.get("pop", 0.0)
    temperature = weather_conditions.get("temp", 20.0)
    df["RainProbability"] = rain_prob
    df["Temperature"] = temperature
    
    # --- QUALIFYING TIME (the #1 most important feature) ---
    if qualifying_data is not None and not qualifying_data.empty:
        # Real qualifying data available
        df = df.merge(qualifying_data[["DriverCode", "QualifyingTime (s)"]],
                      on="DriverCode", how="left")
        # Fill any missing drivers with estimates
        mask = df["QualifyingTime (s)"].isna()
        df.loc[mask, "QualifyingTime (s)"] = df.loc[mask, "DriverCode"].apply(
            lambda code: estimate_qualifying_time(code, circuit_baseline)
        )
    else:
        # No qualifying data — estimate for all drivers
        df["QualifyingTime (s)"] = df["DriverCode"].apply(
            lambda code: estimate_qualifying_time(code, circuit_baseline)
        )
    
    # Wet adjustment on qualifying time
    df["WetFactor"] = df["DriverCode"].map(WET_PERFORMANCE)
    if rain_prob >= 0.75:
        df["QualifyingTime (s)"] = df["QualifyingTime (s)"] * df["WetFactor"]
    
    # --- FEATURE MATRIX (matching original script order) ---
    feature_cols = [
        "QualifyingTime (s)",
        "RainProbability",
        "Temperature",
        "TeamPerformanceScore",
        "CleanAirRacePace (s)"
    ]
    
    imputer = SimpleImputer(strategy="median")
    X = df[feature_cols].copy()
    X_imputed = imputer.fit_transform(X)
    
    df_features = pd.DataFrame(X_imputed, columns=feature_cols)
    return df, df_features
