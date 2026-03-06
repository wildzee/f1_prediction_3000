import fastf1
import pandas as pd
import numpy as np
import os
import json
from src.constants import load_testing_data, DRIVER_MAPPING, DRIVER_TO_TEAM, CODE_TO_NAME

# Enable FastF1 caching
cache_dir = "f1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Session data directory for storing practice/qualifying results
SESSION_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "session_data")
os.makedirs(SESSION_DATA_DIR, exist_ok=True)


def get_historical_race_data(year, race, session_type="R"):
    """
    Load historical race/qualifying data from FastF1 for a given year and circuit.
    Matches the original prediction scripts exactly:
    - Loads laps with Driver, LapTime, Sector1/2/3Time
    - Drops NaN rows
    - Converts timedeltas to seconds
    - Aggregates sector times by driver (mean)
    - Returns per-driver mean LapTime and TotalSectorTime
    """
    try:
        session = fastf1.get_session(year, race, session_type)
        session.load()
        
        laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
        laps.dropna(inplace=True)
        
        # Convert to seconds (matching original scripts)
        for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
            laps[f"{col} (s)"] = laps[col].dt.total_seconds()
        
        # Aggregate sector times by driver (mean, matching originals)
        sector_times = laps.groupby("Driver").agg({
            "Sector1Time (s)": "mean",
            "Sector2Time (s)": "mean",
            "Sector3Time (s)": "mean"
        }).reset_index()
        
        sector_times["TotalSectorTime (s)"] = (
            sector_times["Sector1Time (s)"] +
            sector_times["Sector2Time (s)"] +
            sector_times["Sector3Time (s)"]
        )
        
        # Per-driver mean lap time (the training target, matching originals)
        mean_laps = laps.groupby("Driver")["LapTime (s)"].mean().reset_index()
        
        # Merge sector times with mean laps
        result = mean_laps.merge(sector_times, on="Driver", how="left")
        
        return result, laps
        
    except Exception as e:
        print(f"FastF1 error loading {year} {race} {session_type}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def get_historical_qualifying_data(year, race):
    """
    Load historical qualifying data from FastF1.
    Returns fastest qualifying lap per driver.
    """
    try:
        session = fastf1.get_session(year, race, "Q")
        session.load()
        
        laps = session.laps[["Driver", "LapTime"]].copy()
        laps.dropna(inplace=True)
        laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
        
        # Get fastest lap per driver (qualifying best)
        fastest = laps.groupby("Driver")["LapTime (s)"].min().reset_index()
        fastest.rename(columns={"LapTime (s)": "QualifyingTime (s)"}, inplace=True)
        
        return fastest
        
    except Exception as e:
        print(f"FastF1 error loading qualifying {year} {race}: {e}")
        return pd.DataFrame()


def load_session_results(race_round, session_type="Q"):
    """
    Load saved session results (practice or qualifying) from local JSON.
    These files are saved by the user or auto-populated after sessions happen.
    
    File format: data/session_data/round_{N}_{session_type}.json
    Contents: {"drivers": [{"code": "VER", "time": 75.481}, ...]}
    """
    filename = f"round_{race_round}_{session_type.lower()}.json"
    filepath = os.path.join(SESSION_DATA_DIR, filename)
    
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data["drivers"])
        df.rename(columns={"code": "DriverCode", "time": f"{session_type}Time (s)"}, inplace=True)
        return df
    
    return pd.DataFrame()


def save_session_results(race_round, session_type, driver_times):
    """
    Save session results to local JSON for future use.
    driver_times: dict of {driver_code: time_in_seconds}
    """
    data = {
        "drivers": [{"code": code, "time": time} for code, time in driver_times.items()]
    }
    
    filename = f"round_{race_round}_{session_type.lower()}.json"
    filepath = os.path.join(SESSION_DATA_DIR, filename)
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def get_2026_preseason_data():
    """Load pre-season testing data and return best lap per driver."""
    testing_data = load_testing_data()
    
    best_laps = {}
    
    for driver, time in testing_data["barcelona"]["fastest_laps"].items():
        best_laps[driver] = time
        
    for driver, time in testing_data["bahrain"]["fastest_laps"].items():
        if driver in best_laps:
            best_laps[driver] = min(best_laps[driver], time)
        else:
            best_laps[driver] = time
            
    df = pd.DataFrame(list(best_laps.items()), columns=["DriverCode", "TestingPace (s)"])
    return df


def get_driver_grid_2026():
    """Return the full 22-driver 2026 grid as a DataFrame."""
    drivers = []
    for name, code in DRIVER_MAPPING.items():
        team = DRIVER_TO_TEAM.get(code, "Unknown")
        drivers.append({
            "DriverCode": code,
            "DriverName": name,
            "Team": team
        })
    return pd.DataFrame(drivers)


def get_circuit_baseline(year, race):
    """
    Get the baseline qualifying lap time for a circuit from historical data.
    This is used to scale qualifying estimates for the 2026 grid.
    """
    quali_data = get_historical_qualifying_data(year, race)
    if not quali_data.empty:
        # Return the median qualifying time as the circuit baseline
        return quali_data["QualifyingTime (s)"].median()
    return None

def get_live_practice_data(year, race):
    """
    Attempt to fetch real practice data (FP3, FP2, or FP1) from FastF1 automatically.
    Returns the fastest lap per driver from the latest available practice session.
    """
    sessions = ["FP3", "FP2", "FP1"]
    for s_type in sessions:
        try:
            
            session = fastf1.get_session(year, race, s_type)
            session.load()
            laps = session.laps[["Driver", "LapTime"]].copy()
            laps.dropna(inplace=True)
            
            if not laps.empty:
                laps["PracticeTime (s)"] = laps["LapTime"].dt.total_seconds()
                
                # Filter out slow laps (out-laps, in-laps, installation laps)
                # Only keep laps above a minimum plausible F1 time (60s)
                # and within 107% of the session best (mirrors F1's qualifying rule)
                session_best = laps["PracticeTime (s)"].min()
                laps = laps[
                    (laps["PracticeTime (s)"] >= 60) &
                    (laps["PracticeTime (s)"] <= session_best * 1.07)
                ]
                
                if laps.empty:
                    continue
                
                fastest = laps.groupby("Driver")["PracticeTime (s)"].min().reset_index()
                fastest.rename(columns={"Driver": "DriverCode"}, inplace=True)
                
                return fastest, s_type
        except Exception:
            # Session likely hasn't happened yet in 2026 or no data available
            continue
            
    return pd.DataFrame(), None

def get_live_qualifying_data(year, race):
    """
    Attempt to fetch real qualifying data from FastF1 automatically.
    Returns the fastest lap per driver from the Q session.
    """
    try:
        session = fastf1.get_session(year, race, "Q")
        session.load()
        laps = session.laps[["Driver", "LapTime"]].copy()
        laps.dropna(inplace=True)
        if not laps.empty:
            laps["QualifyingTime (s)"] = laps["LapTime"].dt.total_seconds()
            session_best = laps["QualifyingTime (s)"].min()
            laps = laps[
                (laps["QualifyingTime (s)"] >= 60) &
                (laps["QualifyingTime (s)"] <= session_best * 1.07)
            ]
            if not laps.empty:
                fastest = laps.groupby("Driver")["QualifyingTime (s)"].min().reset_index()
                fastest.rename(columns={"Driver": "DriverCode"}, inplace=True)
                return fastest
    except Exception:
        pass
    return pd.DataFrame()
