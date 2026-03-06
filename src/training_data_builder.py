"""
Build a multi-year historical training dataset from FastF1.
Fetches 2022-2025 race + qualifying data for all circuits and saves as a CSV.
Includes: grid position, qualifying time, sector times, tyre compound,
circuit type, driver/team performance, weather, finishing position.
"""
import fastf1
import pandas as pd
import numpy as np
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "f1_cache"
OUTPUT = DATA_DIR / "historical_training.csv"

# Circuit type classification
CIRCUIT_TYPES = {
    "Australia": "street-ish", "Bahrain": "permanent", "Saudi Arabia": "street",
    "Japan": "permanent", "China": "permanent", "Miami": "street",
    "Imola": "permanent", "Monaco": "street", "Canada": "semi-street",
    "Spain": "permanent", "Austria": "permanent", "Great Britain": "permanent",
    "Hungary": "permanent", "Belgium": "permanent", "Netherlands": "permanent",
    "Italy": "permanent", "Azerbaijan": "street", "Singapore": "street",
    "United States": "permanent", "Mexico": "permanent", "Brazil": "permanent",
    "Las Vegas": "street", "Qatar": "permanent", "Abu Dhabi": "permanent",
}

CIRCUIT_TYPE_MAP = {"permanent": 0, "semi-street": 1, "street-ish": 1, "street": 2}


def build_training_data(years=None, races_per_year=None):
    """Build training data from multiple seasons."""
    if years is None:
        years = [2022, 2023, 2024, 2025]

    fastf1.Cache.enable_cache(str(CACHE_DIR))
    all_rows = []

    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        events = schedule[schedule["EventFormat"] != "testing"]

        if races_per_year:
            events = events.head(races_per_year)

        for _, event in events.iterrows():
            race_name = event["EventName"]
            race_round = event["RoundNumber"]
            print(f"\n📥 {year} R{race_round}: {race_name}")

            try:
                rows = _process_race_weekend(year, race_round, race_name)
                all_rows.extend(rows)
                print(f"   ✅ {len(rows)} driver records")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT, index=False)
    print(f"\n🎯 Saved {len(df)} total records to {OUTPUT}")
    return df


def _process_race_weekend(year, round_num, race_name):
    """Extract features for one race weekend."""
    rows = []

    # --- Load Race session ---
    try:
        race = fastf1.get_session(year, round_num, "R")
        race.load()
    except Exception:
        return []

    # --- Load Qualifying session ---
    try:
        quali = fastf1.get_session(year, round_num, "Q")
        quali.load()
        quali_laps = quali.laps[["Driver", "LapTime"]].dropna()
        quali_laps["QTime"] = quali_laps["LapTime"].dt.total_seconds()
        quali_best = quali_laps.groupby("Driver")["QTime"].min().to_dict()
    except Exception:
        quali_best = {}

    # --- Race results ---
    race_results = race.results
    if race_results is None or race_results.empty:
        return []

    # --- Race laps for sector/tyre analysis ---
    try:
        race_laps = race.laps
    except Exception:
        race_laps = pd.DataFrame()

    # --- Determine circuit type ---
    circuit_name = _guess_circuit_name(race_name)
    circuit_type = CIRCUIT_TYPES.get(circuit_name, "permanent")
    circuit_type_num = CIRCUIT_TYPE_MAP.get(circuit_type, 0)

    for _, result in race_results.iterrows():
        driver = result.get("Abbreviation", "")
        if not driver:
            continue

        # Grid position
        grid_pos = result.get("GridPosition", 20)

        # Finishing position
        finish_pos = result.get("Position", None)
        if pd.isna(finish_pos):
            continue
        finish_pos = int(finish_pos)

        # Status (DNF detection)
        status = str(result.get("Status", "Finished"))
        dnf = 0 if "Finished" in status or status.startswith("+") else 1

        # Team
        team = str(result.get("TeamName", "Unknown"))

        # Qualifying time
        q_time = quali_best.get(driver, np.nan)

        # Sector analysis from race laps
        s1, s2, s3, avg_lap, tyre_stints = np.nan, np.nan, np.nan, np.nan, ""
        if not race_laps.empty:
            driver_laps = race_laps[race_laps["Driver"] == driver].copy()
            if not driver_laps.empty:
                for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
                    if col in driver_laps.columns:
                        driver_laps[col + "_s"] = driver_laps[col].dt.total_seconds()

                valid_laps = driver_laps.copy()
                if "LapTime" in valid_laps.columns:
                    valid_laps["LapTime_s"] = valid_laps["LapTime"].dt.total_seconds()
                    lap_min = valid_laps["LapTime_s"].min()
                    valid_laps = valid_laps[
                        (valid_laps["LapTime_s"] >= 60) &
                        (valid_laps["LapTime_s"] <= lap_min * 1.10)
                    ]
                    avg_lap = valid_laps["LapTime_s"].mean() if not valid_laps.empty else np.nan

                if "Sector1Time_s" in valid_laps.columns:
                    s1 = valid_laps["Sector1Time_s"].median()
                if "Sector2Time_s" in valid_laps.columns:
                    s2 = valid_laps["Sector2Time_s"].median()
                if "Sector3Time_s" in valid_laps.columns:
                    s3 = valid_laps["Sector3Time_s"].median()

                # Tyre stints
                if "Compound" in driver_laps.columns:
                    compounds = driver_laps["Compound"].dropna().unique()
                    tyre_stints = ",".join(str(c) for c in compounds)

        rows.append({
            "Year": year,
            "Round": int(round_num),
            "Race": race_name,
            "Driver": driver,
            "Team": team,
            "GridPosition": int(grid_pos) if not pd.isna(grid_pos) else 20,
            "QualifyingTime": q_time,
            "AvgRaceLapTime": avg_lap,
            "Sector1": s1,
            "Sector2": s2,
            "Sector3": s3,
            "TyreStints": tyre_stints,
            "CircuitType": circuit_type_num,
            "FinishPosition": finish_pos,
            "DNF": dnf,
        })

    return rows


def _guess_circuit_name(event_name):
    """Map event name to our circuit type key."""
    mappings = {
        "Australian": "Australia", "Bahrain": "Bahrain", "Saudi Arabian": "Saudi Arabia",
        "Japanese": "Japan", "Chinese": "China", "Miami": "Miami",
        "Emilia Romagna": "Imola", "Monaco": "Monaco", "Canadian": "Canada",
        "Spanish": "Spain", "Austrian": "Austria", "British": "Great Britain",
        "Hungarian": "Hungary", "Belgian": "Belgium", "Dutch": "Netherlands",
        "Italian": "Italy", "Azerbaijan": "Azerbaijan", "Singapore": "Singapore",
        "United States": "United States", "Mexico": "Mexico", "Mexican": "Mexico",
        "São Paulo": "Brazil", "Brazilian": "Brazil",
        "Las Vegas": "Las Vegas", "Qatar": "Qatar", "Abu Dhabi": "Abu Dhabi",
    }
    for key, val in mappings.items():
        if key.lower() in event_name.lower():
            return val
    return event_name.replace(" Grand Prix", "")


if __name__ == "__main__":
    build_training_data()
