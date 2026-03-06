"""
Dynamic performance tracker — saves predictions, compares to actual results,
and updates team/driver performance weights automatically as the season progresses.
"""
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.constants import CLEAN_AIR_PACE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions_log.json"
ACCURACY_FILE = DATA_DIR / "accuracy_log.json"


def save_prediction(round_num, race_name, prediction_type, predictions_df):
    """Save a prediction run for later comparison with actual results.
    prediction_type: 'qualifying' or 'race'
    predictions_df: DataFrame with at least DriverCode and predicted position/time columns
    """
    log = _load_json(PREDICTIONS_FILE, default=[])
    
    entry = {
        "round": int(round_num),
        "race": race_name,
        "type": prediction_type,
        "timestamp": pd.Timestamp.now().isoformat(),
        "predictions": {}
    }
    
    for _, row in predictions_df.iterrows():
        driver = row.get("DriverCode", "")
        if not driver:
            continue
        entry["predictions"][driver] = {
            "predicted_position": int(row.name) + 1 if "Position" not in row else int(row.get("Position", 0)),
            "predicted_time": float(row.get("EstimatedQuali (s)", row.get("PredictedLapTime", 0))),
        }
    
    # Replace any existing prediction for the same round + type
    log = [e for e in log if not (e["round"] == int(round_num) and e["type"] == prediction_type)]
    log.append(entry)
    
    _save_json(PREDICTIONS_FILE, log)
    return True


def save_actual_result(round_num, race_name, result_type, results_df):
    """Save actual qualifying/race results for accuracy comparison.
    results_df: DataFrame with DriverCode and actual finish position/time
    """
    log = _load_json(ACCURACY_FILE, default=[])
    
    entry = {
        "round": int(round_num),
        "race": race_name,
        "type": result_type,
        "timestamp": pd.Timestamp.now().isoformat(),
        "actuals": {}
    }
    
    for _, row in results_df.iterrows():
        driver = row.get("DriverCode", row.get("Driver", ""))
        if not driver:
            continue
        entry["actuals"][driver] = {
            "actual_position": int(row.get("Position", 0)),
            "actual_time": float(row.get("Time", 0)),
        }
    
    log = [e for e in log if not (e["round"] == int(round_num) and e["type"] == result_type)]
    log.append(entry)
    
    _save_json(ACCURACY_FILE, log)
    return True


def get_accuracy_report():
    """Compare predictions vs actuals for all completed races.
    Returns a DataFrame with per-race accuracy metrics."""
    predictions = _load_json(PREDICTIONS_FILE, default=[])
    actuals = _load_json(ACCURACY_FILE, default=[])
    
    if not predictions or not actuals:
        return pd.DataFrame()
    
    rows = []
    for pred in predictions:
        # Find matching actual
        actual = next(
            (a for a in actuals if a["round"] == pred["round"] and a["type"] == pred["type"]),
            None
        )
        if actual is None:
            continue
        
        # Compare positions
        position_errors = []
        top3_correct = 0
        top10_correct = 0
        
        pred_drivers = pred["predictions"]
        actual_drivers = actual["actuals"]
        
        for driver, p in pred_drivers.items():
            if driver in actual_drivers:
                pred_pos = p["predicted_position"]
                actual_pos = actual_drivers[driver]["actual_position"]
                error = abs(pred_pos - actual_pos)
                position_errors.append(error)
                
                if pred_pos <= 3 and actual_pos <= 3:
                    top3_correct += 1
                if pred_pos <= 10 and actual_pos <= 10:
                    top10_correct += 1
        
        if position_errors:
            rows.append({
                "Round": pred["round"],
                "Race": pred["race"],
                "Type": pred["type"],
                "MAE_Position": round(np.mean(position_errors), 2),
                "Median_Error": round(np.median(position_errors), 1),
                "Top3_Correct": top3_correct,
                "Top10_Correct": top10_correct,
                "Drivers_Compared": len(position_errors),
            })
    
    return pd.DataFrame(rows)


def get_dynamic_pace(recent_n=3):
    """Calculate dynamic driver pace from saved actual results.
    Uses the most recent N race results to update CLEAN_AIR_PACE dynamically.
    Falls back to static constants if no results are available."""
    actuals = _load_json(ACCURACY_FILE, default=[])
    
    if not actuals:
        return CLEAN_AIR_PACE.copy()
    
    # Sort by round, take most recent N race results
    race_results = [a for a in actuals if a["type"] == "race"]
    race_results.sort(key=lambda x: x["round"], reverse=True)
    recent = race_results[:recent_n]
    
    if not recent:
        return CLEAN_AIR_PACE.copy()
    
    # Collect actual times per driver
    driver_times = {}
    for result in recent:
        for driver, data in result["actuals"].items():
            t = data.get("actual_time", 0)
            if t > 0:
                driver_times.setdefault(driver, []).append(t)
    
    # Build dynamic pace: average of recent race lap times
    dynamic_pace = CLEAN_AIR_PACE.copy()
    for driver, times in driver_times.items():
        dynamic_pace[driver] = round(np.mean(times), 3)
    
    return dynamic_pace


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default if default is not None else {}


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
