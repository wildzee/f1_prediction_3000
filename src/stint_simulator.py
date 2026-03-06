"""
Stint-based race simulation — models a full race with tyre degradation,
pit stops, safety car probability, and lap-1 position changes.
"""
import numpy as np
import pandas as pd


# Tyre degradation rates (seconds per lap of performance loss)
TYRE_DEGRADATION = {
    "SOFT": 0.08,      # Fastest but degrades quickly
    "MEDIUM": 0.05,    # Balanced
    "HARD": 0.03,      # Slowest but most durable
}

# Optimal stint lengths per compound (laps before performance cliff)
STINT_LENGTHS = {
    "SOFT": 18,
    "MEDIUM": 28,
    "HARD": 40,
}

# Pit stop time loss (seconds)
PIT_STOP_LOSS = 22.0

# Compound pace deltas from soft-equivalent baseline
COMPOUND_DELTA = {
    "SOFT": 0.0,
    "MEDIUM": 0.7,
    "HARD": 1.5,
}


def simulate_race(drivers_df, total_laps, circuit_type="permanent", weather=None):
    """
    Simulate a full race for all drivers.
    
    Args:
        drivers_df: DataFrame with columns [DriverCode, Team, EstimatedRacePace, GridPosition]
        total_laps: Number of race laps
        circuit_type: 'permanent', 'street', or 'semi-street'
        weather: dict with 'pop' (rain probability)
    
    Returns:
        DataFrame with simulation results per driver including total race time,
        strategy used, and position changes.
    """
    if weather is None:
        weather = {"pop": 0.0}
    
    rain_prob = weather.get("pop", 0.0)
    
    # Safety car probability per circuit type
    sc_prob = {"permanent": 0.35, "semi-street": 0.50, "street": 0.65}
    safety_car_chance = sc_prob.get(circuit_type, 0.35)
    
    # Determine optimal strategy based on race length
    strategies = _get_strategies(total_laps, rain_prob)
    
    results = []
    
    for _, driver in drivers_df.iterrows():
        driver_code = driver["DriverCode"]
        base_pace = driver.get("EstimatedRacePace", driver.get("PredictedLapTime", 90.0))
        grid_pos = int(driver.get("GridPosition", 10))
        team = driver.get("Team", "Unknown")
        
        # Choose best strategy for this driver
        best_time = float("inf")
        best_strategy = None
        
        for strategy in strategies:
            total_time = _simulate_stints(
                base_pace, strategy, total_laps, grid_pos, safety_car_chance
            )
            if total_time < best_time:
                best_time = total_time
                best_strategy = strategy
        
        # Lap 1 position change simulation
        lap1_delta = _simulate_lap1(grid_pos, circuit_type)
        
        results.append({
            "DriverCode": driver_code,
            "Team": team,
            "GridPosition": grid_pos,
            "TotalRaceTime": round(best_time, 3),
            "Strategy": " → ".join(best_strategy),
            "PitStops": len(best_strategy) - 1,
            "Lap1PositionChange": lap1_delta,
        })
    
    results_df = pd.DataFrame(results)
    results_df.sort_values("TotalRaceTime", ascending=True, inplace=True)
    results_df["SimulatedPosition"] = range(1, len(results_df) + 1)
    results_df.reset_index(drop=True, inplace=True)
    
    return results_df


def _get_strategies(total_laps, rain_prob):
    """Generate candidate strategies based on race length."""
    strategies = []
    
    if rain_prob >= 0.75:
        # Wet: likely multiple stops, intermediate + wet compounds
        strategies = [
            ["MEDIUM", "SOFT"],
            ["MEDIUM", "MEDIUM"],
            ["HARD", "MEDIUM", "SOFT"],
        ]
    else:
        # Dry strategies
        # 1-stop
        if total_laps <= 50:
            strategies.append(["MEDIUM", "HARD"])
            strategies.append(["SOFT", "HARD"])
            strategies.append(["HARD", "SOFT"])
        
        # 2-stop
        strategies.append(["SOFT", "MEDIUM", "SOFT"])
        strategies.append(["MEDIUM", "HARD", "SOFT"])
        strategies.append(["SOFT", "HARD", "MEDIUM"])
        
        # Long races — possible 3-stop
        if total_laps >= 60:
            strategies.append(["SOFT", "MEDIUM", "MEDIUM", "SOFT"])
    
    return strategies


def _simulate_stints(base_pace, strategy, total_laps, grid_pos, sc_chance):
    """Simulate race time for a given strategy."""
    total_time = 0.0
    laps_done = 0
    num_stops = len(strategy) - 1
    
    # Distribute laps across stints proportionally to compound durability
    stint_durations = []
    total_durability = sum(STINT_LENGTHS.get(c, 25) for c in strategy)
    for compound in strategy:
        proportion = STINT_LENGTHS.get(compound, 25) / total_durability
        stint_laps = max(1, round(total_laps * proportion))
        stint_durations.append(stint_laps)
    
    # Adjust last stint to exactly match total_laps
    stint_durations[-1] = total_laps - sum(stint_durations[:-1])
    if stint_durations[-1] <= 0:
        stint_durations[-1] = 1
    
    for stint_idx, (compound, stint_laps) in enumerate(zip(strategy, stint_durations)):
        degradation = TYRE_DEGRADATION.get(compound, 0.05)
        compound_delta = COMPOUND_DELTA.get(compound, 0.5)
        
        for lap in range(stint_laps):
            # Base lap time + compound delta + degradation over stint
            lap_time = base_pace + compound_delta + (degradation * lap)
            
            # First lap: traffic/position penalty
            if laps_done == 0:
                lap_time += max(0, (grid_pos - 1) * 0.15)  # ~0.15s per grid slot
            
            total_time += lap_time
            laps_done += 1
        
        # Pit stop time (except after last stint)
        if stint_idx < num_stops:
            total_time += PIT_STOP_LOSS
    
    # Safety car simulation (random but consistent via expected value)
    # Average SC adds ~15s to everyone, ~40% chance
    sc_time_impact = sc_chance * 12.0  # expected time addition
    total_time += sc_time_impact
    
    return total_time


def _simulate_lap1(grid_pos, circuit_type):
    """Simulate typical lap 1 position changes based on grid slot and circuit."""
    # Front-runners tend to maintain, mid-field gains, back gets chaos
    np.random.seed(grid_pos * 7)  # deterministic per driver
    
    if grid_pos <= 3:
        delta = np.random.choice([-1, 0, 0, 0, 1])
    elif grid_pos <= 10:
        delta = np.random.choice([-2, -1, 0, 1, 1, 2])
    else:
        delta = np.random.choice([-3, -2, -1, 0, 1, 2, 3])
    
    # Street circuits: more chaos on lap 1
    if circuit_type == "street":
        delta += np.random.choice([-1, 0, 0, 1])
    
    return delta


# Race lap counts per circuit (approximate for 300km)
RACE_LAPS = {
    "Australia": 58, "Bahrain": 57, "Saudi Arabia": 50, "Japan": 53,
    "China": 56, "Miami": 57, "Monaco": 78, "Canada": 70,
    "Spain": 66, "Austria": 71, "Great Britain": 52, "Hungary": 70,
    "Belgium": 44, "Netherlands": 72, "Italy": 53, "Azerbaijan": 51,
    "Singapore": 62, "United States": 56, "Mexico": 71, "Brazil": 71,
    "Las Vegas": 50, "Qatar": 57, "Abu Dhabi": 58, "Madrid": 66,
}
