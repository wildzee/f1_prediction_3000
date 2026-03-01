import json
import os
from pathlib import Path

# Paths to data files
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Driver mappings for 2026 (22 cars)
DRIVER_MAPPING = {
    "Max Verstappen": "VER",
    "Isack Hadjar": "HAD",
    "George Russell": "RUS",
    "Andrea Kimi Antonelli": "ANT",
    "Lewis Hamilton": "HAM",
    "Charles Leclerc": "LEC",
    "Lando Norris": "NOR",
    "Oscar Piastri": "PIA",
    "Fernando Alonso": "ALO",
    "Lance Stroll": "STR",
    "Pierre Gasly": "GAS",
    "Franco Colapinto": "COL",
    "Alexander Albon": "ALB",
    "Carlos Sainz": "SAI",
    "Liam Lawson": "LAW",
    "Arvid Lindblad": "LIN",
    "Oliver Bearman": "BEA",
    "Esteban Ocon": "OCO",
    "Nico Hülkenberg": "HUL",
    "Gabriel Bortoleto": "BOR",
    "Valtteri Bottas": "BOT",
    "Sergio Pérez": "PER"
}

# Reverse mapping
CODE_TO_NAME = {v: k for k, v in DRIVER_MAPPING.items()}

# Driver to Team mapping (2026)
DRIVER_TO_TEAM = {
    "VER": "Red Bull",
    "HAD": "Red Bull",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "HAM": "Ferrari",
    "LEC": "Ferrari",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "COL": "Alpine",
    "ALB": "Williams",
    "SAI": "Williams",
    "LAW": "Racing Bulls",
    "LIN": "Racing Bulls",
    "BEA": "Haas",
    "OCO": "Haas",
    "HUL": "Audi",
    "BOR": "Audi",
    "BOT": "Cadillac",
    "PER": "Cadillac"
}

# Clean Air Race Pace approximations (lower is better, used as a baseline feature)
# Averaged out from previous years and testing data
CLEAN_AIR_PACE = {
    "VER": 93.19, "HAM": 93.42, "LEC": 93.42, "NOR": 93.43, "ALO": 94.78,
    "PIA": 93.23, "RUS": 93.83, "SAI": 94.49, "STR": 95.31, "GAS": 94.50,
    "OCO": 95.68, "BEA": 95.00, "HAD": 94.50, "ANT": 94.00, "COL": 95.10,
    "ALB": 95.20, "LAW": 94.80, "LIN": 95.50, "HUL": 95.34, "BOR": 95.70,
    "BOT": 96.00, "PER": 95.80
}

# Team Colors for UI display
TEAM_COLORS = {
    "Red Bull": "#3671C6",
    "Mercedes": "#27F4D2",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "Racing Bulls": "#6692FF",
    "Haas": "#B6BABD",
    "Audi": "#F50537",
    "Cadillac": "#FFD700"  # Placeholder color for Cadillac
}

# Wet Performance Factors (from original implementation, slightly modified for 2026 lineup)
WET_PERFORMANCE = {
    "VER": 0.975, "HAM": 0.976, "LEC": 0.975, "NOR": 0.978, "ALO": 0.972,
    "RUS": 0.968, "SAI": 0.978, "OCO": 0.981, "GAS": 0.978, "STR": 0.979,
    "PIA": 0.975, "ALB": 0.980, "HUL": 0.985, "BEA": 0.980, "HAD": 0.980,
    "ANT": 0.975, "COL": 0.980, "LAW": 0.980, "LIN": 0.985, "BOR": 0.985,
    "BOT": 0.975, "PER": 0.980
}

def load_calendar():
    with open(DATA_DIR / "calendar_2026.json", "r") as f:
        return json.load(f)

def load_testing_data():
    with open(DATA_DIR / "testing_2026.json", "r") as f:
        return json.load(f)
