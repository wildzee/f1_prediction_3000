import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Try XGBoost first, fall back to sklearn GradientBoosting if libomp is missing
try:
    from xgboost import XGBRegressor
    _USE_XGBOOST = True
except (ImportError, Exception):
    from sklearn.ensemble import GradientBoostingRegressor
    _USE_XGBOOST = False


def train_model(X, y):
    """
    Train a model matching the original prediction scripts.
    
    Uses XGBoost with monotone constraints if available.
    Falls back to sklearn GradientBoostingRegressor otherwise.
    
    Monotone constraints (XGBoost only):
      col 0 QualifyingTime:       +1 (higher quali time → higher/slower race time)
      col 1 RainProbability:       0 (no constraint)
      col 2 Temperature:           0 (no constraint)
      col 3 TeamPerformanceScore: -1 (higher team score → lower/faster race time)
      col 4 CleanAirRacePace:     +1 (higher pace value = slower)
    """
    if _USE_XGBOOST:
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.9,
            max_depth=3,
            random_state=39,
            monotone_constraints='(1, 0, 0, -1, 1)'
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.9,
            max_depth=3,
            random_state=39
        )
    
    # Train-test split matching original scripts
    if len(X) > 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=39
        )
    else:
        # Too small for split, train on all
        X_train, y_train = X, y
        X_test, y_test = X, y
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, mae


def predict_race(model, X):
    """Predict race lap times for all drivers."""
    return model.predict(X)


def get_feature_importances(model, feature_names):
    """Return feature importances as a sorted DataFrame."""
    importances = model.feature_importances_
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)
    return df
