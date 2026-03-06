import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime

# Setup Open-Meteo API client with cache and retry
cache_session = requests_cache.CachedSession('.weather_cache', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def get_weather_forecast(lat, lon, race_date_str):
    """
    Fetches the weather forecast for a given latitude, longitude, and race date.
    Uses Open-Meteo (free, no API key, 16-day forecast range).
    Returns a dictionary with rain probability, temperature, and description.
    """
    default_weather = {"pop": 0.0, "temp": 22.0, "description": "Unknown - using historical average", "wind_speed": 10.0, "wind_gusts": 20.0}

    try:
        race_date = datetime.strptime(race_date_str, "%Y-%m-%d")
        today = datetime.now()
        days_away = (race_date - today).days

        # Open-Meteo supports up to 16 days ahead
        if days_away > 16 or days_away < 0:
            return default_weather

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "precipitation_probability_max",
                "temperature_2m_max",
                "temperature_2m_min",
                "weathercode",
                "wind_speed_10m_max",
                "wind_gusts_10m_max"
            ],
            "timezone": "auto",
            "forecast_days": min(days_away + 1, 16),
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        daily = response.Daily()

        # Build a DataFrame of daily forecasts
        dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )

        daily_df = pd.DataFrame({
            "date": dates.date,
            "pop": daily.Variables(0).ValuesAsNumpy() / 100.0,   # convert % → 0-1
            "temp_max": daily.Variables(1).ValuesAsNumpy(),
            "temp_min": daily.Variables(2).ValuesAsNumpy(),
            "weathercode": daily.Variables(3).ValuesAsNumpy(),
            "wind_speed": daily.Variables(4).ValuesAsNumpy(),
            "wind_gusts": daily.Variables(5).ValuesAsNumpy(),
        })

        # Find the row matching race date
        target_date = race_date.date()
        row = daily_df[daily_df["date"] == target_date]

        if row.empty:
            return default_weather

        row = row.iloc[0]
        temp = round((row["temp_max"] + row["temp_min"]) / 2, 1)
        pop = float(row["pop"])
        description = _wmo_to_description(int(row["weathercode"]))

        wind_speed = round(float(row["wind_speed"]), 1)
        wind_gusts = round(float(row["wind_gusts"]), 1)

        return {"pop": pop, "temp": temp, "description": description, "wind_speed": wind_speed, "wind_gusts": wind_gusts}

    except Exception as e:
        print(f"Open-Meteo weather error: {e}")
        return default_weather


def _wmo_to_description(code):
    """Convert WMO weather interpretation code to a readable string."""
    wmo_map = {
        0: "Clear sky",
        1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Icy fog",
        51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
        61: "Slight rain", 63: "Rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Snow", 75: "Heavy snow",
        80: "Slight showers", 81: "Showers", 82: "Heavy showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail",
    }
    return wmo_map.get(code, f"Weather code {code}")
