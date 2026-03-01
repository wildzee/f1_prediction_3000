import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather_forecast(lat, lon, race_date_str):
    """
    Fetches the weather forecast for a given latitude, longitude, and date.
    Returns a dictionary with rain probability and temperature.
    """
    default_weather = {"pop": 0.0, "temp": 22.0, "description": "Unknown - using historical average"}
    
    if not API_KEY or API_KEY == "your_openweather_api_key_here":
        # Missing API Key, return defaults
        return default_weather
        
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            return default_weather
            
        weather_data = response.json()
        
        race_date = datetime.strptime(race_date_str, "%Y-%m-%d")
        
        # OpenWeather's free forecast API only goes up to 5 days
        today = datetime.now()
        days_away = (race_date - today).days
        
        if days_away > 5 or days_away < 0:
            # Race is too far out or in the past for the 5-day forecast
            return default_weather
            
        # Target Sunday afternoon ~14:00 local time
        target_time_str = f"{race_date_str} 15:00:00" 
        
        # Try to find exact match
        forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == target_time_str), None)
        
        # If not exact match, get the closest one on that day
        if not forecast_data:
            day_forecasts = [f for f in weather_data["list"] if f["dt_txt"].startswith(race_date_str)]
            if day_forecasts:
                # Just take the mid-day one
                forecast_data = day_forecasts[len(day_forecasts)//2]
                
        if forecast_data:
            pop = forecast_data.get("pop", 0)
            temp = forecast_data["main"]["temp"]
            desc = forecast_data["weather"][0]["description"] if forecast_data.get("weather") else ""
            return {"pop": pop, "temp": temp, "description": desc}
            
    except Exception as e:
        print(f"Weather API Error: {e}")
        
    return default_weather
