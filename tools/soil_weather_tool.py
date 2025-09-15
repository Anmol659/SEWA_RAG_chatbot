# tools/soil_weather_tool.py

import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np

# --- API Client Setup ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- Geocoding (Location to Coordinates) ---
PUNJAB_COORDINATES = {
    "amritsar": {"latitude": 31.63, "longitude": 74.87},
    "ludhiana": {"latitude": 30.90, "longitude": 75.86},
    "jalandhar": {"latitude": 31.33, "longitude": 75.58},
    "patiala": {"latitude": 30.34, "longitude": 76.38},
    "bathinda": {"latitude": 30.21, "longitude": 74.94},
    "firozpur": {"latitude": 30.92, "longitude": 74.60}
}

def get_soil_data(location: str) -> str:
    """
    Analyzes and retrieves recent soil temperature and moisture for a given location.
    This tool provides critical data for irrigation scheduling and understanding
    the soil's health, which is vital for crop growth. The model will determine
    the location from the user's query.

    Args:
        location: The name of the city or district in Punjab.

    Returns:
        A formatted string containing the latest soil data, or an error message.
    """
    location_lower = location.lower()
    if location_lower not in PUNJAB_COORDINATES:
        return f"Error: Location '{location}' is not a recognized key location in Punjab for soil data."

    coords = PUNJAB_COORDINATES[location_lower]

    # --- API Request Parameters - Using Forecast API to get recent past data ---
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": ["soil_temperature_0cm", "soil_temperature_6cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm"],
        "past_days": 1  # Fetch data for the last day to get the most recent reading
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # --- Process Data ---
        hourly = response.Hourly()
        if hourly is None:
            return f"No recent soil data found for {location.title()}."

        def get_last_valid(arr):
            """Filters out NaNs and gets the last element if the array is not all-NaN."""
            valid_arr = arr[~np.isnan(arr)]
            return valid_arr[-1] if valid_arr.size > 0 else None

        soil_temp_0cm = get_last_valid(hourly.Variables(0).ValuesAsNumpy())
        soil_temp_6cm = get_last_valid(hourly.Variables(1).ValuesAsNumpy())
        soil_moisture_0_1cm = get_last_valid(hourly.Variables(2).ValuesAsNumpy())
        soil_moisture_1_3cm = get_last_valid(hourly.Variables(3).ValuesAsNumpy())

        if any(v is None for v in [soil_temp_0cm, soil_temp_6cm, soil_moisture_0_1cm, soil_moisture_1_3cm]):
            return f"Could not retrieve complete recent soil data for {location.title()}. Some values were missing."

        return (
            f"Latest Soil Analysis for {location.title()}:\n"
            f"- Surface Temperature: {soil_temp_0cm:.2f}°C\n"
            f"- Temperature at 6cm depth: {soil_temp_6cm:.2f}°C\n"
            f"- Surface Soil Moisture (0-1cm): {soil_moisture_0_1cm:.3f} m³/m³\n"
            f"- Root Zone Soil Moisture (1-3cm): {soil_moisture_1_3cm:.3f} m³/m³"
        )
    except Exception as e:
        return f"An unexpected error occurred while fetching soil data: {e}"

if __name__ == "__main__":
    print("--- Testing TerraMoist Soil Data Tool ---")
    test_location = "Ludhiana"
    soil_report = get_soil_data(test_location)
    print(soil_report)