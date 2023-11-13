from datetime import datetime, timedelta, timezone
from pyowm import OWM
from pyowm.utils import config
from matplotlib import pyplot as plt
import pandas as pd


def query_weather_data(api_key_path: str, city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generates a dataset of weather data for a given city and time range.

    Args:
        api_key_path (str): The path to the file containing the OpenWeatherMap API key.
        city (str): The name of the city to get weather data for.
        start_date (str): The start date of the time range to get weather data for, in the format 'YYYY-MM-DD'.
        end_date (str): The end date of the time range to get weather data for, in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the weather data for the specified time range.
    """
    with open(api_key_path, 'r') as f:
        api_key = f.read().strip()

    # Config and manager setup
    config_dict = config.get_default_config_for_subscription_type('professional')
    owm = OWM(api_key, config_dict)
    mgr = owm.weather_manager()
    geo_mgr = owm.geocoding_manager()

    # Find city
    list_of_locations = geo_mgr.geocode(city, limit=3)

    # Format times
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    delta = timedelta(hours=1)
    current_datetime = end_datetime

    # Get data
    data = []
    while current_datetime <= start_datetime:
        window_epoch = int(current_datetime.replace(tzinfo=timezone.utc).timestamp())
        history = mgr.one_call_history(lat=list_of_locations[0].lat, lon=list_of_locations[0].lon, dt=window_epoch)
        data.append(history.current.to_dict())
        current_datetime += delta

    # Export to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Plot
    plt.plot([d['temp'] for d in df['temperature']])
    plt.show()
    
    return df



if __name__ == '__main__':
    api_key_path = '/home/darcones/Projects/API keys/owm.txt'
    city = 'Worms'
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    df = query_weather_data(api_key_path, city, start_date, end_date)
    df.to_json('example.json')
    