import pandas as pd
from datetime import datetime, timedelta
from TestBridgeTemperatureDT.query_weather_data import query_weather_data
from astral.sun import zenith, azimuth, elevation
from astral import LocationInfo
import matplotlib.pyplot as plt


def generate_data_set(api_key_path: str, city: str, start_date: str, end_date: str, output_path: str, show: bool = False):

    weather_data = query_weather_data(api_key_path, city, start_date, end_date)

    city_info = LocationInfo(city, 'Europe', 'Berlin', 49.63, 8.38)
    sun_azimuths = []
    sun_zeniths = []
    sun_elevations = []
    for ref_time in weather_data['reference_time']:
        # s = sun(city_info.observer, date=datetime.fromtimestamp(ref_time))
        sun_azimuths.append(azimuth(city_info.observer, datetime.fromtimestamp(ref_time)))
        sun_zeniths.append(zenith(city_info.observer, datetime.fromtimestamp(ref_time)))
        sun_elevations.append(elevation(city_info.observer, datetime.fromtimestamp(ref_time)))

    temperatures = [d['temp'] for d in weather_data['temperature']]
    wind_speeds = [d['speed'] for d in weather_data['wind']]
    cloud_coverages = weather_data['clouds']
    reference_time = weather_data['reference_time']

    # Create a pandas DataFrame with the data we want to plot
    data = pd.DataFrame({
        'Temperature': temperatures,
        'Wind Speed': wind_speeds,
        'Cloud Coverage': cloud_coverages.to_dict().values(), # somehow this is a pandas Series, not a list
        'Sun Azimuth': sun_azimuths,
        'Sun Zenith': sun_zeniths,
        'Sun Elevation': sun_elevations
    }, index=reference_time)
    
    # Store data as hdf5 at output path
    data.to_hdf(output_path, key='data', mode='w')

    # Create a plot of the data
    if show:
        for col in data.columns:
            plt.plot(data.index, data[col])
            plt.title(col)
            plt.xlabel('Time')
            plt.ylabel(col)
            plt.show()

    # return data


if __name__ == '__main__':
    output_path = 'example.hdf5'
    api_key_path = '/home/darcones/Projects/API keys/owm.txt'
    city = 'Worms'
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    df = generate_data_set(api_key_path, city, start_date, end_date, output_path=output_path, show=False)
    df.to_json('example_full.json')
    