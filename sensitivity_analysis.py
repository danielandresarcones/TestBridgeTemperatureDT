from SALib.sample import saltelli
from SALib.analyze import sobol
from TestBridgeTemperatureDT.dummy_model import DummyModel
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

def sensitivity_analysis(num_samples, num_param, input_names, bounds, weather_file, output_path):
    # Generate samples
    problem = {
        'num_vars': len(input_names),
        'names': input_names,
        'bounds': bounds
    }
    param_values = saltelli.sample(problem, num_samples, calc_second_order=True)

    # Load weather data
    weather_data = pd.read_hdf(weather_file, key='data')
    temperatures = weather_data['Temperature'].to_numpy()
    wind_speeds = weather_data['Wind Speed'].to_numpy()
    sun_zeniths = weather_data['Sun Zenith'].to_numpy()
    sun_azimuths = weather_data['Sun Azimuth'].to_numpy()
    sun_elevations = weather_data['Sun Elevation'].to_numpy()

    # Initialize model
    model = DummyModel(0, 0, 0, 0, 0, 10, 10)
    model.store_inputs(temperatures, wind_speeds, sun_zeniths, sun_azimuths, sun_elevations)
    grid = (model.update_grid(param_values[0][0], param_values[0][1], param_values[0][2], param_values[0][3], param_values[0][4])).flatten()

    # Run model
    Y = np.zeros([param_values.shape[0], grid.shape[0]])
    for i, X in enumerate(param_values):
        Y[i] = (model.update_grid(X[0], X[1], X[2], X[3], X[4])).flatten()
        

    # Perform analysis
    Si = sobol.analyze(problem, Y.T.flatten(), calc_second_order=True)

    # Plot Sobol indices as histogram
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(input_names)))
    plt.bar(input_names, Si['ST'], color=colors)
    plt.xticks(rotation=90)
    plt.ylabel('Sobol Index')
    plt.title('Sobol Indices')
    plt.savefig('figures/sobol_indices.png')
    plt.show()

    # Get the indices of the num_param most relevant variables
    sorted_indices = np.argsort(Si['ST'])[::-1][:num_param]

    # Create a new problem definition with only the most relevant variables
    new_problem = {
        'num_vars': num_param,
        'names': [input_names[i] for i in sorted_indices],
        'bounds': [bounds[i] for i in sorted_indices]
    }

    # Generate new samples with only the most relevant variables
    new_param_values = saltelli.sample(new_problem, num_samples, calc_second_order=True)

    # Run model with new samples
    new_Y = np.zeros([new_param_values.shape[0], grid.shape[0]])
    for i, X in enumerate(new_param_values):
        full_X = np.zeros(len(input_names))
        full_X[sorted_indices] = X
        new_Y[i] = (model.update_grid(full_X[0], full_X[1], full_X[2], full_X[3], full_X[4])).flatten()

    # Perform analysis with new samples
    new_Si = sobol.analyze(new_problem, new_Y.T.flatten(), calc_second_order=True)

    # Plot Sobol indices as histogram
    colors = plt.cm.Set1(np.linspace(0, 1, len(input_names)))
    plt.bar([input_names[i] for i in sorted_indices], new_Si['ST'], color=colors)
    plt.xticks(rotation=90)
    plt.ylabel('Sobol Index')
    plt.title('Sobol Indices')
    plt.savefig('figures/sobol_indices_reduced.png')
    plt.show()

    # Save results to file with param names
    results = pd.DataFrame({
        'S1': new_Si['S1'],
        'ST': new_Si['ST'],
        'S1_conf': new_Si['S1_conf'],
        'ST_conf': new_Si['ST_conf']
    }, index=[input_names[i] for i in sorted_indices])
    results.to_json(output_path)

