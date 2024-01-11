
import matplotlib.pyplot as plt
from TestBridgeTemperatureDT.dummy_model import DummyModel
from TestBridgeTemperatureDT.utils import load_dataset
import pandas as pd
import arviz as az
import h5py
import json

def postprocess_case(case_name, figure_settings, results_data_path, output_data_path, prior, ground_truth, biased):

    dataset_mean, bias_mean = load_dataset(results_data_path, biased=biased)
    dataset_stdp, bias_stdp = load_dataset(results_data_path.replace(".h5","_plus_std.h5"), biased=biased)
    dataset_stdn, bias_stdn = load_dataset(results_data_path.replace(".h5","_minus_std.h5"), biased=biased)

    # Plot the fitted model
    if biased:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figure_settings["figure_size_double"])
    else:
        fig, ax1 = plt.subplots(ncols=1, figsize=figure_settings["figure_size"])

    ax1.plot(dataset_mean[0],'b-', label=case_name)
    ax1.plot(dataset_stdp,'b--', label='__no_label__')
    ax1.plot(dataset_stdn,'b--', label='__no_label__')
    ax1.plot(ground_truth,'r.', label='Ground truth')
    ax1.plot(prior,'k-', label='Prior Model')
    ax1.legend()
    ax1.set_title('Fitted Model')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature')

    if biased:
        ax2.plot(dataset_mean[0]+bias_mean[0],'b-', label=case_name)
        ax2.plot(dataset_mean[0]+bias_stdp[0],'b--', label='__no_label__')
        ax2.plot(dataset_mean[0]+bias_stdn[0],'b--', label='__no_label__')
        ax2.plot(ground_truth,'r.', label='Ground truth')
        ax2.plot(prior,'k-', label='Prior Model')
        ax2.legend()
        ax2.set_title('Bias-corrected Model')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Temperature')
        
    fig.savefig(output_data_path + 'comparison_fitted.png', format=figure_settings["figure_format"], dpi=figure_settings["figure_dpi"])
    fig.show()

def postprocess(figure_settings:dict, case_list, dataset_input_file, dataset_output_file):

        
    weather_data = pd.read_hdf(dataset_input_file, key='data')
    temperatures = weather_data['Temperature'].to_numpy()
    wind_speeds = weather_data['Wind Speed'].to_numpy()
    sun_zeniths = weather_data['Sun Zenith'].to_numpy()
    sun_azimuths = weather_data['Sun Azimuth'].to_numpy()
    sun_elevations = weather_data['Sun Elevation'].to_numpy()
    with h5py.File(dataset_output_file, 'r') as f:
        ground_truth = f['Model output'][:]
    
    prior_model = DummyModel(1.0, 0.01, 0.0, 0.0, 0.0, 11, 11)
    prior_model.calculate_grid(temperatures, wind_speeds, sun_zeniths, sun_azimuths, sun_elevations)

    prior = prior_model.grid[5,0,:]

    for case in case_list.values():
        postprocess_case(case['case_name'], figure_settings, case['results_data_path'], case['output_data_path'], prior, ground_truth, biased=case['biased'])

    