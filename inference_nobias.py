# third party imports
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import arviz as az

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (problem solving)
from probeye.inference.scipy.solver import MaxLikelihoodSolver
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.dynesty.solver import DynestySolver

# local imports (inference data post-processing)
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot

from TestBridgeTemperatureDT.dummy_model import DummyModel
from TestBridgeTemperatureDT.utils import save_results_to_h5
import pandas as pd
import json
import h5py
from pathlib import Path

class DummyForwardModel(ForwardModelBase):
    def __init__(self, name: str, *args, **kwargs):

        super().__init__(name)
        self.args = args
        self.kwargs = kwargs
        self.model = DummyModel(0.0, 0.0, 0.0, 0.0, 0.0, 11, 11)
        self.data_path = Path(kwargs["parameters"]["data_path"])
        self.figures_path = Path(kwargs["parameters"]["figures_path"])
        self.file_name = kwargs["parameters"]["file_name"]

    def interface(self):
        self.parameters = ['T', 'WS', 'Zen', 'Az', 'Elev']
        self.input_sensors = [Sensor("temperatures"), Sensor("wind_speeds"), Sensor("sun_zeniths"), Sensor("sun_azimuths"), Sensor("sun_elevations") ]
        self.output_sensors = Sensor("y", std_model="sigma")

    def response(self, inp: dict) -> dict:
        temperatures = inp["temperatures"]
        wind_speeds = inp["wind_speeds"]
        sun_zeniths = inp["sun_zeniths"]
        sun_azimuths = inp["sun_azimuths"]
        sun_elevations = inp["sun_elevations"]
        input_dict = {"temp_const":0.0, "wind_const":0.0, "zenith_constant":0.0, "azimuth_constant":0.0, "elevation_constant":0.0}
        equivalence_map = {"T":"temp_const", "WS":"wind_const", "Zen":"zenith_constant", "Az":"azimuth_constant", "Elev":"elevation_constant"}
        for parameter in self.parameters:
            if parameter in inp.keys():
                input_dict[equivalence_map[parameter]] = inp[parameter]
        self.model.update_parameters(**input_dict)
        self.model.calculate_grid(temperatures, wind_speeds, sun_zeniths, sun_azimuths, sun_elevations)
        
        return {"y": self.model.grid[5,0,:].flatten()}


def inference_nobias(parameters_file, dataset_input_file, dataset_output_file, show=False, steps=1000, burn=100, 
              output_data_path="data/", output_figures_path="figures/", output_filename="no_bias"):
    ## Generate dataset for inference

    # Load the HDF5 file as a dataframe
    training_input = pd.read_hdf(dataset_input_file, key='data')
    temperatures = training_input['Temperature'].to_numpy()
    wind_speeds = training_input['Wind Speed'].to_numpy()
    sun_zeniths = training_input['Sun Zenith'].to_numpy()
    sun_azimuths = training_input['Sun Azimuth'].to_numpy()
    sun_elevations = training_input['Sun Elevation'].to_numpy()

    with h5py.File(dataset_output_file, 'r') as f:
        training_output = f['Model output'][:]


    with open(parameters_file) as f:
        sensitivity_analysis_results = json.load(f)

    parameters_dict = sensitivity_analysis_results["S1"].keys()

    # Initialize forward model
    parameters_forward_model = {"data_path": output_data_path, "figures_path": output_figures_path, "file_name": output_filename}
    model = DummyForwardModel("Dummy Forward Model", parameters = parameters_forward_model)

    # initialize the problem (the print_header=False is only set to avoid the printout of
    # the probeye header which is not helpful here)
    problem = InverseProblem("Dummy problem", print_header=False)

    # add the problem's parameters

    possible_parameters = ['T', 'WS', 'Zen', 'Az', 'Elev']

    for parameter in possible_parameters:
        if parameter in parameters_dict:
            problem.add_parameter(
                parameter,
                tex=parameter,
                info=f"Parameter {parameter}",
                prior=Normal(mean=0.0, std=1.0),
            )
        else:
            problem.add_parameter(
                parameter,
                tex=parameter,
                info=f"Parameter {parameter}",
                value=0.0,
            )

    problem.add_parameter(
        "sigma",
        tex=r"$\sigma$",
        info="Standard deviation, of zero-mean Gaussian noise model",
        value=2E-2,
    )

    input_dict = {"temperatures": temperatures, "wind_speeds": wind_speeds, "sun_zeniths": sun_zeniths, "sun_azimuths": sun_azimuths, "sun_elevations": sun_elevations,  "y": training_output.flatten()}
    # experimental data
    problem.add_experiment(
        name="TestSeries_1",
        sensor_data=input_dict,
    )

    # forward model
    problem.add_forward_model(model, experiments="TestSeries_1")

    # likelihood model
    problem.add_likelihood_model(
        GaussianLikelihoodModel(experiment_name="TestSeries_1", model_error="additive")
    )

    # print problem summary
    problem.info(print_header=True)

    # this is for using the emcee-solver (MCMC sampling)
    solver = EmceeSolver(problem, show_progress=True)
    inference_data = solver.run(n_steps=steps, n_initial_steps=burn)
    az.to_netcdf(inference_data, output_data_path + output_filename + "_inference.nc")

    # save the results to HDF5
    parameters_results_dict = {"parameters_mean_dict":{}, "parameters_sd_dict":{}, "parameters_median_dict":{}, "bias_mean_dict":None, "bias_sd_dict":None}
    for key_parameter, value_parameter in inference_data.posterior.items():
        parameters_results_dict["parameters_mean_dict"][key_parameter] =  np.mean(value_parameter).values
        parameters_results_dict["parameters_sd_dict"][key_parameter] =  np.std(value_parameter).values
        parameters_results_dict["parameters_median_dict"][key_parameter] =  np.median(value_parameter)

    save_results_to_h5(model, parameters_results_dict, input_dict=input_dict, sensors_flag=False)
        
#############################################################################
    # Compare the datasets

    true_values = {"T": 1.0, "WS":0.01}
    # this is an overview plot that allows to visualize correlations
    pair_plot_array = create_pair_plot(
        inference_data,
        solver.problem,
        focus_on_posterior=True,
        true_values = true_values,
        show_legends=True,
        title="Sampling results from emcee-Solver (pair plot)",
    )
    # figure = pair_plot_array.figure
    
    figure = pair_plot_array[0][0].figure
    figure.savefig(output_figures_path + output_filename + "_pair.png")

    # this is a posterior plot, without including priors
    post_plot_array = create_posterior_plot(
        inference_data,
        solver.problem,
        # title="Simple 1D-Case",
        round_to=3,
        show=show
    )
    # figure = post_plot_array.figure
    figure = post_plot_array[0].figure
    figure.savefig(output_figures_path + output_filename + "_posterior.png")

    # trace plots are used to check for "healthy" sampling
    trace_plot_array = create_trace_plot(
        inference_data,
        solver.problem,
        # title="Simple 1D-Case",
        show=show
    )
    # figure = trace_plot_array.ravel()[0].figure

    figure = trace_plot_array[0][0].figure
    figure.savefig(output_figures_path + output_filename + "_trace.png")

