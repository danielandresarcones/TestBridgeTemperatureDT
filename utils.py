from pathlib import Path
import h5py

def save_data_from_sensors_to_h5(sensors: dict, filename: str, bias = None) -> None:
    
    with h5py.File(filename, "w") as f:
        for (sensor_name, measurement), i in zip(sensors.items(), range(len(sensors))):
            sensor_group = f.create_group(sensor_name)
            sensor_group.create_dataset("Measurements", data=measurement.get_data_list().magnitude)
            sensor_group.create_dataset("Coordinates", data=measurement.where)
            if bias is not None:
                sensor_group.create_dataset("Bias", data=bias[i])

def save_data_from_dict_to_h5(array_dict: dict, filename: str, bias: dict = None) -> None:
    
    with h5py.File(filename, "w") as f:
        for key, value in array_dict.items():
            sensor_group = f.create_group(key)
            sensor_group.create_dataset("Measurements", data=value)
            if bias is not None:
                sensor_group.create_dataset("Bias", data=bias[key])

def save_results_to_h5(model, paramaters_results_dict: dict, input_dict: dict = None, sensors_flag: bool =False ) -> None:
    
    parameters_dict_mean = paramaters_results_dict["parameters_mean_dict"]
    parameters_dict_sd = paramaters_results_dict["parameters_sd_dict"]
    if paramaters_results_dict["bias_mean_dict"] is not None:
        bias_flag = True
        bias_dict_mean = paramaters_results_dict["bias_mean_dict"]
        bias_dict_sd = paramaters_results_dict["bias_sd_dict"]
    else:
        bias_flag = False

    # TODO: It is bad practice to suppose that model has data_path and file_name attributes
    # Mean dataset
    model_input_dict = {**input_dict, **parameters_dict_mean}
    example_model_samples = model.response(model_input_dict)
    if not bias_flag:
        bias_dict_mean = None
    if sensors_flag:
        save_data_from_sensors_to_h5(model.problem.sensors, model.data_path / (model.file_name + "_sensors.h5"), bias = bias_dict_mean)
        model.problem.clean_sensor_data()
    else:
        save_data_from_dict_to_h5(example_model_samples, model.data_path / (model.file_name + "_samples.h5"), bias = bias_dict_mean)

    # Plus std dataset
    parameters_dict_plus = {key: parameters_dict_mean[key] + parameters_dict_sd[key] for key in parameters_dict_mean}
    model_input_dict.update(parameters_dict_plus)
    if bias_flag:
        bias_dict_plus = {key: bias_dict_mean[key] + bias_dict_sd[key] for key in bias_dict_mean}
    else:
        bias_dict_plus = None
    example_model_samples_stdp = model.response(model_input_dict)
    if sensors_flag:
        save_data_from_sensors_to_h5(model.problem.sensors, model.data_path / (model.file_name + "_sensors_plus_std.h5"), bias = bias_dict_plus)
        model.problem.clean_sensor_data()
    else:
        save_data_from_dict_to_h5(example_model_samples_stdp, model.data_path / (model.file_name + "_samples_plus_std.h5"), bias = bias_dict_plus)

    # Minus std dataset
    parameters_dict_minus = {key: parameters_dict_mean[key] - parameters_dict_sd[key] for key in parameters_dict_mean}
    model_input_dict.update(parameters_dict_minus)
    if bias_flag:
        bias_dict_minus = {key: bias_dict_mean[key] - bias_dict_sd[key] for key in bias_dict_mean}
    else:
        bias_dict_minus = None
    example_model_samples_stdn = model.response(model_input_dict)
    if sensors_flag:
        save_data_from_sensors_to_h5(model.problem.sensors, model.data_path / (model.file_name + "_sensors_minus_std.h5"), bias = bias_dict_minus)
        model.problem.clean_sensor_data()
    else:
        save_data_from_dict_to_h5(example_model_samples_stdn, model.data_path / (model.file_name + "_samples_minus_std.h5"), bias = bias_dict_minus)

def load_dataset(filename: Path, flatten_data: bool = True, biased: bool = False):
     
    with h5py.File(filename, 'r') as f:
        data = {}
        bias = {}
        for key, value in f.items():
            data[key] = value["Measurements"][...]
            if biased:
                bias[key] = value["Bias"][...]

    if flatten_data:
        flattened_data=[value for value in data.values()]
        if biased:
            flattened_bias=[value for value in bias.values()]
            return flattened_data,flattened_bias
        else:
            return flattened_data, None
    else:
        if biased:
            return data, bias
        else:
            return data, None
        

def scale_coordinates(arr):
    """
    Scales an n-dimensional numpy array of coordinates to the range [0, 1] 
    based on the original min and max values present in the array.
    Also works with 1D arrays.
    """
    
    # Check if the input is a 1D array
    is_1d = len(arr.shape) == 1
    if is_1d:
        arr = arr.reshape(-1, 1)
    
    # Iterate over dimensions and scale each dimension
    for dim in range(arr.shape[1]):
        min_val, max_val = arr[:, dim].min(), arr[:, dim].max()
        arr[:, dim] = (arr[:, dim] - min_val) / (max_val - min_val)
    
    # If it was 1D, return it in its original shape
    if is_1d:
        arr = arr.ravel()
    
    return arr