import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os




def print_netcdf_summary(file_path, return_variables=None):

    print(return_variables)
    """Prints a summary and optionally returns data for specific variables from a NetCDF file."""
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    try:
        dataset = nc.Dataset(file_path, mode='r')
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    variable_data = {}

    print("Global Attributes:")
    for attr in dataset.ncattrs():
        print(f"  {attr}: {dataset.getncattr(attr)}")

    print("\nDimensions:")
    for dim, dim_obj in dataset.dimensions.items():
        print(f"  {dim}: length {dim_obj.size} (unlimited: {dim_obj.isunlimited()})")

    print("\nVariables:")
    for var in dataset.variables:
        print(f"Variable: {var}")
        print(f"  Type: {dataset.variables[var].dtype}")
        print(f"  Dimensions: {dataset.variables[var].dimensions}")
        print(f"  Shape: {dataset.variables[var].shape}")
        for attr in dataset.variables[var].ncattrs():
            print(f"    {attr}: {dataset.variables[var].getncattr(attr)}")
        if return_variables and var in return_variables:
            variable_data[var] = dataset.variables[var][:]

    if 'time' in dataset.variables:
        T_var = dataset.variables['time']
        T_units = T_var.units if 'units' in T_var.ncattrs() else 'No units available'
        T_data = T_var[:]
        print(f"\nT units: {T_units} Data: {T_data}")

    dataset.close()

    if return_variables:
        print('retorna!')
        return variable_data




def read_station(folder_path, file_name):
    
    full_path = os.path.join(folder_path, f'{file_name}.txt')
    try:
        # Load observation data from a text file
        # Specify -99999 and -1 as NaN values
        df = pd.read_table(full_path, delim_whitespace=True, na_values=[-99999, -1])

        df['date'] = pd.to_datetime(df['date'])

    except FileNotFoundError:
        print(f"No data: {full_path}")
        return pd.DataFrame()  # Return empty DataFrame if file is not found
    
    return df




def list_files_and_dates(directory_path, file_prefix):
    """Checks if the directory exists and returns a DataFrame containing filenames and their corresponding dates."""
    if os.path.exists(directory_path):
        all_files = os.listdir(directory_path)
        # Filter files that start with the specified prefix
        file_list = [file for file in all_files if file.startswith(file_prefix)]
        file_dates = [file.split('_')[-1].split('.')[0][:-2] for file in file_list]
        
        # Convert strings to datetime objects
        file_dates = [datetime.strptime(date, '%Y%m%d') for date in file_dates]
        
        # Create DataFrame with filenames and dates
        file_data = pd.DataFrame({
            'File Name': file_list,
            'Date': file_dates
        }).sort_values(by='Date').reset_index(drop=True)
        
        return file_data
    else:
        print(f"The directory '{directory_path}' does not exist.")
        return None
    


def post_process_memory(obs_ts, for_ts, scale_factor=2):
    # Ensure indices are in datetime format if they are not
    if not pd.api.types.is_datetime64_any_dtype(for_ts.index):
        for_ts.index = pd.to_datetime(for_ts.index)
    if not pd.api.types.is_datetime64_any_dtype(obs_ts.index):
        obs_ts.index = pd.to_datetime(obs_ts.index)

    # Calculate the forecast horizon based on the length of the forecast DataFrame
    horizon = len(for_ts)

    # Extract the index of the first row of the forecast DataFrame
    index_value = for_ts.index[0]

    # Extract the corresponding observed value using the index from the forecast DataFrame
    if index_value in obs_ts.index:
        obs_value = obs_ts.loc[index_value, 'discharge']
    else:
        raise KeyError(f"Index value {index_value} not found in observed data index.")

    # Extract the forecast value for the first index
    for_val = for_ts.loc[index_value, 'dis24_station']

    # Calculate the initial difference between the observed and forecasted values
    delta = obs_value - for_val

    # Initialize an empty DataFrame to store nudged forecasts
    nudged_forecasts = pd.DataFrame(index=for_ts.index, columns=for_ts.columns)

    # Iterate over each lead time (row index in the forecast DataFrame)
    for i in range(horizon):
        # Calculate the adjusted delta for the current timestep using a linear decay
        adjusted_delta = np.sign(delta) * np.maximum(0, abs(delta) * (1 - scale_factor * i / (horizon - 1)))
        
        # Adjust each forecast by adding the adjusted delta
        nudged_forecasts.iloc[i] = for_ts.iloc[i] + adjusted_delta

    # Return the nudged forecasts
    return nudged_forecasts



def post_process_memory_log(obs_ts, for_ts, decay_factor=0.5):
    # Ensure indices are in datetime format
    for_ts.index = pd.to_datetime(for_ts.index)
    obs_ts.index = pd.to_datetime(obs_ts.index)

    # Extract the forecast horizon
    horizon = len(for_ts)

    # Find the initial forecast and observation
    index_value = for_ts.index[0]
    if index_value in obs_ts.index:
        obs_value = obs_ts.loc[index_value, 'discharge']
    else:
        raise KeyError(f"Index value {index_value} not found in observed data index.")

    for_val = for_ts.loc[index_value, 'dis24_station']

    # Calculate the scaling factor
    if for_val != 0:
        scaling_factor = np.log(obs_value / for_val) * decay_factor
    else:
        scaling_factor = 0

    # Initialize an empty DataFrame to store adjusted forecasts
    adjusted_forecasts = pd.DataFrame(index=for_ts.index, columns=for_ts.columns)

    # Adjust each forecast using the scaling factor
    for i in range(horizon):
        # Calculate decayed scaling for current time step
        decayed_scaling = np.exp(scaling_factor * (1 - i / (horizon - 1)))

        # Apply the scaling to adjust forecasts proportionally
        adjusted_forecasts.iloc[i] = for_ts.iloc[i] * decayed_scaling

    return adjusted_forecasts



def post_process_logarithmic(obs_ts, for_ts):
    # Ensure indices are in datetime format
    for_ts.index = pd.to_datetime(for_ts.index)
    obs_ts.index = pd.to_datetime(obs_ts.index)

    # Extract the first index of the forecast DataFrame
    index_value = for_ts.index[0]
    
    # Check if the initial observed value is available
    if index_value in obs_ts.index:
        obs_value = obs_ts.loc[index_value, 'discharge']
    else:
        raise KeyError(f"Index value {index_value} not found in observed data index.")

    # Get the initial forecast value
    for_val = for_ts.loc[index_value, 'dis24_station']

    # Calculate the scaling factor as the logarithm of the ratio of observed to forecasted values
    if for_val != 0:
        scaling_factor = np.log(obs_value / for_val)
    else:
        scaling_factor = 0

    # Initialize an empty DataFrame to store adjusted forecasts
    adjusted_forecasts = pd.DataFrame(index=for_ts.index, columns=for_ts.columns)

    # Apply the constant scaling to adjust forecasts uniformly
    adjusted_forecasts = for_ts.apply(lambda x: x * np.exp(scaling_factor))

    return adjusted_forecasts


def post_process_constant(obs_ts, for_ts, var_name):
    # Calculate the forecast horizon based on the length of the forecast DataFrame
    horizon = len(for_ts)

    # Extract the index of the first row of the forecast DataFrame
    index_value = for_ts.index[0]

    # Extract the corresponding observed value using the index from the forecast DataFrame
    obs_value = obs_ts.loc[index_value, 'discharge']

    # Extract the forecast value for the first index
    for_val = for_ts.loc[index_value, var_name]

    # Calculate the difference between the observed and forecasted values
    delta = obs_value - for_val

    # Initialize an empty DataFrame to store nudged forecasts
    postprocessed_forecasts = pd.DataFrame(index=for_ts.index, columns=for_ts.columns)

    # Iterate over each lead time (row index in the forecast DataFrame)
    for i in range(horizon):
        # Adjust each forecast by adding the delta
        postprocessed_forecasts.iloc[i] = for_ts.iloc[i] + delta

    # Optionally, you can return the nudged forecasts
    return postprocessed_forecasts



def read_dataset(processed_data, issued_date, dataset_type):
    """
    General function to process different datasets based on their type.
    Handles masked data where applicable, and applies date filters for Google data.

    Parameters:
    - processed_data: dict, keys are dates, values are dataset-specific dictionaries containing 'time' and 'dis24_station'.
    - issued_date: datetime, the date against which to normalize and check.
    - dataset_type: str, type of the dataset (e.g., 'HYPE', 'GEO', 'Google', 'IPH MGB', 'IPH HEC', 'NASA').

    Returns:
    - pd.DataFrame with 'dis24_station' or appropriate variable name as column, indexed by datetime.
    """
    key = next((date for date in processed_data.keys() if date.normalize() == issued_date), None)
    if key:
        data = processed_data[key]
        time = pd.to_datetime(data['time'])

        if isinstance(data['dis24_station'], np.ma.MaskedArray):
            # Specifically handle numpy masked arrays
            data_array = np.where(data['dis24_station'].mask, np.nan, data['dis24_station'].data)
        elif isinstance(data['dis24_station'], (np.ndarray, list)):
            # Handle numpy arrays and lists
            data_array = np.array(data['dis24_station'])
        else:
            # Safe conversion to numpy array for pandas Series or similar types
            data_array = data['dis24_station'].to_numpy()

        # Create DataFrame
        data_frame = pd.DataFrame(data_array, columns=['dis24_station'], index=time)

        if dataset_type == 'Google':
            # Google data needs additional filtering by issue date
            valid_indices = data_frame.index > issued_date
            data_frame = data_frame.loc[valid_indices]

        # Adjust column name for NASA data
        # column_name = 'streamflow' if dataset_type == 'NASA' else 'dis24_station'
        column_name = 'dis24_station'
        data_frame.columns = [column_name]  # Renaming the column after filtering

        return data_frame

    return pd.DataFrame()  # Return an empty DataFrame if no data is found



def read_ensemble(processed_data, issued_date, dataset_type):
    """
    General function to process different datasets based on their type.
    Handles masked data where applicable, and applies date filters for Google data.

    Parameters:
    - processed_data: dict, keys are dates, values are dataset-specific dictionaries containing 'time' and 'dis24_ensem'.
    - issued_date: datetime, the date against which to normalize and check.
    - dataset_type: str, type of the dataset (e.g., 'HYPE', 'GEO', 'Google', 'IPH MGB', 'IPH HEC', 'NASA').

    Returns:
    - pd.DataFrame with 'dis24_ensem' or appropriate variable name as column, indexed by datetime.
    """
    key = next((date for date in processed_data.keys() if date.normalize() == issued_date), None)
    if key:
        data = processed_data[key]
        time = pd.to_datetime(data['time'])

        if isinstance(data['dis24_ensem'], np.ma.MaskedArray):
            # Specifically handle numpy masked arrays
            data_array = np.where(data['dis24_ensem'].mask, np.nan, data['dis24_ensem'].data)
        elif isinstance(data['dis24_ensem'], (np.ndarray, list)):
            # Handle numpy arrays and lists
            data_array = np.array(data['dis24_ensem'])
        else:
            # Safe conversion to numpy array for pandas Series or similar types
            data_array = data['dis24_ensem'].to_numpy()

        # Create DataFrame
        data_frame = pd.DataFrame(data_array, index=time)

        # return data_frame
        return data_frame 
    

    return pd.DataFrame()  # Return an empty DataFrame if no data is found

