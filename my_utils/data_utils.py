import pandas as pd
import numpy as np
import scipy.stats as stats # For normal distribution sampling
import torch
import matplotlib.pyplot as plt
from .helper_functions import print_dict
from .normalisation_utils import normalise_gaussian, normalise_minmax
import random

def preprocess_data(params):
    '''
    Function to preprocess data for the experimental dataset.

    input:
        params: dict
        additional_cleaning: bool (default=False)
        stress_threshold: float (default=float('inf'))
        remove_outlier_curves: bool (default=False)
        
    '''
    
    # Extract parameters from params
    data_path = params['data_path']
    param_name = params['param_name']
    input_labels = params['input_labels']
    output_labels = params['output_labels']
    additional_cleaning = params['additional_cleaning']
    stress_threshold = params['stress_threshold']
    remove_outlier_curves = params['remove_outlier_curves']

    # Read csv file into Pandas DataFrame
    data_df = pd.read_csv(data_path) # type: ignore
    # Rename cols
    data_df = data_df.rename(columns={'Stress': 'EngStress'})
    data_df = data_df.rename(columns={'Strain': 'EngStrain'})

    if additional_cleaning:
        # Remove all stresses above the threshold
        data_df = data_df[data_df['EngStress'] < stress_threshold]
        # Remove all strains in TestID = 1 above 0.55
        data_df = data_df[~((data_df['TestID'] == 1) & (data_df['EngStrain'] > 0.55))]

    if remove_outlier_curves:
        # Remove short curves
        data_df = data_df[~((data_df['TestID'] == 4) & (data_df['RepeatID'] == 3))]
        data_df = data_df[~((data_df['TestID'] == 4) & (data_df['RepeatID'] == 4))]

        data_df = data_df[~((data_df['TestID'] == 5) & (data_df['RepeatID'] == 2))]
        data_df = data_df[~((data_df['TestID'] == 5) & (data_df['RepeatID'] == 3))]

    # Covert Void_Fraction column from 'float%' string to float
    data_df['Void_Fraction'] = data_df['Void_Fraction'].str.rstrip('%').astype(float)

    # Set all input and output labels to floats (32-bit)
    for col_name in input_labels:
        data_df[col_name] = data_df[col_name].astype(np.float32)
    for col_name in output_labels:
        data_df[col_name] = data_df[col_name].astype(np.float32)
    
    # Drop columns that aren't needed
    data_df = data_df[input_labels + output_labels + ['TestID', 'RepeatID']]

    # Get number of unique TestIDs
    num_tests = data_df['TestID'].nunique()

    # Get min and max number of repeats
    repeat_ids = []
    for _, group in data_df.groupby('TestID'):
        repeat_ids.append(max(group['RepeatID'].values))
    num_repeats_tup = (min(repeat_ids), max(repeat_ids))

    # Print information
    print("Varying parameter:", param_name)
    print("Number of unique tests:", num_tests)
    print("Min number of repeats:", num_repeats_tup[0])
    print("Max number of repeats:", num_repeats_tup[1])
    print("Input labels:", input_labels)
    print("Output labels:", output_labels)

    return data_df, num_tests


def generate_x_vals_for_bins(
    x_values,
    location,
    spread,
    density,
    curve_length_before_binning=None,
    strain_delta=None,
    verbose=False,
    plot=False,
    ax=None
):
    """
    Generate x-values from x_min to x_max where:
      - If x_current is within [location - spread, location + spread],
        then sample Δx from a normal distribution with std=spread (to get smaller steps).
      - Otherwise, sample Δx from a uniform distribution [uniform_low, uniform_high].

    Result: higher density (more points) near 'location' 
            and sparser coverage outside that region.
    """
    x_min = x_values.min()
    x_max = x_values.max()

    if density >= 1 or density < 0:
        raise ValueError("density must be between [0, 1)")
    if location < x_min or location > x_max:
        raise ValueError("location must be within the range [x_min, x_max]")
    if not (isinstance(strain_delta, float) or isinstance(curve_length_before_binning, int)):
        error_msg = "strain_delta must be provided as float and curve_length_before_binning must be provided as int\n" + \
            "strain_delta type: " + str(type(strain_delta)) + "\n" + \
            "curve_length_before_binning type: " + str(type(curve_length_before_binning))
        raise ValueError(error_msg)
    if curve_length_before_binning is not None and strain_delta is not None:
        raise ValueError("only one of curve_length_before_binning or strain_delta must be provided")
    if strain_delta is not None and strain_delta < np.diff(x_values).min():
        raise ValueError("strain_delta must be greater than the minimum strain increment")

    
    # Calulate strain_delta
    if strain_delta is None:
        strain_delta = (x_max - x_min) / curve_length_before_binning

    # Normalise spread
    spread_norm = spread * (x_max - x_min)
    # Create a normal distribution object centered at location with scale=spread_norm
    normal_dist = stats.norm(loc=location, scale=spread_norm)

    # Create a function that scales the PDF values
    def sample_diff_from_composite_pdf(normal_dist, strain_delta, x):
        max_pdf = normal_dist.pdf(location)
        scaled_x_from_pdf = normal_dist.pdf(x) / max_pdf * strain_delta
        return strain_delta - scaled_x_from_pdf * density

    x_vals = [x_min]
    
    while x_vals[-1] < x_max:
        x_current = x_vals[-1]

        # Sample from normal distribution at x_current
        delta_x = sample_diff_from_composite_pdf(normal_dist, strain_delta, x_current)
        # Ensure we step forward
        delta_x = abs(delta_x)

        x_new = x_current + delta_x

        # Stop if we exceed x_max
        if x_new > x_max:
            break

        x_vals.append(x_new)

    if plot and ax is None:
        raise ValueError('ax must be provided if plot is True')
    if plot and ax is not None:

        
        # Plot uniform density
        blue = '#1F77B4'
        uniform_density = np.ones_like(x_vals[1:]) / strain_delta  # Inverse of uniform delta
        uniform_density_norm = uniform_density / max(uniform_density)  # Normalize to same scale
        ax.plot(x_vals[1:], uniform_density_norm, 
                color=blue,
                linestyle=':',
                linewidth=0.75,
                label='Uniform density')
        
        # Calculate point density
        x_vals_density = np.diff(x_vals)  # Get intervals
        x_vals_density = 1/x_vals_density  # Convert to density
        x_vals_density_norm = x_vals_density / max(uniform_density)  # Normalize
        
        # Plot non-uniform density
        # ax.scatter(x_vals[1:], x_vals_density_norm, 
        #           alpha=0.05,
        #           color='b', 
        #           s=1,
        #           label='Point density')         
        ax.plot(x_vals[1:], x_vals_density_norm, 
                color=blue,
                linestyle='--',
                linewidth=0.75,
                label='Gaussian density') 
        
        # ax.set_ylabel(r'$\frac{\mathrm{Actual\ Density}}{\mathrm{Uniform\ Density}}$', fontsize=14)
        ax.set_ylabel('Relative point density')
        ax.set_ylim(0.9, max(x_vals_density_norm) * 1.5)

        return np.array(x_vals), ax, x_vals[1:], uniform_density_norm, x_vals_density_norm
    if verbose:
        print(f"Total number of points: {len(x_vals)}")

    return np.array(x_vals)


def interpolate_curves(data_df, params):
    """
    Interpolates curves for each test in the dataframe with non-uniform binning.
    Uses 'curve_len' equidistant points between x_min_uniform and x_max_uniform.
    Outside this range, the number of bins increases linearly with the x value.
    
    Parameters:
        data_df: DataFrame containing the curves
        binning_params: Dictionary containing binning parameters

    Returns:
        tuple: (interpolated DataFrame, total number of points used for interpolation)
    """
    # Extract parameters
    binning_params = params['binning_params']
    probabilistic_output = params['probabilistic_output']
    average_all_curves = params['average_all_curves']

    interpolated_frames = []

    # Group by test ID (or another column if desired)
    for _, group_test in data_df.groupby('TestID'):

        for gid, group_repeat in group_test.groupby('RepeatID'):
            # Sort to ensure 'x_col' is monotonic
            group_repeat = group_repeat.sort_values(by='EngStrain')

            # Get non-uniform bin values
            x_values = group_repeat['EngStrain'].values
            x_new = generate_x_vals_for_bins(x_values, **binning_params, plot=False, ax=None) # type: ignore

            # Interpolate y values
            df_dict = {'RepeatID': gid, 'EngStrain': x_new}
            if probabilistic_output:
                # Interpolate error and mean values
                y_error = group_repeat['EngStress_error'].values
                y_mean = group_repeat['EngStress_mean'].values
                y_error_new = np.interp(x_new, x_values, y_error)
                y_mean_new = np.interp(x_new, x_values, y_mean)

                df_dict['EngStress_error'] = y_error_new
                df_dict['EngStress_mean'] = y_mean_new
            elif average_all_curves:
                # Interpolate mean stress values
                y_values = group_repeat['EngStress_mean'].values
                y_new = np.interp(x_new, x_values, y_values)
                df_dict['EngStress_mean'] = y_new
            else:
                # Interpolate stress values
                y_values = group_repeat['EngStress'].values
                y_new = np.interp(x_new, x_values, y_values)
                df_dict['EngStress'] = y_new

            # Build a new DataFrame with interpolated values
            interp_df = pd.DataFrame(df_dict)

            # Include any additional columns (e.g., parameters) by taking their mean or first value
            # across the group – adjust as needed
            for col in group_repeat.columns:
                if col not in ['RepeatID', 'EngStrain', 'EngStress', 'EngStress_error', 'EngStress_mean']:
                    interp_df[col] = group_repeat[col].iloc[0]

            interpolated_frames.append(interp_df)

    # Combine all interpolated DataFrames
    df_interp = pd.concat(interpolated_frames, ignore_index=True)

    return df_interp


# def get_mean_and_error_columns_v2(df, params, only_average=False):
#     '''
#     Function to average test curves and compute error bars.
#     Returns a DataFrame with averaged curves and error bars.
#     '''


#     def crop_to_shortest_curve(df):
#         '''
#         Function to crop each RepeatID with a TestID to the shortest curve.
#         '''
#         df_frames = []
#         # Crop each RepeatID to the minimum strain value
#         for _, group_test in df.groupby('TestID'):
#             # Get the maximum strain value of the shortest curve
#             max_strain = group_test.groupby('RepeatID')['EngStrain'].max().min()
#             for _, group_repeat in group_test.groupby('RepeatID'):
#                 # Crop to the maximum strain value
#                 df_frames.append(group_repeat[group_repeat['EngStrain'] <= max_strain])
        
#         return pd.concat(df_frames, ignore_index=True)

#     # Extract parameters
#     remove_discontinuities = params['remove_discontinuities']
#     window = params['window']
#     additional_labels = params['additional_labels']

#     # Define output labels
#     output_label_mean = 'EngStress_mean'
#     output_label_error = 'EngStress_error'

#     # Remove discontinuities
#     if remove_discontinuities:
#         df = crop_to_shortest_curve(df)

#     # Compute mean and error bars
#     # Sort df so that the rolling is applied in ascending order of RepeatID
#     df = df.sort_values(["TestID", "RepeatID"])

#     # Use transform() with a custom function that does a .rolling() on each group’s val1
#     df[output_label_mean] = (
#         df.groupby("TestID")["EngStress"]
#         .transform(lambda s: s.rolling(window=window, min_periods=1, center=True).mean())
#     )
#     return df


def get_mean_and_error_columns(df, params, only_average=False):
    '''
    Function to average test curves and compute error bars.
    Returns a DataFrame with averaged curves and error bars.
    '''
    # Extract parameters
    remove_discontinuities = params['remove_discontinuities']
    additional_labels = params['additional_labels']
    window = params['window']
        
    data_df_test_frames = []
    output_label_mean = 'EngStress_mean'
    output_label_error = 'EngStress_error'
    
    for gid, group_test in df.groupby('TestID'):
        
        # Get a common set of strain bins for all RepeatIDs in this TestID
        curve_lengths, _, _ = get_curve_lengths(group_test)
        group_curve_len = max(curve_lengths)
        group_test['EngStrainBins'] = pd.cut(group_test['EngStrain'], bins=group_curve_len, labels=False)

        if remove_discontinuities:
            # To prevent discontinuities, only keep the EngStrainBins that exist in all repeats
            min_max_bin = group_curve_len
            for _, g in group_test.groupby('RepeatID'):
                max_bin_in_repeat = g['EngStrainBins'].max()
                if max_bin_in_repeat < min_max_bin:
                    min_max_bin = max_bin_in_repeat
            # Remove all EngStrainBins above the maximum
            group_test = group_test[group_test['EngStrainBins'] <= min_max_bin]

        # Drop duplicate EngStrainBins for each repeat
        group_test = group_test.groupby('RepeatID', group_keys=False).apply(lambda g: g.drop_duplicates(subset=['EngStrainBins'])).reset_index(drop=True)
        # Pivot the data so each RepeatID's stresses become separate columns
        df_pivot = group_test.pivot(index='EngStrainBins', columns='RepeatID', values='EngStress')

        # Smooth the data
        df_pivot = df_pivot.rolling(window=window, min_periods=1, center=True).mean()

        # Compute the mean and error bar
        # error_col = df_pivot.std(axis=1)
        error_col = df_pivot.max(axis=1) - df_pivot.min(axis=1)
        mean_col = df_pivot.mean(axis=1)
        
        # Add the mean and error bar to the pivot
        df_pivot[output_label_error] = error_col
        df_pivot[output_label_mean] = mean_col

        # Merge the average stresses back into the original DataFrame
        group_test = group_test.merge(df_pivot[[output_label_mean, output_label_error]], on='EngStrainBins', how='left')

        # Take average of additional labels across repeats
        for label in additional_labels:
            group_test[label] = group_test[label].mean()

        # Remove duplicate EngStrainBins from TestID (it doesn't matter which repeat we keep, because we've averaged across them)
        group_test = group_test.drop_duplicates(subset=['EngStrainBins'])
        # Set repeat ID to 0 for all rows (because we've averaged across them)
        group_test['RepeatID'] = 0
        # Drop EngStress and EngStrainBins columns
        group_test = group_test.drop(columns=['EngStrainBins'])

        # Append to list
        data_df_test_frames.append(group_test)

    # Combine all interpolated DataFrames
    df_new = pd.concat(data_df_test_frames, ignore_index=True)

    if only_average:
        # Drop EngStress_error
        df_new = df_new.drop(columns=['EngStress', output_label_error])
        output_labels = [output_label_mean]
    else:
        # Drop EngStress
        df_new = df_new.drop(columns=['EngStress'])
        output_labels = [output_label_mean, output_label_error]

    return df_new, output_labels
    

def get_curve_lengths(df_train, df_test=None):
    '''
    Function to get the curve lengths across all RepeatIDs.
    '''
    curve_lengths_dict = {}
    curve_lengths_dict_with_repeats = {}
    curve_lengths_all = []
    for gid, group in df_train.groupby('TestID'):
        for rid, g in group.groupby('RepeatID'):
            curve_lengths_dict[gid] = g.shape[0]
            curve_lengths_dict_with_repeats[gid, rid] = g.shape[0]
            curve_lengths_all.append(g.shape[0])

    if df_test is not None:
        for gid, group in df_test.groupby('TestID'):
            for rid, g in group.groupby('RepeatID'):
                curve_lengths_dict[gid] = g.shape[0]
                curve_lengths_dict_with_repeats[gid, rid] = g.shape[0]
                curve_lengths_all.append(g.shape[0])

    return curve_lengths_all, curve_lengths_dict, curve_lengths_dict_with_repeats


def crop_to_min_curve_len(df_train, df_test):
        '''
        Crops each RepeatID group to have the same number of rows (the minimum number across all groups).
        '''
        # Find the minimum number of rows across all RepeatIDs
        curve_lengths, _, _ = get_curve_lengths(df_train, df_test)

        if len(set(curve_lengths)) != 1:
            print(f'Repeat tests have different numbers of points after interpolation: {max(curve_lengths)}, {min(curve_lengths)}. '
                f'Cropping all repeats to the shortest length {min(curve_lengths)}...')
    
            # Get the new curve length
            new_curve_len = min(curve_lengths)

            # Crop each RepeatID group to the minimum number of rows
            cropped_frames_train = []
            cropped_frames_test = []
            for _, group in df_train.groupby('TestID'):
                for _, g in group.groupby('RepeatID'):
                    cropped_frames_train.append(g.iloc[:new_curve_len])
            for _, group in df_test.groupby('TestID'):
                for _, g in group.groupby('RepeatID'):
                    cropped_frames_test.append(g.iloc[:new_curve_len])
            
            df_train = pd.concat(cropped_frames_train, ignore_index=True)
            df_test = pd.concat(cropped_frames_test, ignore_index=True)

            # Combine all cropped DataFrames
            return df_train, df_test, new_curve_len
        else:
            # Don't crop anything
            return df_train, df_test, curve_lengths[0]


def add_derivative_features(
    data_df,
    group_cols = ['TestID', 'RepeatID'],
    ):
    """
    Computes the derivative d(Stress)/d(Strain) within each (TestID, RepeatID) group.
    Returns a new dataframe with the column 'dEngStress_dEngStrain' added.

    - If 'EngStress_mean' is present (from averaging/probabilistic mode),
      it uses that for the derivative instead of 'EngStress'.
    - The first row of each group’s derivative is NaN, so we fill it with 0.
    """

    # Decide which stress column to differentiate (e.g. EngStress_mean if it exists)
    stress_col = 'EngStress_mean' if 'EngStress_mean' in data_df.columns else 'EngStress'

    def compute_deriv(group):
        group = group.sort_values('EngStrain')
        group['dEngStress_dEngStrain'] = group[stress_col].diff() / group['EngStrain'].diff()

        # Fill the first NaN derivative by carrying the next value backward (or could fill with 0)
        group['dEngStress_dEngStrain'] = group['dEngStress_dEngStrain'].fillna(0)

        # Smooth the derivative
        group['dEngStress_dEngStrain'] = group['dEngStress_dEngStrain'].rolling(window=50, min_periods=1, center=False).mean()
        return group

    # Group so you don’t “diff across” different repeats
    data_df = data_df.groupby(group_cols, group_keys=False).apply(compute_deriv)
    return data_df


def train_test_split(data_df, params, warnings_dict):
    '''
    Function to split data into training and test sets.

    Parameters:
     - data_df (pd.DataFrame): Data to split 
     - param_name (str): Name of the column to split data on
     - testIDs_of_test_data (list): List of TestIDs of test data
     - input_labels (list): List of input labels
     - output_labels (list): List of output labels
     - normalisation (str): Normalisation method to use
     - binning_params (dict): Dictionary containing binning parameters
     - remove_discontinuities (bool): Whether to remove discontinuities from the test curves (default=False)
     - window (int): Window size for averaging (default=None, =False to disable)
     - all_curves_same_length (bool): Whether all curves have the same length (default=True)
     - probabilistic_output (bool): Whether to output probabilistic curves (default=False)
     - average_all_curves (bool): Whether to average all curves (default=False)
    '''

    # Extract parameters from params
    param_name = params['param_name']
    testIDs_of_test_data = params['testIDs_of_test_data']
    input_labels = params['input_labels']
    output_labels = params['output_labels']
    binning_params = params['binning_params']
    window = params['window']
    all_curves_same_length = params['all_curves_same_length']
    probabilistic_output = params['probabilistic_output']
    average_all_curves = params['average_all_curves']

    # Define all labels
    all_labels = [*input_labels, *output_labels]
    
    # Get test data from testID_of_test_data
    if not any(x in data_df['TestID'].unique() for x in testIDs_of_test_data):
        raise ValueError('One of the specified TestIDs not in dataset')
    data_df_train = data_df[~data_df['TestID'].isin(testIDs_of_test_data)].copy()
    data_df_test = data_df[data_df['TestID'].isin(testIDs_of_test_data)].copy()
    # Get testIDs of train data
    params['testIDs_of_train_data'] = data_df_train['TestID'].unique()

    # Smooth the data
    data_df_train = data_df_train.groupby(['TestID', 'RepeatID'], group_keys=False).apply(lambda x: x.rolling(window=window, min_periods=1, center=True).mean())
    data_df_test = data_df_test.groupby(['TestID', 'RepeatID'], group_keys=False).apply(lambda x: x.rolling(window=window, min_periods=1, center=True).mean())

    # Averaging options
    if probabilistic_output and average_all_curves:
        raise ValueError('Invalid combination of probabilistic_output and average_all_curves')
    elif probabilistic_output:
        # Get mean and error columns
        data_df_train, output_labels = get_mean_and_error_columns(data_df_train, params)
        data_df_test, _ = get_mean_and_error_columns(data_df_test, params)
        # Redefine all labels
        all_labels = [*input_labels, *output_labels]
    elif average_all_curves:
        # Get mean and error columns
        data_df_train, output_labels = get_mean_and_error_columns(data_df_train, params, only_average=True)
        data_df_test, _ = get_mean_and_error_columns(data_df_test, params, only_average=True)
        # Redefine all labels
        all_labels = [*input_labels, *output_labels]

    # Interpolate the curves
    data_df_train = interpolate_curves(data_df_train, params)
    data_df_test = interpolate_curves(data_df_test, params)
    
    # Add derivative features
    if params['use_derivative_features']:
        data_df_train = add_derivative_features(data_df_train)
        data_df_test = add_derivative_features(data_df_test)

        # Make sure our derivative column is included in the input features
        if 'dEngStress_dEngStrain' not in input_labels:
            input_labels.append('dEngStress_dEngStrain')
            all_labels = [*input_labels, *output_labels]

    if all_curves_same_length:
        # Crop data_df_train and data_df_test to have the same number of rows per RepeatID    
        data_df_train, data_df_test, new_curve_len = crop_to_min_curve_len(data_df_train, data_df_test)
        print("Cropped to new curve length:", new_curve_len)

    # Reorder columns
    data_df_train = data_df_train[['TestID', 'RepeatID'] + all_labels]
    data_df_test = data_df_test[['TestID', 'RepeatID'] + all_labels]

    # Get the test parameter value
    test_param_vals = data_df_test[param_name].unique()
    train_param_vals = data_df_train[param_name].unique()

    # Get final curve lengths
    curve_lengths, _, _ = get_curve_lengths(data_df_test, data_df_train)

    # Print information
    print('all curves same length:', all_curves_same_length)
    print('Curve length range:', min(curve_lengths), max(curve_lengths))
    curve_length_diff = max(curve_lengths)/min(curve_lengths)
    if curve_length_diff > 1.2:
        warning = f'Curve length difference is greater than 1.2: curve_length_diff={curve_length_diff:.3f}'
        print('========================================')
        print(f'WARNING: {warning}')
        print('========================================')
        warnings_dict['data_utils:curve_length_diff'] = warning
    print("testIDs_of_test_data:", testIDs_of_test_data)
    print("all_labels:", all_labels)
    print("probabilistic_output:", probabilistic_output)
    print("average_all_curves:", average_all_curves)
    print("Binning parameters:")
    print_dict(binning_params, indent=4)

    # Update params
    params['test_param_vals'] = test_param_vals
    params['train_param_vals'] = train_param_vals
    params['curve_lengths'] = curve_lengths
    params['output_labels'] = output_labels
    params['input_labels'] = input_labels
    params['all_labels'] = all_labels

    return data_df_train, data_df_test, params, warnings_dict


def normalise_data(
        data_df_train, 
        data_df_test, 
        params,
        do_not_normalise=False, # required because data from this function is required later on
        denormalise=False, # denormalisation param
        data_df_results=None # denormalisation param
    ):
    '''
    Function to normalise data using min-max normalisation.

    Parameters:
     - data_df_train (pd.DataFrame): Training data
     - data_df_test (pd.DataFrame): Test data
     - params (dict): Dictionary of parameters
     - do_not_normalise (bool): Whether to not normalise the data
     - denormalise (bool): Whether to revert the normalisation
     - data_df_results (pd.DataFrame): Data to revert normalisation to
     - output_labels (list): List of output labels
    '''

    # Extract parameters
    input_labels = params['input_labels']
    output_labels = params['output_labels']
    probabilistic_output = params['probabilistic_output']
    average_all_curves = params['average_all_curves']

    # Checks
    if denormalise and data_df_results is None:
        raise ValueError('data_df_results must be provided if denormalise is True')
    
    # Copy dataframes
    data_df_train = data_df_train.copy()
    data_df_test = data_df_test.copy()

    # Define all labels
    output_labels_pred = [col + '_pred' for col in output_labels]
    all_labels = input_labels + output_labels
    if denormalise:
        all_labels = all_labels + output_labels_pred
    
    # Define custom numerator-denominator pairs
    custom_denoms = {}
    main_scaler_label = 'EngStress' # Default denominator
    if average_all_curves or probabilistic_output:
        main_scaler_label = 'EngStress_mean' # use EngStress_mean as denominator
    if probabilistic_output:
        custom_denoms['EngStress_error'] = main_scaler_label # Error is always denormalised by main_scaler_label
    if denormalise: # All predicted outputs are denormalised by main_scaler_label
        for pred_label in output_labels_pred:
            custom_denoms[pred_label] = main_scaler_label            


    # Define all denominators
    ignore_labels = custom_denoms.keys()
    normalise_labels = [col for col in all_labels if col not in ignore_labels]
    normalise_denoms = {col: col for col in normalise_labels}
    all_denoms = {**custom_denoms, **normalise_denoms}

    if not denormalise:
        # Check that all labels are present in both dataframes
        for col_name in normalise_labels:
            if col_name not in data_df_train.columns:
                raise ValueError(f"Column {col_name} not found in training data")
            if col_name not in data_df_test.columns:
                raise ValueError(f"Column {col_name} not found in test data")
        # Store main_scaler_label (for denormalisation of normalised losses later)
        params['denormalisation_params'] = {'main_scaler_label': main_scaler_label}

    # Normalise/Denormalise data
    for num_label, denom_label in all_denoms.items():
        if denormalise:
            # Get normalisation params
            if num_label in output_labels_pred:
                norm_val_1 = params['denormalisation_params'][main_scaler_label]['norm_val_1']
                norm_val_2 = params['denormalisation_params'][main_scaler_label]['norm_val_2']
            else:
                norm_val_1 = params['denormalisation_params'][num_label]['norm_val_1']
                norm_val_2 = params['denormalisation_params'][num_label]['norm_val_2']

            # Denormalise data
            if params['normalisation'] == 'minmax':
                data_df_results.loc[:, num_label] = normalise_minmax(data_df_results[num_label], norm_val_1, norm_val_2, denormalise=True) # type: ignore
            elif params['normalisation'] == 'gaussian':
                data_df_results.loc[:, num_label] = normalise_gaussian(data_df_results[num_label], norm_val_1, norm_val_2, denormalise=True) # type: ignore
        else:
            if do_not_normalise:
                print(f'WARNING: do_not_normalise is True, so normalisation=minmax is used')
                params['normalisation'] = 'minmax'
                norm_val_1 = 0
                norm_val_2 = 1
            else:
                if params['normalisation'] == 'minmax':
                    # Get min and max values
                    norm_val_1 = data_df_train[denom_label].min()
                    norm_val_2 = data_df_train[denom_label].max()
                elif params['normalisation'] == 'gaussian':
                    # Get mean and std
                    norm_val_1 = data_df_train[denom_label].mean()
                    norm_val_2 = data_df_train[denom_label].std()
            # Store important params (for denormalisation of normalised losses later)
            params['denormalisation_params'][num_label] = {
                'norm_val_1': norm_val_1, # min or mean
                'norm_val_2': norm_val_2 # max or std
            }
            # Normalise data
            if params['normalisation'] == 'minmax':
                data_df_train.loc[:, num_label] = normalise_minmax(data_df_train[num_label], norm_val_1, norm_val_2)
                data_df_test.loc[:, num_label] = normalise_minmax(data_df_test[num_label], norm_val_1, norm_val_2)
            elif params['normalisation'] == 'gaussian':
                data_df_train.loc[:, num_label] = normalise_gaussian(data_df_train[num_label], norm_val_1, norm_val_2)
                data_df_test.loc[:, num_label] = normalise_gaussian(data_df_test[num_label], norm_val_1, norm_val_2)
            else:
                raise ValueError(f'Normalisation method {params["normalisation"]} not supported')

    return data_df_train, data_df_test, data_df_results, params


def batch_tensors(input_train_shuffled, output_train_shuffled, input_test, output_test, batch_size, warnings_dict):
    '''
    Function to batch tensors.
    '''
    n_input_features = input_train_shuffled.shape[2]
    n_output_features = output_train_shuffled.shape[2]

    if batch_size is not None:

        N = input_train_shuffled.shape[1]
        if batch_size > N / 3:
            raise ValueError(f'Batch size {batch_size} must be less than 1/3 of the number of training examples {N}')
        
        # Handle leftovers
        leftover = N % batch_size
        if leftover > 0:
            # Number of extra entries needed to make the dataset size a multiple of batch_size
            pad_size = batch_size - leftover

            # If the needed zero-padding is more than 50% of batch_size, discard the leftover
            if pad_size > (batch_size / 2):
                # Discard leftover examples rather than padding
                input_train_shuffled = input_train_shuffled[:, :N - leftover, :]
                output_train_shuffled = output_train_shuffled[:, :N - leftover, :]
                # Warning
                if leftover > (N / 10):
                    warning = f"{leftover} leftover examples discarded (>10% of training examples)"
                    warnings_dict["data_utils:batch_size"] = warning
                    print('================================================')
                    print(f'WARNING: {warning}')
                    print('================================================')
            else:
                # Otherwise, pad as normal
                pad_input = torch.zeros(
                    (input_train_shuffled.shape[0], pad_size, n_input_features),
                    dtype=input_train_shuffled.dtype,
                    device=input_train_shuffled.device
                )
                pad_output = torch.zeros(
                    (output_train_shuffled.shape[0], pad_size, n_output_features),
                    dtype=output_train_shuffled.dtype,
                    device=output_train_shuffled.device
                )

                # Concatenate along the batch dimension (dim=1)
                input_train_shuffled = torch.cat((input_train_shuffled, pad_input), dim=1)
                output_train_shuffled = torch.cat((output_train_shuffled, pad_output), dim=1)
                # Warning
                if pad_size > (N / 10):
                    warning = f"{pad_size} zeroes were padded to the train data (>10% of training examples)"
                    warnings_dict["data_utils:batch_size"] = warning
                    print('================================================')
                    print(f'WARNING: {warning}')
                    print('================================================')

        # Recompute after discarding/padding
        input_train_shuffled = input_train_shuffled.view(-1, batch_size, n_input_features)
        output_train_shuffled = output_train_shuffled.view(-1, batch_size, n_output_features)

        # Info
        num_batches = input_train_shuffled.shape[0]
        batch_size_ratio = batch_size / N
    else:
        num_batches = 1
        batch_size_ratio = 1

    return input_train_shuffled, output_train_shuffled, input_test, output_test, num_batches, batch_size_ratio, warnings_dict


def get_tensors(data_df_train, data_df_test, params, warnings_dict, device=None, verbose=False):
    '''
    Function to convert data to PyTorch tensors.
    '''

    # Set seed for reproducibility (TBC why this is needed, but it is)
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # if using CUDA

    # Extract parameters from params
    input_labels = params['input_labels']
    output_labels = params['output_labels']
    batch_size = params['batch_size']

    # Drop the TestID and RepeatID columns
    data_df_train = data_df_train.drop(columns=['TestID', 'RepeatID'])
    data_df_test = data_df_test.drop(columns=['TestID', 'RepeatID'])

    # Define input and output
    input_train = data_df_train[input_labels].values
    output_train = data_df_train[output_labels].values
    input_test = data_df_test[input_labels].values
    output_test = data_df_test[output_labels].values

    # Convert to PyTorch tensors (batch, examples, input_size)
    input_train = torch.tensor(input_train, dtype=torch.float32).unsqueeze(0)
    output_train = torch.tensor(output_train, dtype=torch.float32).unsqueeze(0)
    input_test = torch.tensor(input_test, dtype=torch.float32).unsqueeze(0)
    output_test = torch.tensor(output_test, dtype=torch.float32).unsqueeze(0)

    # Randomise order of train data
    shuffle = torch.randperm(input_train.shape[1])
    input_train_shuffled = input_train[:, shuffle, :]
    output_train_shuffled = output_train[:, shuffle, :]

    # Batch tensors
    input_train_shuffled, output_train_shuffled, input_test, output_test, num_batches, batch_size_ratio, warnings_dict = batch_tensors(
        input_train_shuffled, 
        output_train_shuffled, 
        input_test, 
        output_test, 
        batch_size,
        warnings_dict)

    print(f"Batch size ratio: {batch_size_ratio:.2f}")
    if verbose:
        # Print shapes
        print(f"Batch size: {batch_size}")
        print("input train (shuffled) shape:", input_train_shuffled.shape)
        print("output train (shuffled) shape:", output_train_shuffled.shape)
        print("input test shape:", input_test.shape)
        print("output test shape:", output_test.shape)
        train_test_split = input_train.shape[1] / (input_train.shape[1] + input_test.shape[1])
        print(f"Train-test split: {train_test_split:.2f}")

    # Update params
    params['num_batches'] = num_batches
    params['shuffle_indices'] = shuffle
    params['batch_size_ratio'] = batch_size_ratio

    # Put data on GPU (if available)
    input_train_shuffled = input_train_shuffled.to(device)
    output_train_shuffled = output_train_shuffled.to(device)
    input_train = input_train.to(device)
    output_train = output_train.to(device)
    input_test = input_test.to(device)
    output_test = output_test.to(device)

    return input_train, output_train, input_train_shuffled, output_train_shuffled, input_test, output_test, params, warnings_dict


def unbatch_tensors(tensor_dict, architecture=None, convert_to_numpy=False, verbose=False):
    '''
    Function to unbatch data.
    '''
    
    # Convert to numpy if SVR or RF
    if architecture == 'RF' or architecture == 'SVR':
        convert_to_numpy = True
    
    for key, value in tensor_dict.items():
        if not convert_to_numpy:
            tensor_dict[key] = value.view(1, -1, value.shape[2])
        # Convert to numpy and sqeeze if not ANN
        else:
            tensor_dict[key] = value.numpy().squeeze()

    if verbose:
        print('----------------- UNBATCHED SHAPES -----------------')
        for key, value in tensor_dict.items():
            print(f'{key} shape:', value.shape)
        print('----------------------------------------------------')
    
    return tensor_dict