import copy
import torch.nn as nn
from .helper_functions import print_dict, print_warnings



def get_default_parameters(architecture, model_config_id):
    '''
    Get default parameters for a given architecture and model configuration ID.
    '''
    params_default = {}
    params_default['model_config_id'] = model_config_id
    params_default['architecture'] = architecture

    # Common data parameters
    params_default['data_path'] = 'data/exp_data_18-03-2025.csv' # Path to the data file
    params_default['param_name'] = 'Volume_Fraction' # Name of the varying parameter
    params_default['stress_threshold'] = 120 # [MPa] Remove stresses above this value
    params_default['additional_cleaning'] = True  # Perform additional data cleaning
    params_default['remove_outlier_curves'] = True # Remove outlier curves

    # Data loading + preprocessing parameters
    params_default['window'] = 5 # Window size for averaging (=None to disable) (default=5)
    params_default['normalisation'] = 'minmax' # 'minmax' or 'gaussian' or None
    params_default['testIDs_of_test_data'] = [1, 3, 5] # List of testIDs to use for testing
    params_default = get_model_config_id_params(params_default, model_config_id)

    # Curve length and binning parameters
    params_default['remove_discontinuities'] = True # Remove discontinuities in the test data due to averaging
    params_default['all_curves_same_length'] = False # True=All curves have the same length
    params_default['binning_params'] = {
        'curve_length_before_binning': 1000, # Number of points for uniform binning (default=1000) (if None, then strain_delta is used)
        'strain_delta': None, # Strain increment for uniform binning (default=0.0005) (if None, then curve_length_before_binning is used)
        'density': 0, # [0,1) where 0->1 = Uniform -> Gaussian (default=0.9)
        'location': 0.07, # (0,1) = left->right (default=0.07)
        'spread': 0.1, # [0,inf) = tight->wide (default=0.1)
        'verbose': False
    }
    
    # Input and output labels
    params_default['additional_labels'] = [] # ['Parallelism', 'Void_Fraction', 'Density', 'Aspect_Ratio']
    params_default['input_labels'] = get_input_labels(params_default)
    params_default['output_labels'] = ['EngStress']
    params_default['use_derivative_features'] = False # Calculates derivatives along stress-strain curve

    # Common model parameters
    params_default['criterion_string'] = 'MSE' # Loss function (MSE, MAE, RMS)
    if params_default['criterion_string'] == 'MSE':
        params_default['criterion'] = nn.MSELoss()
    params_default['seed'] = 0 # Random seed for reproducibility
    params_default['use_validation_set'] = True # True=Train using validation sets
    params_default['validation_TestIDs'] = [0, 2, 4, 6] # List of TestIDs to use for validation (if None, then use all TestIDs in train data)
    params_default['save_all_models'] = False # True=Save all models that are trained (cannot be used if use_validation_set=True)

    # Model parameters
    if architecture == 'ANN':
        # Training parameters
        params_default['batch_size'] = 400 # =number of data points in batch, None->no batching
        params_default['learning_rate'] = 0.001 # 0.01 more suitable when batch_size=None
        params_default['n_epochs'] = 2500
        params_default['activation'] = nn.ReLU()
        params_default['scheduler_step_size'] = 500
        params_default['scheduler_gamma'] = 0.5
        # Grid search parameters
        hidden_layer_sizes = [32, 64, 128, 256]
        layer_combinations2 = [[s1, s2] for s1 in hidden_layer_sizes for s2 in hidden_layer_sizes]
        layer_combinations1 = [[s1] for s1 in hidden_layer_sizes]
        # hidden_layer_sizes = [32, 64, 128]
        # layer_combinations3 = [[s1, s2, s3] for s1 in hidden_layer_sizes for s2 in hidden_layer_sizes for s3 in hidden_layer_sizes]
        layer_combinations3 = [[128, 64, 32], [32, 64, 128], [128, 32, 32], [32, 32, 128], [128, 128, 32], [32, 128, 128]]
        params_default['hyper_grid'] = {
            'candidate_hidden_configs': layer_combinations1 + layer_combinations2 + layer_combinations3,
            # 'candidate_hidden_configs': layer_combinations,
            'dropout_rate': [0], # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
            'weight_decay': [0] # 0.0, 0.001, 0.01, 0.1, 1
        }

    elif architecture == 'RF':
        params_default['batch_size'] = None
        # Grid search parameters
        params_default['hyper_grid'] = {
            'n_estimators': [100, 200, 300], # Number of trees in the forest
            'max_depth': [None, 20, 40], # Maximum depth of the tree
            'min_samples_split': [2, 5], # Minimum number of samples needed in a node before it can be considered for splitting (higher minimum -> prevents overfitting)
            'min_samples_leaf': [1, 2, 4] # Minimum number of samples required in a leaf node (higher minimum -> prevents overfitting)
        }
        # Set random seed for reproducibility
        params_default['hyper_grid']['random_state'] = [params_default['seed']] # List of random seeds (set to single seed for deterministic results)

    elif architecture == 'SVR':
        params_default['batch_size'] = None
        # Grid search parameters
        params_default['hyper_grid'] = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], # Kernel type
            'C': [0.1, 1, 10], # Regularization parameter
            'epsilon': [0.1, 0.2, 0.5], # The epsilon-tube: how much error you’re willing to tolerate without penalty
            'gamma': ['scale', 'auto'] # Kernel coefficient for 'rbf' (length scale), 'poly' and 'sigmoid'
        }
    else:
        raise ValueError(f'Architecture {architecture} not supported')
    
    return params_default




def parameter_checks(params, params_default, warnings_dict):
    '''
    Check if all keys in custom_params are in params_default, and show other warnings.
    '''

    # Check if all keys in custom_params are in params_default
    for key in params:
        if key not in params_default:
            warning_msg = f'Unknown key {key} in custom_params'
            warnings_dict[key] = warning_msg

    # Hard warnings
    if params['probabilistic_output'] and params['average_all_curves']:
        raise ValueError('Probabilistic output and averaged tests are not compatible')
    if params['use_validation_set'] and any(testID in params['validation_TestIDs'] for testID in params['testIDs_of_test_data']):
        raise ValueError(f'One of the validation TestIDs {params["validation_TestIDs"]} is in the test set {params["testIDs_of_test_data"]}')
    if params['model_config_id'] == 'all_curves' and params['remove_discontinuities']:
        warnings_dict['param_utils:remove_discontinuities'] = 'Removing discontinuities when model_config_id=all_curves has no effect'
    if params['model_config_id'] == 'all_curves' and params['remove_outlier_curves']:
        warnings_dict['param_utils:remove_outlier_curves'] = 'Training set may become unbalanced when using model_config_id=all_curves and remove_outlier_curves=True.'

    # Soft warnings
    if params['all_curves_same_length'] and params['binning_params']['density'] != 0:
        warnings_dict['param_utils:all_curves_same_length'] = 'all_curves_same_length will likely not have desired effect when binning_params["density"] != 0'
    if len(params['input_labels']) > 4 and params['model_config_id'] == 'mean_and_error_curves':
        warnings_dict['param_utils:input_labels'] = 'Using too many labels to predict average/probabilistic outputs is not recommended.'
    # ANN
    if params['architecture'] == 'ANN':
        if params['batch_size'] is not None and params['learning_rate'] > 0.001:
            warnings_dict['param_utils:learning_rate'] = 'learning_rate > 0.001 may be unstable when batch_size is not None'

    # Unsupported parameters
    if params['criterion_string'] != 'MSE':
        raise ValueError('This program only supports the MSE loss function currently.')
    
    return warnings_dict


def print_model_config_id_descriptions(model_config_id):
    if model_config_id == 'all_curves':
        print('Model config ID all_curves: Data point -> Data point')
    elif model_config_id == 'mean_and_error_curves':
        print('Model config ID mean_and_error_curves: Mean and error data point -> Mean and error data point')
    elif model_config_id == 'mean_curves':
        print('Model config ID mean_curves: Mean data point -> Mean data point')
    else:
        raise ValueError('Model config ID not valid')


def get_input_labels(params):
    if 'additional_labels' in params and 'param_name' in params:
        params['input_labels'] = [params['param_name'], 'EngStrain', *params['additional_labels']]
        return params
    else:
        raise ValueError('param_name and additional_labels must be in params dictionary')
    

def get_model_config_id_params(params_default, model_config_id):
    if model_config_id == 'all_curves':
        params_default['probabilistic_output'] = False # True=Use probabilistic output
        params_default['average_all_curves'] = False # True=Average all curves (same as probabilistic_output, but without error bars)
        params_default['remove_outlier_curves'] = False # True=Remove outlier curves
    elif model_config_id == 'mean_and_error_curves':
        params_default['probabilistic_output'] = True
        params_default['average_all_curves'] = False
    elif model_config_id == 'mean_curves':
        params_default['probabilistic_output'] = False
        params_default['average_all_curves'] = True
    else:
        raise ValueError(f'model_config_id is None or not valid')
    return params_default


def get_architecture_params(params, architecture, warnings_dict):
    new_params = {}
    if architecture == 'ANN':
        pass
    elif architecture == 'RF':
        pass
    elif architecture == 'SVR':
        pass

    # Add new params to params
    for key, value in new_params.items():
        params[key] = value
        # Add to warnings_dict
        warnings_dict[f'param_reset_warning:{key}'] = f'{key} has been reset to {value} based on the {architecture} architecture'

    return params, warnings_dict

def get_parameters(common_params, model_params, model_config_id, architecture, warnings_dict, verbose=False):
    '''
    Get parameters for a given configuration ID
    '''

    # Get default parameters
    params_default = get_default_parameters(architecture, model_config_id)

    # Combine common and model parameters
    custom_params = common_params | model_params

    # Add default parameters to params
    params = copy.deepcopy(custom_params)
    for key in params_default:
        if key not in params:
            params[key] = params_default[key]
        # if params[key] has more keys, then for loop through params[key]
        if isinstance(params_default[key], dict):
            for subkey in params_default[key]:
                if subkey not in params[key]:
                    params[key][subkey] = params_default[key][subkey]

    # Refresh condition-based parameters
    params = get_input_labels(params)
    params = get_model_config_id_params(params, model_config_id=model_config_id)
    params, warnings_dict = get_architecture_params(params, architecture, warnings_dict)

    # Checks
    warnings_dict = parameter_checks(params, params_default, warnings_dict)
    print_warnings(warnings_dict)

    # Print key parameters
    print_model_config_id_descriptions(params['model_config_id'])
    valid_labels = ['Volume_Fraction', 'Parallelism', 'Void_Fraction', 'Density', 'Aspect_Ratio']
    print(f'Omitted labels: {[label for label in valid_labels if label not in params["input_labels"]]}')
    print_dict(custom_params)
    if verbose:
        print_dict(params)
    
    return params, warnings_dict