import pandas as pd
import copy
import numpy as np
from sklearn.model_selection import ParameterGrid
import time
import random
import torch
# Custom imports
from my_utils.helper_functions import get_string_from_dict_values
from my_utils.data_utils import get_tensors, unbatch_tensors, normalise_minmax, normalise_gaussian
from my_utils.ann_training_utils import train_ann_model
from my_utils.skl_training_utils import train_scikit_learn_model



def train_with_grid_search(
        df_train, 
        df_test, 
        params, 
        warnings_dict, 
        device, 
        model_class,
        architecture,
        best_config=None,
        process_results=True,
        verbose=False
    ):
    '''
    Function to train multiple multiples using grid search (or only the best config if specified).
    '''

    # Extract params
    hyper_grid = params['hyper_grid'].copy()
    save_all_models = params['save_all_models']

    # Get tensors
    _, _, input_train_shuffled, output_train_shuffled, input_test, output_test, params, warnings_dict = get_tensors(
        df_train,
        df_test,
        params,
        warnings_dict,
        device=device,
        verbose=False
    )

    # Store tensors in dict
    tensor_dict = {
        'input_train_shuffled': input_train_shuffled,
        'output_train_shuffled': output_train_shuffled,
        'input_test': input_test,
        'output_test': output_test
    }


    # Ensure data is unbatched for SVR or RF
    if architecture == 'SVR' or architecture == 'RF':
        tensor_dict = unbatch_tensors(tensor_dict, convert_to_numpy=True, verbose=False)

    # Only train best config if specified
    if best_config is not None:
        for key, value in best_config.items():
            hyper_grid[key] = [value]

    # Initialize variables
    best_loss = float('inf')
    all_results = [] 
    all_models = []

    # Hyperparameter search
    num_combinations = len(ParameterGrid(hyper_grid))
    print(f"Total number of parameter combinations: {num_combinations}")
    for idx, hyper_dict in enumerate(ParameterGrid(hyper_grid)):
        # Set seeds
        seed = params['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # if using CUDA
        
        # Train ANN
        if architecture == 'ANN':
            model_curr, losses_train, loss_test, test_preds = train_ann_model(
                model_class,
                device,
                params,
                hyper_dict,
                tensor_dict,
                verbose=verbose)
        
        # Train RF or SVR
        elif architecture == 'RF' or architecture == 'SVR':
            model_curr, loss_train, loss_test = train_scikit_learn_model(model_class, hyper_dict, tensor_dict, params)
            losses_train = [loss_train]
            test_preds = None

        # Print progress
        print(f'Trained model {idx + 1}/{num_combinations} with {hyper_dict} (test loss: {loss_test:.2e})')

        # Save results
        results_curr = {
            'hyperparam_config': hyper_dict,
            'loss_test': loss_test,
            'losses_train': losses_train,
            'test_preds': test_preds
        }
        all_results.append(results_curr)
        # Save all models
        if save_all_models:
            all_models.append(copy.deepcopy(model_curr))
        # Track the best
        if loss_test < best_loss:
            best_results = results_curr
            best_model = copy.deepcopy(model_curr)
            best_loss = loss_test

    if process_results:
        # Process results
        all_results, params = process_training_results(all_results, 'all_results', params)
        best_results, params = process_training_results(best_results, 'best_results', params)

    
    return all_results, best_results, best_model, all_models, warnings_dict



def process_training_results(df_results, results_type, params):
    '''
    Function to convert loss_test to denormalised RMS and rename columns.
    '''


    def convert_results_to_DataFrame(results):
        '''
        Input: results is a list of dictionaries or a single dictionary
        Output: DataFrame with hyperparam_config as index
        '''
        if not isinstance(results, list) and isinstance(results, dict):
            results = [results]
        elif not isinstance(results, list) and not isinstance(results, dict):
            raise ValueError(f'Results should be a list of dictionaries or a single dictionary. Got {type(results)} instead.')
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        # Replace hyperparam_config with stringified configs
        df_results['hyperparam_config'] = (
            df_results['hyperparam_config']
            .copy()
            .apply(get_string_from_dict_values)
        )
        # Set hyperparam_config as index
        df_results = df_results.set_index('hyperparam_config')
        return df_results


    # Extract parameters
    main_scaler_label = params['denormalisation_params']['main_scaler_label']
    norm_val_1 = params['denormalisation_params'][main_scaler_label]['norm_val_1']
    norm_val_2 = params['denormalisation_params'][main_scaler_label]['norm_val_2']

    # Convert to DataFrame if not already
    if not isinstance(df_results, pd.DataFrame):
        df_results = convert_results_to_DataFrame(df_results)

    # Define loss_type columns
    if results_type == 'validation_results':
        loss_type = 'mean_validation'
    elif results_type == 'best_results' or results_type == 'all_results':
        loss_type = 'test'
    old_loss_column = f'loss_{loss_type}'
    new_loss_column = f'{loss_type}_MSE [-]'
    new_loss_column_RMS = f'{loss_type}_RMS [MPa]'
    new_losses_train = f'losses_train_MSE [-]'

    # Convert loss_test to denormalised RMS
    if params['normalisation'] == 'minmax':
        df_results[new_loss_column_RMS] = np.sqrt(normalise_minmax(df_results[old_loss_column], norm_val_1, norm_val_2, denormalise_MSE=True))
    elif params['normalisation'] == 'gaussian':
        df_results[new_loss_column_RMS] = np.sqrt(normalise_gaussian(df_results[old_loss_column], norm_val_1, norm_val_2, denormalise_MSE=True))
    # Rename columns
    df_results.rename(columns={old_loss_column: new_loss_column}, inplace=True)
    if 'losses_train' in df_results.columns:
        df_results.rename(columns={'losses_train': new_losses_train}, inplace=True)
    # Sort by RMS
    df_results = df_results.sort_values(by=new_loss_column_RMS, ascending=True)
    # Store in params
    params[f'new_{results_type}_column_names'] = {
        'loss_test': new_loss_column,
        'loss_test_RMS': new_loss_column_RMS,
        'losses_train': new_losses_train
    }
    return df_results, params



def get_best_config_from_validation_results(df_validation_results):

    # Create a dictionary of configs
    configs_dict = {}
    for d in df_validation_results['hyperparam_config']:
        key = get_string_from_dict_values(d)
        configs_dict[key] = d

    # Replace hyperparam_config with stringified configs
    df_validation_results['hyperparam_config'] = (
        df_validation_results['hyperparam_config']
        .copy()
        .apply(get_string_from_dict_values)
    )

    df_validation_results = df_validation_results.pivot(index='hyperparam_config', columns='validation_TestID', values='loss_test')
    df_validation_results_mean = df_validation_results.mean(axis=1).to_frame(name='loss_mean_validation')
    df_validation_results_sorted = df_validation_results_mean.sort_values(by='loss_mean_validation', ascending=True)

    return configs_dict, df_validation_results_sorted
    


def train_with_validation(
        df_train, 
        params, 
        architecture,
        warnings_dict, 
        device, 
        model_class,
        verbose=False,
    ):
    '''
    Function to train the model with validation set if specified, then train best config on all data.
    '''

    # Extract parameters
    use_validation_set = params['use_validation_set']
    save_all_models = params['save_all_models']
    validation_TestIDs = params['validation_TestIDs']

    # Checks
    if save_all_models and use_validation_set:
        print('WARNING: If using lots of configurations, saving all models may take up lots of memory.')
    
    # Initialize results
    best_config = None
    config_dict = {}
    df_validation_results = None
    time_start = time.time()
    if use_validation_set:
        
        # Iterate over each validation set
        n_validation_sets = len(validation_TestIDs)
        df_val_tmp = None
        for idx, val_TestID in enumerate(validation_TestIDs):
            # Set seeds
            seed = params['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)  # if using CUDA

            print(f'--------------- Validation set TestID: {val_TestID} '
                  f'(Set {idx + 1}/{n_validation_sets}) ---------------')
            # Create train/validation splits
            df_train_validation = df_train[df_train['TestID'] == val_TestID]
            df_train_train = df_train[df_train['TestID'] != val_TestID]

            # Train
            all_results, _, _, _, warnings_dict = train_with_grid_search(
                df_train_train, 
                df_train_validation, 
                params, 
                warnings_dict, 
                device, 
                model_class,
                architecture,
                verbose=verbose,
                process_results=False)

            # Store results
            df_results = pd.DataFrame(all_results)
            df_results['validation_TestID'] = [val_TestID] * len(all_results)
            if df_val_tmp is None:
                df_val_tmp = df_results
            else:
                df_val_tmp = pd.concat([df_val_tmp, df_results], ignore_index=True)
        

        print('--------------------- Validation training complete ---------------------')
        # Get validation training results (average test loss for all validation set configurations)
        config_dict, df_validation_results = get_best_config_from_validation_results(df_val_tmp)
        df_validation_results, params = process_training_results(df_validation_results, 'validation_results', params)
        # Extract best config
        best_config = config_dict[df_validation_results.index[0]]
        # Extract all test predictions
        params['val_preds'] = df_val_tmp['test_preds'].copy() # type: ignore

        # Print training time
        time_end = time.time()
        validation_training_time = time_end - time_start
        print(f'Validation training time: {validation_training_time:.2f} seconds')
        params['validation_training_time'] = validation_training_time
    else:
        warning = 'Not using validation set'
        print('============================================================')
        print(f'WARNING: {warning}')
        print('============================================================')
        warnings_dict['train_and_predict_utils:use_validation_set'] = warning


    return df_validation_results, config_dict, best_config, params, warnings_dict



def retrain_best_config(
        df_train,
        device,
        architecture,
        params,
        warnings_dict,
        model_class,
        best_config,
        validation_set_index=0, 
    ):

    # Extract params
    validation_TestIDs = params['validation_TestIDs']
    testIDs_of_train_data = params['testIDs_of_train_data']
    
    print(f'Retraining model on selected validation set (index={validation_set_index})')
    # Get TestIDs
    validation_TestIDs = [validation_TestIDs[validation_set_index]]
    train_train_TestIDs = [testID for testID in testIDs_of_train_data if testID not in validation_TestIDs]
    # Get validation datasets
    df_validation = df_train[df_train['TestID'].isin(validation_TestIDs)].copy()
    df_train_train = df_train[df_train['TestID'].isin(train_train_TestIDs)].copy()

    # Retrain model on validation set
    _, best_results, best_model, _, warnings_dict = train_with_grid_search(
        df_train_train,
        df_validation, 
        params, 
        warnings_dict,
        device, 
        model_class,
        architecture,
        best_config=best_config, 
        process_results=True,
        verbose=False)
    
    return best_results, best_model, df_train_train, df_validation, warnings_dict