import torch
import numpy as np
import pandas as pd
from .plotting_utils import plot_curves
from .data_utils import get_tensors, unbatch_tensors, get_curve_lengths, normalise_data
from autogluon.tabular import TabularDataset
# -------------------------------------------------------
# Data processing for predictions
# -------------------------------------------------------

def get_df_from_arrays(
        input, 
        output, 
        output_pred,
        input_labels,
        output_labels,
        curve_lengths_dict_with_repeats,
        keys
    ):
    '''
    Function to get a dataframe from a tensor.

    Tensors must have their dimensions in order of their labels.
    '''

    def format_array(arr):
        # Convert to numpy if tensor
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        # Remove batch dimension
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        else:
            return arr.reshape(-1, arr.shape[-1])


    def get_data_for_a_label(data, idx):
        data_label = []
        TestID_list = []
        repeat_list = []
        curve_lengths_cum = 0
        for key, value in curve_lengths_dict_with_repeats.items():
            if key in keys:
                # Get the curve indices
                curve_lengths_cum += value
                start_idx = curve_lengths_cum - value
                end_idx = start_idx + value
                # Collect the data
                data_curve = data[start_idx:end_idx, idx]
                data_label.extend(data_curve)
                # Get the test and repeat IDs
                test_ids = key[0] * np.ones(value)
                repeat_ids = key[1] * np.ones(value)
                TestID_list.extend(test_ids)
                repeat_list.extend(repeat_ids)

        return data_label, TestID_list, repeat_list

    # Initialise dataframe
    df = pd.DataFrame()

    # Format arrays
    input, output, output_pred = map(format_array, [input, output, output_pred])

    # Get test and repeat IDs
    _, TestID_list, repeat_list = get_data_for_a_label(input, 0)
    df['TestID'] = TestID_list
    df['RepeatID'] = repeat_list
    
    # Get input data
    for idx, label in enumerate(input_labels):
        data_label, _, _ = get_data_for_a_label(input, idx)
        df[label] = data_label

    # Get output data
    for idx, label in enumerate(output_labels):
        data_label, _, _ = get_data_for_a_label(output, idx)
        df[label] = data_label

    # Get predicted output data
    for idx, label in enumerate(output_labels):
        data_label, _, _ = get_data_for_a_label(output_pred, idx)
        df[label + '_pred'] = data_label

    return df


def get_params_for_analysis(params, data_df_train, data_df_test):
    '''
    Function to get the parameters for analysis.
    '''

    # Get curve length information
    _, curve_lengths_dict, curve_lengths_dict_with_repeats = get_curve_lengths(data_df_train, data_df_test)
    _, _, cld_train = get_curve_lengths(data_df_train)
    train_keys = list(cld_train.keys())
    _, _, cld_test = get_curve_lengths(data_df_test)
    test_keys = list(cld_test.keys())

    # Store in params
    params['curve_lengths_dict'] = curve_lengths_dict
    params['curve_lengths_dict_with_repeats'] = curve_lengths_dict_with_repeats
    params['train_keys'] = train_keys
    params['test_keys'] = test_keys

    return params


def get_data_df_with_predictions(
        params, 
        data_df_train, 
        data_df_test,
        tensor_dict,
        output_pred_train, 
        output_pred_test,
        denormalise=False
    ):
    '''
    Function to prepare the data for analysis.
    '''

    # Extract params
    input_labels = params['input_labels']
    output_labels = params['output_labels']
    # Put tensors on cpu
    if isinstance(output_pred_train, torch.Tensor):
        output_pred_train = output_pred_train.cpu()
    if isinstance(output_pred_test, torch.Tensor):
        output_pred_test = output_pred_test.cpu()
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_dict[key] = value.cpu()
    # Extract tensors
    input_train = tensor_dict['input_train']
    output_train = tensor_dict['output_train']
    input_test = tensor_dict['input_test']
    output_test = tensor_dict['output_test']

    # Get params
    params = get_params_for_analysis(params, data_df_train, data_df_test)
    # Extract params
    curve_lengths_dict_with_repeats = params['curve_lengths_dict_with_repeats']
    train_keys = params['train_keys']
    test_keys = params['test_keys']

    # Turn tensors into dataframes
    # Train
    data_df_train_results = get_df_from_arrays(
        input_train,
        output_train,
        output_pred_train,
        input_labels,
        output_labels,
        curve_lengths_dict_with_repeats,
        train_keys)
    # Test
    data_df_test_results = get_df_from_arrays(
        input_test,
        output_test,
        output_pred_test,
        input_labels,
        output_labels,
        curve_lengths_dict_with_repeats,
        test_keys)
    # Concatenate train and test results
    data_df_results_normalised = pd.concat([data_df_train_results, data_df_test_results])

    # Revert normalisation
    _, _, data_df_final_results, _ = normalise_data(
        data_df_train,
        data_df_test,
        params,
        data_df_results=data_df_results_normalised,
        denormalise=denormalise)

    return params, data_df_final_results




# -------------------------------------------------------------------
# Prediction utils
# -------------------------------------------------------------------




def get_prediction_tensors(model_class, architecture, data_df_train, data_df_test, params, warnings_dict, device):
    '''
    Function to get the predictions of the model on the train and test set.
    '''

    # Get ordered tensors for all data
    input_train, output_train, _, _, input_test, output_test, params, warnings_dict = get_tensors(
                data_df_train,
                data_df_test,
                params,
                warnings_dict,
                device=device,
                verbose=False)
    
    tensor_dict = {
        'input_train': input_train,
        'output_train': output_train,
        'input_test': input_test,
        'output_test': output_test
    }

    # Ensure data is unbatched
    input_train, output_train, input_test, output_test = unbatch_tensors(
        tensor_dict, 
        architecture=architecture, 
        verbose=False
    ).values()    

    # Get predictions
    print(f'------------------ {architecture} Predictions ------------------')
    if architecture == 'ANN':
        with torch.no_grad():
            # Train set
            output_pred_train = model_class(input_train)
            # Test set
            output_pred_test = model_class(input_test)
    elif architecture == 'RF' or architecture == 'SVR':
        # Train set
        output_pred_train = model_class.predict(input_train)
        # Test set
        output_pred_test = model_class.predict(input_test)
    elif architecture == 'AutoGluon':
        # AutoGluon TabularDataset
        input_train = TabularDataset(data_df_train)
        input_test = TabularDataset(data_df_test)
        # Train set
        output_pred_train = torch.as_tensor(model_class.predict(input_train))
        # Test set
        output_pred_test = torch.as_tensor(model_class.predict(input_test))

    # Get losses
    criterion = params['criterion']
    loss_test = criterion(torch.as_tensor(output_test), torch.as_tensor(output_pred_test)).item()
    loss_train = criterion(torch.as_tensor(output_train), torch.as_tensor(output_pred_train)).item()
    print(f"Test loss: {loss_test:.2e}")
    print(f"Train loss: {loss_train:.2e}")

    return tensor_dict, output_pred_train, loss_train, output_pred_test, loss_test



def get_predictions_and_plot(
        df_train, 
        df_test, 
        best_model,
        architecture, 
        device, 
        params, 
        warnings_dict, 
        denormalise=True,
        export_to_csv=False
    ):
    """
    Get predictions and plot them. If retrain_model is True, the model will be retrained on the validation set.
    """
    # Get TestIDs from input dataframes
    train_TestIDs = df_train['TestID'].unique()
    test_TestIDs = df_test['TestID'].unique()

    # Get predictions
    tensor_dict, output_pred_train, _, output_pred_test, _ = get_prediction_tensors(
        best_model,
        architecture,
        df_train,
        df_test,
        params,
        warnings_dict,
        device)
    
    # Get dataframes with predictions
    params, data_df_results = get_data_df_with_predictions(
        params, 
        df_train, 
        df_test, 
        tensor_dict,
        output_pred_train, 
        output_pred_test,
        denormalise=denormalise)
    
    # Plot train set predictions
    plot_curves(data_df_results, 
                params, 
                TestID_list=train_TestIDs, 
                plot_predictions=True,
                legend_type='test')

    # Plot train set predictions
    plot_curves(data_df_results, 
                params, 
                TestID_list=test_TestIDs, 
                plot_predictions=True,
                legend_type='test',
                export_to_csv=export_to_csv)
    
    return data_df_results, params, warnings_dict