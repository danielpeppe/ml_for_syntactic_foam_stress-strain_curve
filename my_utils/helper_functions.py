from tabulate import tabulate
import numpy as np
import torch
from .normalisation_utils import normalise_minmax, normalise_gaussian

def print_warnings(warnings_dict):
    '''
    Function to print the warnings.
    '''
    if warnings_dict:
        print("========================================")
        for key, value in warnings_dict.items():
            print(f"WARNING: {key}: {value}")
        print("========================================")
    else:
        print("No warnings")



def print_dict(d, indent=4):
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{' ' * indent}{key}:")
            print_dict(value, indent+4)
        else:
            print(f"{' ' * indent}{key}: {value}")


def get_string_from_dict_values(d):
    '''
    Function to get a string from the values of a dictionary.

    Requirements:
    - The dictionary should not have nested lists.
    '''
    if not isinstance(d, dict):
        raise ValueError(f"Cannot convert {d} to string. It is not a dictionary.")
    else:
        string_list = []
        for key, value in d.items():
            if isinstance(value, list):
                if isinstance(value[0], list):
                    raise ValueError(f"Cannot convert {value} to string. It is a nested list.")
                else:
                    string_list.append('-'.join(str(v) for v in value))
            else:
                string_list.append(str(value))
        return '-'.join(string_list)
    


def print_training_results(params, df, results_type):
    '''
    Function to print the results in a pretty format using scientific notation.
    '''


    def print_df(df):
        if df is None:
            print(None)
        else:
            # Apply scientific notation only to float columns
            for col in df.select_dtypes(include='float'):
                df[col] = df[col].map(lambda x: f"{x:.2e}" if x < 1e-1 else f"{x:.2f}")
                df[col] = df[col].map(lambda x: x.replace('e', 'x10^'))
            print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

    # Print the title and the dataframe
    if results_type == 'validation_results':
        title = '================================ VALIDATION TRAINING RESULTS ================================='
    elif results_type == 'best_results':
        title = '================================= BEST RESULTS ON TEST SET ==================================='
    elif results_type == 'all_results':
        title = '================================== ALL RESULTS ON TEST SET ==================================='
    else:
        raise ValueError(f'Invalid results_type: {results_type}')

    if df is None:
        # Print None if the dataframe is None
        print(title)
        print(None)
    else:
        # Drop the losses_train_MSE [-] column if the dataframe is best_results or all_results
        if results_type == 'best_results' or results_type == 'all_results':
            new_losses_train = params[f'new_{results_type}_column_names']['losses_train']
            df_tmp = df.copy().drop(columns=[new_losses_train])
        else:
            df_tmp = df.copy()

        # Print the title and the dataframe
        print(title)
        print_df(df_tmp)



def print_train_loss_from_losses(losses_train, params):

    final_train_loss = losses_train[-1]
    print(f'Final train loss: {final_train_loss:.2e}')

    norm_val_1 = params['denormalisation_params'][params['denormalisation_params']['main_scaler_label']]['norm_val_1']
    norm_val_2 = params['denormalisation_params'][params['denormalisation_params']['main_scaler_label']]['norm_val_2']
    if params['normalisation'] == 'minmax':
        final_train_loss_RMS = np.sqrt(normalise_minmax(final_train_loss, norm_val_1, norm_val_2, denormalise_MSE=True))
    elif params['normalisation'] == 'gaussian':
        final_train_loss_RMS = np.sqrt(normalise_gaussian(final_train_loss, norm_val_1, norm_val_2, denormalise_MSE=True))
        
    print(f'Final train loss (RMS_denorm): {final_train_loss_RMS:.2e}')



def print_mean_stress_RMSE_from_results(data, error_label='test_error'):
    df_results = data['data_df_final_results'].copy()
    # df_results_val = data['data_df_validation_results'].copy()
    params_tmp = data['params'].copy()

    # Get only test set data
    if error_label == 'test_error':
        TestIDs = params_tmp['testIDs_of_test_data']
        df = df_results[df_results['TestID'].isin(TestIDs)]
    elif error_label == 'train_error':
        TestIDs = params_tmp['testIDs_of_train_data']
        df = df_results[df_results['TestID'].isin(TestIDs)]
    # elif error_label == 'validation_error':
    #     TestIDs = params_tmp['validation_TestIDs']
    #     df = df_results_val[df_results_val['TestID'].isin(TestIDs)]
    else:
        raise ValueError(f'Invalid error_label: {error_label}')

    # Get the mean stress and error stress
    mean_stress = df['EngStress_mean']
    mean_stress_pred = df['EngStress_mean_pred']
    error_stress = df['EngStress_error']
    error_stress_pred = df['EngStress_error_pred']
    
    # Convert to torch tensors
    mean_stress = torch.tensor(mean_stress.values, dtype=torch.float32)
    mean_stress_pred = torch.tensor(mean_stress_pred.values, dtype=torch.float32)
    error_stress = torch.tensor(error_stress.values, dtype=torch.float32)
    error_stress_pred = torch.tensor(error_stress_pred.values, dtype=torch.float32)

    # Combine the mean stress and error stress into a single tensor
    combined_stress = torch.cat((mean_stress.unsqueeze(1), error_stress.unsqueeze(1)), dim=1)
    combined_stress_pred = torch.cat((mean_stress_pred.unsqueeze(1), error_stress_pred.unsqueeze(1)), dim=1)

    criterion = params_tmp['criterion']

    # if error_label == 'validation_error':
    #     # data is normalised in data_df_validation_results
    #     MSE = criterion(combined_stress, combined_stress_pred)
    #     MSE_mean_stress = criterion(mean_stress, mean_stress_pred)
    #     # Denormalise the output stress
    #     main_scaler_label = params_tmp['denormalisation_params']['main_scaler_label']
    #     main_scaler_norm_val_1 = params_tmp['denormalisation_params'][main_scaler_label]['norm_val_1']
    #     main_scaler_norm_val_2 = params_tmp['denormalisation_params'][main_scaler_label]['norm_val_2']

    #     if params_tmp['normalisation'] == 'minmax':
    #         RMSE_star_all_labels = np.sqrt(normalise_minmax(MSE, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise_MSE=True))
    #         RMSE_star_mean_stress = np.sqrt(normalise_minmax(MSE_mean_stress, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise_MSE=True))
    #     elif params_tmp['normalisation'] == 'gaussian':
    #         RMSE_star_all_labels = np.sqrt(normalise_gaussian(MSE, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise_MSE=True))
    #         RMSE_star_mean_stress = np.sqrt(normalise_gaussian(MSE_mean_stress, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise_MSE=True))
    # else:
    MSE_all_labels = criterion(combined_stress, combined_stress_pred)
    MSE_mean_stress = criterion(mean_stress, mean_stress_pred)
    MSE_error_stress = criterion(error_stress, error_stress_pred)

    # Data is denormalised in data_df_final_results
    RMSE_star_mean_stress = np.sqrt(MSE_mean_stress)
    RMSE_star_error_stress = np.sqrt(MSE_error_stress)
    RMSE_star_all_labels = np.sqrt(MSE_all_labels)

    
    # Get RMSE* for all labels
    # RMSE_star_all_labels = float(data['df_best_results']['test_RMS [MPa]'].values.squeeze())

    print('Calculating MSEs/RMSE*s for best model:')
    print(f'    All labels: MSE: {MSE_all_labels:.2e}={(MSE_mean_stress + MSE_error_stress)/2:.2e} RMSE*: {RMSE_star_all_labels:.2f}')
    print(f'    Mean stress: MSE: {MSE_mean_stress:.2e} RMSE*: {RMSE_star_mean_stress:.2f}')
    print(f'    Error stress: MSE: {MSE_error_stress:.2e} RMSE*: {RMSE_star_error_stress:.2f}')



def print_yield_strength(data=None, df_results=None, strain_threshold=0.1, strain_label='EngStrain'):
    
    if data is None and df_results is None:
        raise ValueError('Either data or df_results must be provided')
    elif data is not None and df_results is not None:
        raise ValueError('Only one of data or df_results must be provided')
    elif data is not None:
        df_results = data['data_df_final_results'].copy()
    if df_results is None:
        raise ValueError('df_results must be provided')
    
    ys_dict = {}
    ys_pred_dict = {}
    for gid, group in df_results.groupby('TestID'):
        group_clipped = group[group[strain_label] < strain_threshold]
        ys = np.max(group_clipped['EngStress_mean'])
        ys_dict[gid] = ys

        ys_pred = np.max(group_clipped['EngStress_mean_pred'])
        ys_pred_dict[gid] = ys_pred

    for gid, ys in ys_dict.items():
        print(f'Yield strength for TestID {gid}: {ys:.2f} MPa (Predicted: {ys_pred_dict[gid]:.2f} MPa)')

    return ys_dict, ys_pred_dict

