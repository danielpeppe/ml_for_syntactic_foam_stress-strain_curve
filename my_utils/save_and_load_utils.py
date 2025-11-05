import os
import pickle
import torch
import joblib
import datetime as dt

def save_and_load_model_and_results(
        mode=None, 
        filename_base=None, 
        overwrite_existing_filename=False,
        model=None, 
        version=1, 
        architecture=None,
        params=None,
        warnings_dict=None, 
        data_df_final_results=None,
        data_df_validation_results=None, # v2
        best_results=None,  # v1
        all_results=None,  # v1
        df_results_dict=None,  # v2
        model_config_id=None, # v2
        date=None, # v2
    ):
    '''
    Function to save or load model config, warnings, and results to a file.
    '''


    def get_model_path(filename_base, model_dir, version_dir, ext, file_type=None, version=None, model_config_id=None, date=None):
        if version == 2:
            if file_type is None:
                raise ValueError('file_type must be provided when version is 2')
            if date is None:
                date = dt.datetime.now().strftime('%m-%d')
            if model_config_id == 'all_curves':
                config_int = 1
            elif model_config_id == 'mean_and_error_curves':
                config_int = 2
            elif model_config_id == 'mean_curves':
                config_int = 3
            else:
                raise ValueError(f'model_config_id must be one of {["all_curves", "mean_and_error_curves", "mean_curves"]}')
            model_file_name = f'{filename_base}_{file_type}_{config_int}_{date}.{ext}'
        else:
            model_file_name = f'{filename_base}.{ext}'
        dir_path = os.path.join(model_dir, version_dir)
        model_file_path = os.path.join(dir_path, model_file_name)
        return dir_path, model_file_path
    

    
    valid_versions = [1, 2]
    if version not in valid_versions:
        raise ValueError(f'version must be one of {valid_versions}')
    else:
        model_dir = 'models'
        version_dir = os.path.join('v' + str(version))
    if mode != 'save' and mode != 'load':
        raise ValueError(f'mode must be one of {["save", "load"]}')

    if version == 1:

        if mode == 'save':
            if model is None or params is None or warnings_dict is None or best_results is None or data_df_final_results is None or all_results is None:
                raise ValueError('model, params, warnings_dict, best_results, data_df_final_results, and all_results must be provided when mode is save')
            
            # Create a dictionary to store all the data
            data = {}
            data['params'] = params
            # Results
            data['best_results'] = best_results
            data['all_results'] = all_results
            # Warnings
            data['warnings_dict'] = warnings_dict
            # Predictions
            data['data_df_final_results'] = data_df_final_results
            
            if filename_base is None:
                # Create model base name
                hidden_layer_sizes = data['best_results'][0]
                testIDs_of_test_data = data['params']['testIDs_of_test_data']
                model_config_id = data['params']['model_config_id']
                # params_hash = dict_hash(data['params'])
                
                hidden_layer_config_str = '-'.join(map(str, hidden_layer_sizes))
                testIDs_str = '-'.join(map(str, testIDs_of_test_data))
                # filename_base = f'model_v{version}_{model_config_id}_{hidden_layer_config_str}_{testIDs_str}_{params_hash}'
                filename_base = f'model_v{version}_{model_config_id}_{hidden_layer_config_str}_{testIDs_str}'

            # Define model path
            dir_path, model_file_path = get_model_path(filename_base, model_dir, version_dir, 'pkl', version=version)
            # Checks
            if os.path.exists(model_file_path) and overwrite_existing_filename:
                os.remove(model_file_path)
            elif os.path.exists(model_file_path) and not overwrite_existing_filename:
                raise ValueError(f'Model {model_file_path} already exists. Set overwrite_existing_filename=True to overwrite.')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            # Save the data
            with open(model_file_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Get file size in MB
            pickle_size_mb = os.path.getsize(model_file_path) / (1024 * 1024)
            
            # Save the model
            dir_path, model_file_path = get_model_path(filename_base, model_dir, version_dir, 'pth', version=version)
            torch.save(model, model_file_path)
            
            # Info
            print(f'Model config and results saved to models/{filename_base}')
            print(f'Pickle file size: {pickle_size_mb:.2f} MB')

        elif mode == 'load':
            if filename_base is None:
                raise ValueError('filename_base must be provided when mode is load')
            if not os.path.exists(model_dir):
                raise ValueError('models directory does not exist')
            
            # Define model path
            dir_path, model_file_path = get_model_path(filename_base, model_dir, version_dir, 'pkl', version=version)
            # Checks
            if not os.path.exists(model_file_path):
                raise ValueError(f'Model {model_file_path} does not exist')
            # Load the data
            with open(model_file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Load the model
            dir_path, model_file_path = get_model_path(filename_base, model_dir, version_dir, 'pth', version=version)
            model = torch.load(model_file_path)

            return data, model

        

    elif version == 2:

        if mode == 'save':
            required_data = ['model', 'params', 'warnings_dict', 'df_results_dict', 'data_df_final_results']
            for data in required_data:
                if data is None:
                    raise ValueError(f'{data} must be provided when mode is save')
            else:
                print(f'Saving {", ".join(required_data)} and any additional data provided')
            if filename_base is None:
                raise ValueError('filename_base must be provided for version 2')
            
            # Create a dictionary to store all the data
            data = {}
            data['params'] = params
            # Training results
            data['df_validation_results'] = df_results_dict['validation_results'] # type: ignore
            if data_df_validation_results is not None:
                data['data_df_validation_results'] = data_df_validation_results
            # Test results
            data['df_best_results'] = df_results_dict['best_results'] # type: ignore
            data['df_all_results'] = df_results_dict['all_results'] # type: ignore
            data['data_df_final_results'] = data_df_final_results
            # Warnings
            data['warnings_dict'] = warnings_dict

            # Define model path
            dir_path, data_file_path = get_model_path(filename_base, model_dir, version_dir, 'pkl', file_type='data', version=version, model_config_id=model_config_id, date=date)
            # Checks
            if os.path.exists(data_file_path) and overwrite_existing_filename:
                os.remove(data_file_path)
            elif os.path.exists(data_file_path) and not overwrite_existing_filename:
                raise ValueError(f'Model {data_file_path} already exists. Set overwrite_existing_filename=True to overwrite.')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            # Save the data
            with open(data_file_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save the model
            if architecture == 'ANN':
                dir_path, model_file_path = get_model_path(filename_base, model_dir, version_dir, 'pth', file_type='model', version=version, model_config_id=model_config_id, date=date)
                torch.save(model, model_file_path)
            elif architecture == 'RF' or architecture == 'SVR':
                dir_path, model_file_path = get_model_path(filename_base, model_dir, version_dir, 'joblib', file_type='model', version=version, model_config_id=model_config_id, date=date)
                joblib.dump(model, model_file_path)
            
            # Info
            time_str = dt.datetime.now().strftime('%H:%M:%S')
            print(f'{time_str}: Model config and results saved to {os.path.join(model_dir, version_dir, filename_base)}')
            data_size_mb = os.path.getsize(data_file_path) / (1024 * 1024)
            print(f'{time_str}: Data file size: {data_size_mb:.2f} MB')
            model_size_mb = os.path.getsize(model_file_path) / (1024 * 1024)
            print(f'{time_str}: Model file size: {model_size_mb:.2f} MB')

        elif mode == 'load':
            if filename_base is None:
                raise ValueError('filename_base must be provided')
            if not os.path.exists(model_dir):
                raise ValueError('models directory does not exist')
            
            # Define model path
            dir_path, data_file_path = get_model_path(filename_base, model_dir, version_dir, 'pkl', file_type='data', version=version, model_config_id=model_config_id, date=date)
            # Checks
            if not os.path.exists(data_file_path):
                raise ValueError(f'Data {data_file_path} does not exist')
            # Load the data
            with open(data_file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Load the model
            if architecture == 'ANN':
                dir_path, model_file_path = get_model_path(filename_base, model_dir, version_dir, 'pth', file_type='model', version=version, model_config_id=model_config_id, date=date)
                model = torch.load(model_file_path, map_location=torch.device('cpu'))
            elif architecture == 'RF' or architecture == 'SVR':
                dir_path, model_file_path = get_model_path(filename_base, model_dir, version_dir, 'joblib', file_type='model', version=version, model_config_id=model_config_id, date=date)
                model = joblib.load(model_file_path)

            # Notify user
            print(f'Model {model_file_path} loaded')
            print(f'Data {data_file_path} loaded')

            return data, model

