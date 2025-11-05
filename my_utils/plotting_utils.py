from pprint import pformat
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import numpy as np
import torch
import os
import pandas as pd
from scipy.io import savemat
import time
# Relative import
from .data_utils import generate_x_vals_for_bins
from .normalisation_utils import normalise_minmax, normalise_gaussian

def plot_curves(
        data_df, 
        params, 
        TestID_list=None, 
        plot_predictions=False,
        legend_type='test',
        add_markers_to_pred=False,
        raw_curves=False,
        export_to_csv=False
    ):
    '''
    Function to plot all curves in the dataframe.

    data_df: dataframe containing the data
    param_name: name of the parameter to plot
    TestID_list: list of test IDs to plot
    probabilistic_output: whether to plot the probabilistic output
    plot_predictions: whether to plot the predicted curves
    legend_type: 'repeat' or 'test'
    add_markers_to_pred: whether to add markers to the predicted curves
    raw_curves: whether to plot the raw curves
    '''

    if TestID_list is not None and not (isinstance(TestID_list, list) or isinstance(TestID_list, np.ndarray)):
        raise ValueError("TestID_list must be a list or numpy array")

    # Extract params
    probabilistic_output = params['probabilistic_output']
    average_all_curves = params['average_all_curves']
    param_name = params['param_name']
    if raw_curves:
        probabilistic_output = False
        average_all_curves = False

    fig, ax = plt.subplots(figsize=(10, 6))

    # Select tests to plot
    if TestID_list is None:
        TestID_list = data_df['TestID'].unique()
    if not all(test in data_df['TestID'].unique() for test in TestID_list):
        raise ValueError("Some tests in TestID_list are not present in the data")
    data_df = data_df[data_df['TestID'].isin(TestID_list)]

    # Get colour for each test
    cmap = plt.cm.viridis # type: ignore
    norm = mcolors.Normalize(vmin=min(TestID_list), vmax=max(TestID_list))
    color_list = cmap(norm(TestID_list))

    # Define plot args
    plot_args_true = {
        'alpha': 0.5,
    }
    plot_args_pred = {
        'alpha': 1.0,
        'linestyle': '--',
    }
    if add_markers_to_pred:
        plot_args_pred['marker'] = 'o'
        plot_args_pred['markersize'] = 0.5
    fill_between_plot_args = {
        'alpha': 0.2,
    }    

    def get_data_dict(data_i, test_id, repeat_id, strain, output_label_list):
        # For exporting to csv
        data_dict = {'TestID': test_id, 'RepeatID': repeat_id, 'strain': strain.values.tolist()}
        for label in output_label_list:
            data_dict[label] = data_i[label].values.tolist()
        
        return data_dict


    def plot_curve(data_i, test_counter):
        test_id = int(data_i['TestID'].iloc[0])
        repeat_id = int(data_i['RepeatID'].iloc[0])
        param_i = data_i[param_name].iloc[0]
        strain = data_i['EngStrain']
        color = color_list[test_counter]
        if legend_type == 'test':
            label = f"T{test_id}: ({param_name}={param_i:.2f})"
        elif legend_type == 'repeat':
            label = f"T{test_id}-R{repeat_id}: ({param_name}={param_i:.2f})"

        output_label_list = []
        if probabilistic_output:           
            output_label_list.append('EngStress_mean')
            mean = data_i['EngStress_mean']
            ax.plot(strain, mean, label=label, color=color, **plot_args_true) # type: ignore

            if plot_predictions:
                output_label_list.append('EngStress_mean_pred')
                mean_pred = data_i['EngStress_mean_pred']
                ax.plot(strain, mean_pred, label=label, color=color, **plot_args_pred)
                output_label_list.append('EngStress_error_pred')
                error_pred = data_i['EngStress_error_pred']
                error_lower_pred = mean_pred - error_pred/2
                error_upper_pred = mean_pred + error_pred/2
                ax.fill_between(strain, error_lower_pred, error_upper_pred, color=color, **fill_between_plot_args) # type: ignore
            else:
                output_label_list.append('EngStress_error')
                error = data_i['EngStress_error']
                error_lower = mean - error/2
                error_upper = mean + error/2
                ax.fill_between(strain, error_lower, error_upper, color=color, **fill_between_plot_args) # type: ignore
        elif average_all_curves:
            output_label_list.append('EngStress_mean')
            mean = data_i['EngStress_mean']
            ax.plot(strain, mean, label=label, color=color, **plot_args_true) # type: ignore
            if plot_predictions:
                output_label_list.append('EngStress_mean_pred')
                mean_pred = data_i['EngStress_mean_pred']
                ax.plot(strain, mean_pred, label=label, color=color, **plot_args_pred) # type: ignore
        else:
            output_label_list.append('EngStress')
            stress = data_i['EngStress']
            ax.plot(strain, stress, label=label, color=color, **plot_args_true) # type: ignore
            if plot_predictions:
                output_label_list.append('EngStress_pred')
                stress_pred = data_i['EngStress_pred']
                ax.plot(strain, stress_pred, label=label, color=color, **plot_args_pred) # type: ignore

        data_dict = get_data_dict(data_i, test_id, repeat_id, strain, output_label_list)

        return data_dict

    df_for_csv = pd.DataFrame()
    test_counter = 0
    for _, data_test in data_df.groupby('TestID'):
        for _, data_repeat in data_test.groupby('RepeatID'):
            data_dict = plot_curve(data_repeat, test_counter)
            df_for_csv = pd.concat([df_for_csv, pd.DataFrame(data_dict)], ignore_index=True)
        test_counter += 1

    # Export to csv
    if export_to_csv:
        df_for_csv.to_csv('data_for_csv.csv', index=False)

    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress (MPa)")
    # Set limits
    # max_strain = max(data_df['EngStrain'])
    # ax.set_xlim(0, max_strain)
    # ax.set_ylim(0, 120)
    # Only plot legend for each testID once
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.grid()


def plot_stress_vs_vf(data_df):
    '''
    Function to plot stress vs volume fraction for each strain value.
    '''

    data_df_temp = data_df.copy()

    stress_vals = []
    strain_vals = np.arange(0, 0.5, 0.05)

    for _, group in data_df_temp.groupby('TestID'):
        stress_vals_group = []

        for _, group_repeat in group.groupby('RepeatID'):
            
            stress_vals_group_repeat = []
            group_strain = group_repeat['EngStrain'].values
            group_stress = group_repeat['EngStress'].values

            for strain in strain_vals:
                # Find the closest strain value
                idx = (np.abs(group_strain - strain)).argmin()
                stress_vals_group_repeat.append(group_stress[idx])
            
            stress_vals_group.append(stress_vals_group_repeat)
        
        stress_vals_group = np.vstack(stress_vals_group)
        stress_vals.append(np.mean(stress_vals_group, axis=0))

    stress_vals = np.vstack(stress_vals)

    # Get all volume fractions
    volume_fractions = data_df_temp['Volume_Fraction'].unique()

    # Define colormap
    cmap = plt.cm.viridis # type: ignore
    norm = mcolors.Normalize(vmin=min(strain_vals), vmax=max(strain_vals))  # Normalize strain values


    fig, ax = plt.subplots()  # Create figure and axis
    for i in range(len(strain_vals)):
        ax.plot(volume_fractions, stress_vals[:, i], color=cmap(norm(strain_vals[i])), alpha=0.7)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for colorbar
    cbar = fig.colorbar(sm, ax=ax)  # Attach colorbar to the correct figure
    cbar.set_label('Strain')

    # Labels & Title
    ax.set_xlabel('Volume Fraction')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Stress vs Volume Fraction')

    plt.show()


def plot_num_params_vs_test_loss(results):
    '''
    Function to plot the number of parameters in the model using the candidate_hidden_configs and the test loss for that config
    '''

    hidden_configs = []
    for i, result in enumerate(results):
        hidden_configs.append(result['hyperparam_config'])

    #calculate number of weights in the model
    total_params_list = []
    test_losses_list = []
    for i, config in enumerate(hidden_configs):
        hidden_config_pairs = [config[i:i+2] for i in range(0, len(config) - 1)]
        num_params = [l1 * l2 + l2 for l1, l2 in hidden_config_pairs]
        total_params_list.append(sum(num_params))
        test_losses_list.append(results[i]['loss_test'])
    
    plt.figure()
    plt.scatter(total_params_list, test_losses_list)
    plt.xlabel('Total number of parameters')
    plt.ylabel('Test loss')
    plt.title('Number of ANN parameters vs test loss')
    plt.show()


def plot_strain_histogram(df, params, testID, n_points=1000):

    # Extract params
    probabilistic_output = params['probabilistic_output']
    binning_params = params['binning_params']

    # Get data
    df = df[df['TestID'] == testID]
    # vf = round(df['Volume_Fraction'].values[0] * 100)

    # Get uniform x values
    x_values = df['EngStrain'].values

    # plot reference stress-strain curve
    fig, ax = plt.subplots(figsize=(3, 2.3))

    # Formatting
    from matplotlib import rcParams
    
    # Set standard font sizes globally using rcParams
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['axes.labelsize'] = 10        # Axis label font size
    rcParams['xtick.labelsize'] = 9       # X-tick label font size
    rcParams['ytick.labelsize'] = 9       # Y-tick label font size
    rcParams['legend.fontsize'] = 9      # Legend font size
    # Set standard line width and marker size globally
    rcParams['lines.linewidth'] = 0.75         # Line width for all lines
    rcParams['lines.markersize'] = 3       # Marker size for all markers
    rcParams['errorbar.capsize'] = 3        # Error bar cap size

    if probabilistic_output:
        stress_tmp = df['EngStress_mean'].values
    else:
        stress_tmp = df['EngStress'].values

    # Denormalise the output stress
    main_scaler_label = params['denormalisation_params']['main_scaler_label']
    main_scaler_norm_val_1 = params['denormalisation_params'][main_scaler_label]['norm_val_1']
    main_scaler_norm_val_2 = params['denormalisation_params'][main_scaler_label]['norm_val_2']

    if params['normalisation'] == 'minmax':
        stress_tmp = normalise_minmax(stress_tmp, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise=True)
    elif params['normalisation'] == 'gaussian':
        stress_tmp = normalise_gaussian(stress_tmp, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise=True)

    # ax.scatter(x_values, stress_tmp, label=f'Example curve', color='k', s=2, alpha=0.1)
    ax.plot(x_values, stress_tmp, label=f'Empirical Data', color='k')





    # plot histogram of strain values 
    # Create histogram and normalize it
    # hist_values, bin_edges = np.histogram(x_values, bins=500)
    # normalized_hist_values = hist_values / np.max(hist_values)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # ax.bar(bin_centers, normalized_hist_values, width=(bin_edges[1]-bin_edges[0]), color='blue', alpha=0.1, label='Normalized histogram')

    # ax Labels
    ax.set_ylabel('Engineering Stress (MPa)')
    ax.set_xlabel('Engineering Strain (-)')
    ax.set_ylim(0, max(stress_tmp) * 1.1)
    ax.set_xlim(0, 0.4)

    # Twin axis
    ax2 = ax.twinx()

    blue = '#1F77B4'
    # Set colors for each axis
    ax2.spines['right'].set_color(blue)  # Color for right y-axis
    # Set tick colors
    ax2.tick_params(axis='y', colors=blue) # Color for right y-axis ticks and labels
    # If you want to color the y-axis labels to match
    ax2.yaxis.label.set_color(blue)  # Color for right y-axis label

    # Plot non-uniform binning
    x_values_uniform = np.linspace(min(df['EngStrain']), max(df['EngStrain']), n_points)
    _, ax, x_vals, uniform_density_norm, x_vals_density_norm = generate_x_vals_for_bins(
        x_values_uniform,
        **binning_params,
        plot=True,
        ax=ax2,
    )

    # # Save numpy arrays to csv after converting to DataFrame
    # df_csv = pd.DataFrame({
    #     'EngStrain_example_curve': x_values[1:],
    #     'EngStress_mean_example_curve': stress_tmp[1:],
    #     'EngStrain_density_curve': x_vals,
    #     'Uniform_density_curve': uniform_density_norm,
    #     'Relative_density_curve': x_vals_density_norm,
    # })
    # df_csv.to_csv(f'relative_density_plot.csv', index=False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Create a single combined legend
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.text(0.02, 0.96, f'(a)', transform=plt.gca().transAxes, va='top', ha='left', fontweight='bold')

    # plt.savefig(f'data_aug_plot.png', dpi=300, bbox_inches='tight')


def plot_validation_predictions(data_df_train, params, val_ids_to_plot=None, hc_to_plot=None, export_to_csv=False):

    if val_ids_to_plot is None:
        val_ids_to_plot = params['validation_TestIDs']
    if not all(val_id in params['validation_TestIDs'] for val_id in val_ids_to_plot):
        raise ValueError(f'Some validation IDs to plot are not in the validation TestIDs list: {val_ids_to_plot}')
    
    # Extract validation predictions
    val_preds = params['val_preds']

    # test_preds is a 3-level nested list: [row][sublist][tensor]
    val_preds = torch.tensor([
        [tensor.cpu().numpy() for sublist in row for tensor in sublist]
        for row in val_preds
    ])

    n_val_preds = val_preds.shape[0]
    val_ids = params['validation_TestIDs']

    hc_list = params['hyper_grid']['candidate_hidden_configs']
    n_hc = len(hc_list)
    hc_str_list = ['-'.join(str(hc) for hc in hc) for hc in hc_list]
    hc_index = np.arange(0, n_val_preds, n_hc)

    val_preds_dict = {}
    for i, hc_str in enumerate(hc_str_list):
        index_list = hc_index + i
        val_preds_dict[hc_str] = val_preds[index_list, :, :]
    
    # If a specific hidden config is provided, plot only that config
    if hc_to_plot is not None:
        n_hc = len(hc_to_plot)
        val_preds_dict = {hc_to_plot[i]: val_preds_dict[hc_to_plot[i]] for i in range(n_hc)}
        hc_str_list = hc_to_plot

    colors = plt.cm.viridis(np.linspace(0, 1, n_hc)) # type: ignore
    line_styles = ['-', '--', ':', '-.']
    fig, ax = plt.subplots(figsize=(10, 6))
    df_csv = pd.DataFrame()

    # Plot true curves
    for i, val_id in enumerate(val_ids_to_plot):
        strain = data_df_train[data_df_train['TestID'] == val_id]['EngStrain']
        mean_stress_true = data_df_train[data_df_train['TestID'] == val_id]['EngStress_mean']
        ax.plot(strain, mean_stress_true, label=f'True: T{val_id}', color='k', linestyle=line_styles[i], alpha=0.7, linewidth=1)
        # for saving to csv
        df_csv = pd.concat([df_csv, pd.DataFrame({
            'TestID': val_id, 
            'hidden_config': 'N/A',
            'EngStrain': strain.values.tolist(), 
            'EngStress_mean': mean_stress_true.values.tolist()
        })], ignore_index=True)

    # Plot predicted curves
    counter = 0
    for hc_str, preds in val_preds_dict.items():
        color = colors[counter]
        counter += 1

        for i, val_id in enumerate(val_ids):
            if val_id not in val_ids_to_plot:
                continue
            # error_true = data_df_train[data_df_train['TestID'] == val_ids[i]]['EngStress_error']
            # error_lower_true = mean_stress_true - error_true
            # error_upper_true = mean_stress_true + error_true
            # plt.fill_between(strain, error_lower_true, error_upper_true, alpha=0.5)

            strain = data_df_train[data_df_train['TestID'] == val_id]['EngStrain']
            mean_stress_pred = preds[i, :, 0]

            label = f'{hc_str}: T{val_id}'
            ax.plot(strain, mean_stress_pred, label=label, linestyle=line_styles[i], color=color, alpha=1, linewidth=2)

            # error_pred = preds[i, :, 1]
            # error_lower = mean_stress_pred - error_pred
            # error_upper = mean_stress_pred + error_pred
            # ax.fill_between(strain, error_lower, error_upper, alpha=0.2, color=colors[i])

            # save to csv
            df_csv = pd.concat([df_csv, pd.DataFrame({
                'TestID': val_id, 
                'hidden_config': hc_str,
                'EngStrain': strain.values.tolist(), 
                'EngStress_mean': mean_stress_pred.detach().numpy().tolist()
            })], ignore_index=True)

    ax.legend()
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress')
    plt.show()

    if export_to_csv:
        df_csv.to_csv('validation_predictions_plot.csv', index=False)



# from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_3d_surface(model, data, strain_max, vf_max, stress_max, side_length, verbose, rstride, cstride, save_as_mat_file=False):

    params_tmp = data['params'].copy()

    # Checks
    if side_length > 1000:
        raise ValueError('Side length is too large for 3D surface plotting')
    if params_tmp['architecture'] != 'ANN':
        raise ValueError('Only ANN models are supported for 3D surface plotting')
    if params_tmp['normalisation'] != 'minmax':
        raise ValueError('Only minmax normalisation is supported for 3D surface plotting')
    if params_tmp['probabilistic_output'] is False:
        raise ValueError('Only probabilistic models are supported for 3D surface plotting')

    strain_range = np.linspace(0, strain_max, side_length)
    vf_range = np.linspace(0, vf_max, side_length)

    # Get params used to normalise data
    strain_norm_val_1 = params_tmp['denormalisation_params']['EngStrain']['norm_val_1']
    strain_norm_val_2 = params_tmp['denormalisation_params']['EngStrain']['norm_val_2']
    vf_norm_val_1 = params_tmp['denormalisation_params']['Volume_Fraction']['norm_val_1']
    vf_norm_val_2 = params_tmp['denormalisation_params']['Volume_Fraction']['norm_val_2']

    if verbose:
        print(f'Strain normalisation values: {strain_norm_val_1} {strain_norm_val_2}')
        print(f'VF normalisation values: {vf_norm_val_1} {vf_norm_val_2}')

    # Normalise input ranges
    if params_tmp['normalisation'] == 'minmax':
        strain_range_norm = normalise_minmax(strain_range, strain_norm_val_1, strain_norm_val_2)
        vf_range_norm     = normalise_minmax(vf_range,     vf_norm_val_1,     vf_norm_val_2)
    elif params_tmp['normalisation'] == 'gaussian':
        strain_range_norm = normalise_gaussian(strain_range, strain_norm_val_1, strain_norm_val_2)
        vf_range_norm     = normalise_gaussian(vf_range,     vf_norm_val_1,     vf_norm_val_2)

    # Make sure input labels are in the correct order
    input_mesh_dict = {
        'EngStrain': strain_range_norm,
        'Volume_Fraction': vf_range_norm
    }
    input_labels = params_tmp['input_labels']
    if 'EngStrain' not in input_labels or 'Volume_Fraction' not in input_labels or len(input_labels) != 2:
        print(input_labels)
        raise ValueError('This only works for models that used VF and Strain as input features currently')

    # Create all combinations and convert to tensor
    f1 = input_mesh_dict[input_labels[0]]
    f2 = input_mesh_dict[input_labels[1]]
    input_mesh = np.array(np.meshgrid(f1, f2)).T.reshape(-1, 2)
    input_mesh = torch.tensor(input_mesh, dtype=torch.float32).unsqueeze(0)  # shape: (1, N, 2)

    # Use the ANN to predict outputs for the entire mesh
    with torch.no_grad():
        output_mesh = model(input_mesh)

    # Denormalise the output stress
    main_scaler_label = params_tmp['denormalisation_params']['main_scaler_label']
    main_scaler_norm_val_1 = params_tmp['denormalisation_params'][main_scaler_label]['norm_val_1']
    main_scaler_norm_val_2 = params_tmp['denormalisation_params'][main_scaler_label]['norm_val_2']

    if params_tmp['normalisation'] == 'minmax':
        output_mesh = normalise_minmax(output_mesh, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise=True)
    elif params_tmp['normalisation'] == 'gaussian':
        output_mesh = normalise_gaussian(output_mesh, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise=True)

    # Extract stress values and reshape
    output_mesh_stress = output_mesh[0, :, 0]  # shape: (N,)
    stress_grid = output_mesh_stress.numpy().reshape(side_length, side_length)
    stress_grid_clipped = np.clip(stress_grid, 0, stress_max)
    stress_grid_clipped[stress_grid >= stress_max] = np.nan

    # Prepare meshgrids for plotting in 3D
    strain_mesh, vf_mesh = np.meshgrid(strain_range, vf_range)

    # Create figure/axes
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    cmap = plt.colormaps['viridis']

    surf = ax.plot_surface( # type: ignore
        strain_mesh,
        vf_mesh,
        stress_grid_clipped,
        cmap=cmap,
        edgecolor='none',
        alpha=0.9,
        rstride=rstride,
        cstride=cstride,
    )

    # Plot the true curves with a Line3DCollection
    data_df_final_results_tmp = data['data_df_final_results']

    # for i, TestID in enumerate(range(7)):
    #     true_test = data_df_final_results_tmp[data_df_final_results_tmp['TestID'] == TestID]
    #     true_strain = true_test['EngStrain'].values
    #     true_stress = true_test['EngStress_mean'].values
    #     true_vf = true_test['Volume_Fraction'].values

    #     # Build segments: shape => (n_segments, 2, 3) for the line
    #     points = np.array([true_strain, true_vf, true_stress]).T.reshape(-1, 1, 3)
    #     segments = np.concatenate([points[:-1], points[1:]], axis=1)

    #     # Create a black line collection
    #     lc = Line3DCollection(
    #         segments,
    #         colors='k',
    #         linewidth=2,
    #         label='True Curves' if i == 0 else None
    #     )
    #     ax.add_collection3d(lc)


    # Store true curves
    true_curves_list = []
    for TestID in range(7):
        true_test = data_df_final_results_tmp[data_df_final_results_tmp['TestID'] == TestID]
        true_strain = true_test['EngStrain'].values
        true_stress = true_test['EngStress_mean'].values
        true_vf = true_test['Volume_Fraction'].values
        
        # Store for mat file
        true_curves_list.append({
            'strain': true_strain,
            'stress': true_stress,
            'vf': true_vf
        })

        plt.plot(true_strain, true_vf, true_stress, color='k', label='True Curves')

    # surf.set_zsort('max')

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Stress [MPa]')

    # Axis labels
    ax.set_xlabel('Strain [-]', labelpad=15)
    ax.set_ylabel('VF [-]',     labelpad=15)
    # ax.set_zlabel('Stress [MPa]', labelpad=15)  # Uncomment if needed

    # Viewing angle
    ax.view_init(elev=15, azim=-250) # type: ignore
    # ax.view_init(elev=10, azim=-70)
    # ax.view_init(elev=30, azim=-120)

    # Add legend for the line collection
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.65, 0.8), fontsize=12, frameon=False)

    # Set axis limits
    ax.set_xlim(0, strain_max)
    ax.set_ylim(0, vf_max)
    ax.set_zlim(0, stress_max) # type: ignore

    # Optional layout
    plt.tight_layout()

    if save_as_mat_file:
        if not os.path.exists('report_files'):
            os.makedirs('report_files')
        file_path = 'report_files/surface_data.mat'
        if not os.path.exists(file_path):
            savemat(file_path, {
                'strain_mesh': strain_mesh,
                'vf_mesh': vf_mesh,
                'stress_grid_clipped': stress_grid_clipped,
                'stress_grid': stress_grid,
                'true_curves': true_curves_list,
                'stress_max': stress_max,
                'strain_max': strain_max,
                'vf_max': vf_max,
            })
        else:
            raise ValueError(f'{file_path} already exists')
    
    plt.show()




def plot_yield_strengths(
        model, 
        data, 
        strain_max, 
        vf_max, 
        n_points, 
        strain_threshold, 
        verbose,
        export_to_csv=False, 
        import_best_and_worst=False, 
        filename=None,
        get_densification_data=False
    ):


    def plot_yield_strengths_single(ax, df):
        yield_strength = df['Yield_Strength']
        ax.plot(df.index, yield_strength, label='Yield Strength')
        yield_strength_error = df['Yield_Strength_error']
        error_lower = yield_strength - yield_strength_error/2
        error_upper = yield_strength + yield_strength_error/2
        ax.fill_between(df.index, error_lower, error_upper, alpha=0.2)

        return ax

    # start timer
    start_time = time.time()
    params_tmp = data['params'].copy()

    # Checks
    if n_points < 100:
        raise ValueError('n_points is too small to calculate sufficient strain values.')
    if n_points > 10000:
        raise ValueError('n_points is too large.')
    if params_tmp['architecture'] != 'ANN':
        raise ValueError('Only ANN models are supported.')
    if params_tmp['probabilistic_output'] is False:
        raise ValueError('Only probabilistic models are supported.')

    if export_to_csv and import_best_and_worst:
        raise ValueError('Only one of export_to_csv and import_best_and_worst can be True.')
    if import_best_and_worst and filename is None:
        raise ValueError('filename must be provided if import_best_and_worst is True.')  
    if import_best_and_worst and get_densification_data:
        raise ValueError('get_densification_data must be False if import_best_and_worst is True.')

    if not import_best_and_worst:

        strain_range = np.linspace(0, strain_max, n_points)
        vf_range = np.linspace(0, vf_max, n_points)

        # Get params used to normalise data
        strain_norm_val_1 = params_tmp['denormalisation_params']['EngStrain']['norm_val_1']
        strain_norm_val_2 = params_tmp['denormalisation_params']['EngStrain']['norm_val_2']
        vf_norm_val_1 = params_tmp['denormalisation_params']['Volume_Fraction']['norm_val_1']
        vf_norm_val_2 = params_tmp['denormalisation_params']['Volume_Fraction']['norm_val_2']

        if verbose:
            print(f'Strain normalisation values: {strain_norm_val_1} {strain_norm_val_2}')
            print(f'VF normalisation values: {vf_norm_val_1} {vf_norm_val_2}')

        # Normalise input ranges
        if params_tmp['normalisation'] == 'minmax':
            strain_range_norm = normalise_minmax(strain_range, strain_norm_val_1, strain_norm_val_2)
            vf_range_norm     = normalise_minmax(vf_range,     vf_norm_val_1,     vf_norm_val_2)
        elif params_tmp['normalisation'] == 'gaussian':
            strain_range_norm = normalise_gaussian(strain_range, strain_norm_val_1, strain_norm_val_2)
            vf_range_norm     = normalise_gaussian(vf_range,     vf_norm_val_1,     vf_norm_val_2)

        # Make sure input labels are in the correct order
        input_labels = params_tmp['input_labels']
        if 'EngStrain' not in input_labels or 'Volume_Fraction' not in input_labels or len(input_labels) != 2:
            print(input_labels)
            raise ValueError('This only works for models that used VF and Strain as input features currently')

        # Ensure input labels are in correct order
        input_dict = {
            'EngStrain': strain_range_norm,
            'Volume_Fraction': vf_range_norm
        }
        f1 = input_dict[input_labels[0]]
        f2 = input_dict[input_labels[1]]
        

        # Create all combinations and convert to tensor
        input = np.array(np.meshgrid(f1, f2)).T.reshape(-1, 2)
        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)  # shape: (1, N, 2)

        # Use the ANN to predict outputs for the entire mesh
        with torch.no_grad():
            output = model(input)
            

        # Denormalise the output stress
        main_scaler_label = params_tmp['denormalisation_params']['main_scaler_label']
        main_scaler_norm_val_1 = params_tmp['denormalisation_params'][main_scaler_label]['norm_val_1']
        main_scaler_norm_val_2 = params_tmp['denormalisation_params'][main_scaler_label]['norm_val_2']
        # get location of Volume_Fraction in input_labels
        vf_index = input_labels.index('Volume_Fraction')
        strain_index = input_labels.index('EngStrain')
        input_strain = input[0, :, strain_index]
        input_vf = input[0, :, vf_index]

        if params_tmp['normalisation'] == 'minmax':
            input_strain = normalise_minmax(input_strain, strain_norm_val_1, strain_norm_val_2, denormalise=True)
            input_vf = normalise_minmax(input_vf, vf_norm_val_1, vf_norm_val_2, denormalise=True)
            output = normalise_minmax(output, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise=True)
        elif params_tmp['normalisation'] == 'gaussian':
            input_strain = normalise_gaussian(input_strain, strain_norm_val_1, strain_norm_val_2, denormalise=True)
            input_vf = normalise_gaussian(input_vf, vf_norm_val_1, vf_norm_val_2, denormalise=True)
            output = normalise_gaussian(output, main_scaler_norm_val_1, main_scaler_norm_val_2, denormalise=True)

        # Extract stress values and reshape
        output_stress = output[0, :, 0]  # shape: (N,)
        output_error = output[0, :, 1]  # shape: (N,)

        df_results = pd.DataFrame({
            'EngStrain': input_strain,
            'Volume_Fraction': input_vf,
            'EngStress_mean': output_stress,
            'EngStress_error': output_error
        })
        # Create Fake TestIDs: Make array of unique integers based on volume fraction input[0, :, vf_index]
        df_results['TestID'], _ = pd.factorize(df_results['Volume_Fraction'])
        df_results.set_index('TestID', inplace=True)

        if get_densification_data:
            print('INFO: Setting strain_max to 0.8 for densification data.')
            strain_max = 0.8
            print('INFO: Setting filename to densification_data.csv')
            filename = 'densification_data.csv'
            if os.path.exists(filename):
                raise ValueError(f'File {filename} already exists.')

            df_results.to_csv(filename)
            df_results.info()
            return
        else:
            # remove all rows where EngStrain > strain_threshold
            df_results_clipped = df_results[df_results['EngStrain'] < strain_threshold].copy()

            # get max EngStress_mean
            df_yield_strengths = pd.DataFrame()
            for vf, group_vf in df_results_clipped.groupby('Volume_Fraction'):
                df_yield_strengths.loc[vf, 'Yield_Strength'] = group_vf['EngStress_mean'].max()
                df_yield_strengths.loc[vf, 'Yield_Strength_error'] = group_vf['EngStress_error'].max()

            # Export to csv
            if export_to_csv:
                if filename is None:
                    filename = 'yield_strengths.csv'
                if os.path.exists(filename):
                    raise ValueError(f'File {filename} already exists.')
                df_yield_strengths.to_csv(filename)                    

            # plot
            fig, ax = plt.subplots()
            ax = plot_yield_strengths_single(ax, df_yield_strengths)

    else:
        df_yield_strengths_best = pd.read_csv('yield_strengths_best.csv')
        df_yield_strengths_worst = pd.read_csv('yield_strengths_worst.csv')

        # plot
        fig, ax = plt.subplots()
        for df_yield_strengths in [df_yield_strengths_best, df_yield_strengths_worst]:
            ax = plot_yield_strengths_single(ax, df_yield_strengths)
    
    ax.legend()
    plt.show()

    # end timer
    end_time = time.time()
    print(f'Time taken: {end_time - start_time} seconds')
