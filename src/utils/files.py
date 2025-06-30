import os
import importlib.util
import json
import pandas as pd

def read_configs(args):
    """
    Read configuration parameters from a Python file specified by the command-line arguments.

    Args:
    args (argparse.Namespace): Command-line arguments parsed using argparse.

    Returns:
    list or int: A list containing model_params, dataset_params, features_params, and cv_params if configuration file is provided; 
                 or a list containing path_test and device if path_test and device arguments are provided; 
                 or -1 if neither configurations nor path_test and device arguments are provided.
    """
    if args.configs:
        path_config = args.configs
        spec = importlib.util.spec_from_file_location("config", path_config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        model_params = config.model_params
        dataset_params = config.dataset_params
        if args.path_test and args.path_model: 
            return ['test', model_params, dataset_params, args.path_model, args.path_test]
        elif args.path_image and args.path_model:
            return ['test_single_image', model_params, dataset_params, args.path_model, args.path_image]
        
        dataset_params = config.dataset_params
        return ['run', model_params, dataset_params]

    else:
        return -1


def create_result_folder(model_params, data_params):
    """
    Create a folder structure for storing results based on provided parameters.

    Args:
    model_params (dict): Parameters related to the model, including 'model_name' and 'configs_file_name'.
    data_params (dict): Parameters related to the dataset, including 'norm' and 'dataset_name'.
    features_params (dict): Parameters related to features, including 'num_features'.
    cv (bool, optional): Whether to create a subfolder for cross-validation results. Default is False.

    Returns:
    tuple: A tuple containing the main path where results will be stored and the path for cross-validation results (if cv=True).
    """
    path = ('results/' + model_params['model_name'] + '/' + 
            model_params['configs_file_name'])

    os.makedirs(path, exist_ok=True)
    return path

def save_predictions_csv(filenames, predicted_labels, results_folder):
    """
    Save a CSV with columns: filename, predicted_label in the results folder.
    """
    df = pd.DataFrame({
        'filename': filenames,
        'predicted_label': predicted_labels
    })
    df.to_csv(os.path.join(results_folder, 'predictions.csv'), index=False)
    
