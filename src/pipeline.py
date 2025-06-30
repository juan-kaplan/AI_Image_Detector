from src.utils.files import create_result_folder
from src.models import resnext_real_vs_ai
from src.models import resnext_real_vs_ai_transformer
# from src.models import mcf_features_nn
import pandas as pd
from src.utils.build_real_vs_ai import build
from src.utils.retrieve_datasets import build_datasets

# from src.train_test.train import train
# from src.train_test.test import test

# def check_device(device):
#     """
#     Checks if the specified device is 'cuda' and sets it to CUDA if available, otherwise defaults to CPU.

#     Args:
#     device (str): Device name, typically 'cuda' or 'cpu'.

#     Returns:
#     torch.device: Torch device object representing the selected device.
#     """
#     if device == 'cuda':
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return device

def get_class_names(csv_path, column='season'):
    """
    Read the CSV dataset and extract sorted unique class names from the specified column.
    """
    df = pd.read_csv(csv_path)
    return sorted(df[column].unique())

def load_model(model_params):
    """
    Load a machine learning model based on the specified model name in model_params.

    Args:
    model_params (dict): Parameters containing the 'model_name' key to determine which model to load.

    Returns:
    object: Loaded machine learning model instance corresponding to the specified model_name.
    """


    model_name = model_params['model_name']
    model = None
    
    if model_name == 'resnext_real_vs_ai':
        model = resnext_real_vs_ai.ResNeXtRealVsAI(model_params)
        
    elif model_name == 'resnext_real_vs_ai_transformer':
        model = resnext_real_vs_ai_transformer.ResNeXtRealVsAITransformer(model_params)
    return model, ['real', 'fake']


# def run_model(model_params, data_params):
#     """
#     Run the machine learning model training and evaluation pipeline.

#     Args:
#     model_params (dict): Parameters related to the machine learning model configuration.
#     data_params (dict): Parameters related to the dataset configuration.

#     Returns:
#     dict: A dictionary containing paths to saved results ('save_path' and 'save_cv_path').
#     """
#     save_path = create_result_folder(model_params, data_params)
    
#     model, names = load_model(model_params)
    
#     # Prepare dataloaders using the new method
#     train_loader = model.create_dataloader(
#         names,
#         data_params['data_train_path'],
#         batch_size=model.batch_size_tr,
#         num_workers=model.num_workers,
#         test=False,
#         shuffle=True
#     )
#     val_loader = model.create_dataloader(
#         names,
#         data_params['data_val_path'],
#         batch_size=model.batch_size_va,
#         num_workers=model.num_workers,
#         test=True,
#         shuffle=False
#     )
#     save_name = model_params.get('configs_file_name', None)
#     model.fit(train_loader, val_loader, save_name=save_name)
#     # Optionally, you can add evaluation logic here
#     return save_path

def run_model(model_params, data_params):
    save_path = create_result_folder(model_params, data_params)
    model, _  = load_model(model_params)

    datasets = build_datasets()      # <- returns DatasetDict with train/val/test
    train_loader = model.create_dataloader(None, datasets["train"],
                                           batch_size=model.batch_size_tr,
                                           num_workers=model.num_workers,
                                           shuffle=True)
    val_loader   = model.create_dataloader(None, datasets["validation"],
                                           batch_size=model.batch_size_va,
                                           num_workers=model.num_workers,
                                           shuffle=False)

    model.fit(train_loader, val_loader,
              save_name=model_params.get("configs_file_name"))

    return save_path


def test_model(model_params, data_params, model_path, test_dataset_path, seasons_only=False, topk=None):
    model, names = load_model(model_params)
    save_path = create_result_folder(model_params, data_params)
    if isinstance(model, resnext_real_vs_ai.ResNeXtRealVsAI):
        # Prepare dataloader using the new method
        test_loader = model.create_dataloader(
            names,
            test_dataset_path,
            batch_size=model.batch_size_va,
            num_workers=model.num_workers,
            test=True,
            shuffle=False
        )
        model.load_weights(model_path)
        model.evaluate(test_loader, save_path=save_path)
        pass

def test_single_image(model_params, data_params, model_path, img_path):
    model, names = load_model(model_params)
    model.load_model_for_inference(model_path)
    prediction = model.predict_image(img_path)
    print(f"Image predicted as: {prediction}")
