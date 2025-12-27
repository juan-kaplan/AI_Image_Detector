import argparse
import sys
from src.utils.files import read_configs
from src.pipeline import run_model, test_model, test_single_image

def main(args):
    """
    Main entry point for calculating AI vs Real image detection.
    
    See docs/USAGE.md for detailed running instructions.
    """
    check_args = read_configs(args)
    
    if check_args == -1:
        print("Error: Missing parameters. Please check your arguments.")
        sys.exit(1)
    
    elif check_args[0] == 'run':
        _, model, data = check_args
        run_model(model, data)

    elif check_args[0] == 'test':
        _, model, data, model_path, test_dataset_path = check_args
        seasons_only = getattr(args, "seasons_only", False)
        topk = getattr(args, "topk", None)
        test_model(model, data, model_path, test_dataset_path, seasons_only=seasons_only, topk=topk)
    
    elif check_args[0] == 'test_single_image':
        _, model_params, data_params, model_path, test_image_path = check_args
        test_single_image(model_params, data_params, model_path, test_image_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Image Detector - Traing and Inference CLI")
    
    parser.add_argument("--configs", type=str, help="Path to the configuration file (Python script)", required=False)
    parser.add_argument("--path_test", type=str, help="Path to the test dataset CSV", required=False)
    parser.add_argument("--path_model", type=str, help="Path to the saved model (.pt file)", required=False)
    parser.add_argument("--path_image", type=str, help="Path to a single image for testing", required=False)
    parser.add_argument("--seasons_only", action="store_true", help="Flag for seasons-only validation")
    parser.add_argument("--topk", type=int, help="Top-k accuracy evaluation", required=False)

    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    main(args)