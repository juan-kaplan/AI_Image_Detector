import argparse

from src.utils.files import read_configs
from src.pipeline import run_model, test_model

# Como correrlo:
#   python main.py --configs configs/mod_ensamble1.py 
#python main.py --configs configs/mod_resNeXt_weighted_avg_DeepArmocromia_only_season.py --path_test data/split_dataset/DeepArmocromia_season_only/test_DeepArmocromia_season_only.csv --path_model runs/model_mod_resNeXt_weighted_avg_DeepArmocromia_only_season.pt
#python main.py --configs configs/mod_resNeXt_weighted_avg_DeepArmocromia2.py --path_test data/split_dataset/DeepArmocromia/test_DeepArmocromia.csv --path_model runs/model_mod_resNeXt_weighted_avg_DeepArmocromia2.pt
# python main.py --configs configs/mod_resNeXt_weighted_avg_super2.py --path_test data/split_dataset/SuperDataset/test_SuperDataset.csv --path_model runs/model_mod_resNeXt_weighted_avg_super2.pt
# python main.py --configs configs/mod_resNeXt_weighted_avg_super2.py --path_test data/split_dataset/SuperDataset/val_SuperDataset.csv --path_model runs/model_mod_resNeXt_weighted_avg_super2.pt --seasons_only 
# python main.py --configs configs/mod_resNeXt_weighted_avg_super2.py --path_test data/split_dataset/SuperDataset/val_SuperDataset.csv --path_model runs/model_mod_resNeXt_weighted_avg_super2.pt --topk 2
# python main.py --configs .\configs\mod_resNeXt_ft.py
# python main.py --configs .\configs\mod_resNeXt_ft.py --path_test .\data\splits\test.csv --path_model .\runs\model_mod_resNeXt_ft.pt
# python main.py --configs .\configs\mod_resNeXt_ft.py --path_test .\data\splits\test.csv --path_model .\runs\model_resnext_real_vs_ai_mod_resNeXt_ft_0.9582.pt
# python main.py --configs .\configs\mod_resNeXt_ft.py --path_test .\data\test_data\csvs\test_data.csv --path_model .\runs\model_resnext_real_vs_ai_mod_resNeXt_ft_0.9582.pt

def main(args):
    check_args = read_configs(args)
    
    if check_args == -1:
        print("Some parameters were missing.")
        return -1
    
    elif check_args[0] == 0:
        _, model, data = check_args
        run_model(model, data)

    else:
        _, model, data, model_path, test_dataset_path = check_args
        seasons_only = getattr(args, "seasons_only", False)
        topk = getattr(args, "topk", None)
        test_model(model, data, model_path, test_dataset_path, seasons_only=seasons_only, topk=topk)
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, help="Should add path of the configs file", required=False)
    parser.add_argument("--path_test", type=str, help="Should add path to test dataset", required=False)
    parser.add_argument("--path_model", type=str, help="Should add path to saved model", required=False)

    args = parser.parse_args()
    main(args)