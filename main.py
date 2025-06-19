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


'''
el argumento de top3 permite tomar una prediccion como correcta si la etiqueta verdadera 
esta entre las 3 etiquetas con mayor probabilidad predicha por el modelo.
'''
def main(args):
    check_args = read_configs(args)
    
    if check_args == -1:
        print("Some parameters were missing.")
        return -1
    
    elif check_args[0] == 0:
        _, model, data = check_args
        run_model(model, data)

    else:
        _, model, model_path, test_dataset_path = check_args
        seasons_only = getattr(args, "seasons_only", False)
        topk = getattr(args, "topk", None)
        test_model(model, model_path, test_dataset_path, seasons_only=seasons_only, topk=topk)
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, help="Should add path of the configs file", required=False)
    parser.add_argument("--path_test", type=str, help="Should add path to test dataset", required=False)
    parser.add_argument("--path_model", type=str, help="Should add path to saved model", required=False)
    parser.add_argument("--seasons_only", action="store_true", help="If set, test only on seasons (not subcategories)")
    parser.add_argument("--topk", type=int, help="If set, mark prediction as true if the correct label is in the top k predicted probabilities")
    #parser.add_argument("--device", type=str, help="Should add cuda or cpu", required=False)
    args = parser.parse_args()
    main(args)