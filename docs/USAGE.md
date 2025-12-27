# Usage Guide

This project allows you to train and test ResNeXt models for AI vs Real image detection.

## Commands

Run the following commands from the project root.

### Training / Evaluation with Configs
```bash
# ResNeXt Fine-tuning
python main.py --configs ./configs/mod_resNeXt_ft.py
```

### Testing Manually Split Data
```bash
# Test set
python main.py \
  --configs ./configs/mod_resNeXt_ft.py \
  --path_test ./data/splits/test.csv \
  --path_model ./models/checkpoints/model_mod_resNeXt_ft.pt

# Test with specific model checkpoint
python main.py \
  --configs ./configs/mod_resNeXt_ft.py \
  --path_test ./data/splits/test.csv \
  --path_model ./models/checkpoints/model_resnext_real_vs_ai_mod_resNeXt_ft_0.9582.pt

# Test with alternate data path
python main.py \
  --configs ./configs/mod_resNeXt_ft.py \
  --path_test ./data/test_data/csvs/test_data.csv \
  --path_model ./models/checkpoints/model_resnext_real_vs_ai_mod_resNeXt_ft_0.9582.pt
```

### Single Image Inference
```bash
python main.py \
  --configs configs/mod_resNeXt_transformer_2_ft.py \
  --path_model models/checkpoints/model_resnext_real_vs_ai_mod_resNeXt_transformer_2_ft_0.9423.pt \
  --path_image data/test_data/images/videoframe_5533.png
```
