import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import importlib.util
from src.models import resnext_real_vs_ai
from src.models import resnext_real_vs_ai_transformer
from torchview import draw_graph
from torchinfo import summary

def plot_model_architecture(
    model,
    save_path=None,
    show_shapes=True,
    expand_nested=True,
    dpi=100
):
    """
    Plots the architecture of a PyTorch model using torchinfo.summary and torchview.draw_graph.
    If torchview fails, falls back on torchviz.make_dot.

    Args:
        model:         A wrapper or nn.Module.
        save_path:     If provided, saves both the PNG and a "_summary.txt" here.
        show_shapes:   Whether to include input/output shapes in the text summary.
        expand_nested: Whether to expand nested modules in the text summary.
        dpi:           DPI for the saved figure.
    """
    nn_model = getattr(model, 'model', model).cpu().eval()

    # 1) Text summary
    cols = ("input_size", "output_size", "num_params") if show_shapes else ("num_params",)
    txt = summary(
        nn_model,
        input_size=(1, 3, 224, 224),
        col_names=cols,
        depth=(10 if expand_nested else 1),
        verbose=1
    )
    print(txt)
    if save_path:
        with open(os.path.splitext(save_path)[0] + "_summary.txt", "w") as f:
            f.write(str(txt))

    # 2) Try torchview first
    try:
        graph = draw_graph(
            nn_model,
            input_size=(1, 3, 224, 224),
            expand_nested=expand_nested,
            roll=True,
            graph_name="Model Architecture"
        )
        fig = graph.visual_graph
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        return
    except Exception as e:
        print(f"[torchview] failed: {e}\nFalling back to torchviz…")

    # 3) Fallback: torchviz
    try:
        from torchviz import make_dot
    except ImportError:
        raise ImportError(
            "Both torchview failed and torchviz is not installed. "
            "Install torchviz (`pip install torchviz`) and retry."
        )

    dummy = torch.randn(1, 3, 224, 224)
    out = nn_model(dummy)
    dot = make_dot(out, params=dict(nn_model.named_parameters()))
    dot.format = "png"

    base = os.path.splitext(save_path)[0] if save_path else "model_architecture"
    dot_path = dot.render(base, cleanup=True)  # writes base.png

    img = plt.imread(dot_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
        
def main():
    og_resnext_no_transformer = 'runs/model_resnext_real_vs_ai_mod_resNeXt_ft_0.9582.pt'
    resnext_transformer = 'runs/model_resnext_real_vs_ai_mod_resNeXt_transformer_ft_0.9342.pt'
    
    og_resnext_no_transformer_config = 'configs/mod_resNeXt_ft.py'
    
    spec = importlib.util.spec_from_file_location("config", og_resnext_no_transformer_config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    model_params = config.model_params
    
    model = resnext_real_vs_ai.ResNeXtRealVsAI(model_params)
    
    model.load_checkpoint(og_resnext_no_transformer)
    
    plot_model_architecture(model)
    
    og_resnext_no_transformer_config = 'configs/mod_resNeXt_transformer_ft.py'
    
    spec = importlib.util.spec_from_file_location("config", og_resnext_no_transformer_config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    model_params = config.model_params
    
    model = resnext_real_vs_ai_transformer.ResNeXtRealVsAITransformer(model_params)
    
    model.load_checkpoint(resnext_transformer)
    
    plot_model_architecture(model)
    
main()