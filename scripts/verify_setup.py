import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_verification():
    print("="*50)
    print("Running Repository Verification")
    print("="*50)

    # 1. Check Imports
    print("[1/4] Checking Imports...")
    try:
        import src.pipeline
        import src.models.resnext_real_vs_ai
        print("  ✅ Imports Successful")
    except ImportError as e:
        print(f"  ❌ Import Failed: {e}")
        return

    # 2. Check Model Checkpoints
    print("\n[2/4] Checking Model Checkpoints...")
    checkpoint_path = "models/checkpoints/resnext.pt"
    if os.path.exists(checkpoint_path):
        print(f"  ✅ Found checkpoint: {checkpoint_path}")
    else:
        print(f"  ⚠️ Checkpoint not found at {checkpoint_path}. Skipping model loading test.")
        # Try to find any other .pt file
        avail_pts = [f for f in os.listdir("models/checkpoints") if f.endswith(".pt")]
        if avail_pts:
            checkpoint_path = os.path.join("models/checkpoints", avail_pts[0])
            print(f"  ℹ️ Using alternative checkpoint: {checkpoint_path}")
        else:
            print("  ❌ No checkpoints found. Cannot test model loading.")
            return

    # 3. Model Loading
    print("\n[3/4] Testing Model Loading...")
    try:
        model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # Handle case where model is wrapped or raw dictionary
        if hasattr(model, 'eval'):
            model.eval()
        elif isinstance(model, dict) and 'model' in model:
             # This assumes it might be a state dict or a wrapper, simpler to just check if we can load it
             print("  ℹ️ Loaded checkpoint dictionary.")
        elif hasattr(model, 'model'):
             model.model.eval()
             model = model.model
        
        print("  ✅ Model Loaded Successfully")
    except Exception as e:
        print(f"  ❌ Model Loading Failed: {e}")
        return

    # 4. Inference Test (Dummy Input)
    print("\n[4/4] Testing Inference with Dummy Input...")
    try:
        # Create dummy image (3 channels, 224x224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
        
        print(f"  ✅ Inference Successful. Prediction: {pred}")
    except Exception as e:
        print(f"  ❌ Inference Failed: {e}")
        return

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE: Repository is functional.")
    print("="*50)

if __name__ == "__main__":
    run_verification()
