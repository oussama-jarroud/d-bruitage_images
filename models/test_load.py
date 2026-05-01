import torch
import os
import sys
import importlib.util
import traceback

def load_model():
    base_dir = r"c:\Users\jarro\OneDrive\Desktop\Projet2_Debruitage\models"
    model_path = os.path.join(base_dir, "eu2net_best_model.pth")
    script_path = os.path.join(base_dir, "udiat-busi-improv.py")
    
    try:
        spec = importlib.util.spec_from_file_location("udiat", script_path)
        udiat = importlib.util.module_from_spec(spec)
        sys.modules["udiat"] = udiat
        spec.loader.exec_module(udiat)
        
        IMG_HEIGHT = udiat.IMG_HEIGHT
        IMG_WIDTH = udiat.IMG_WIDTH
        IMG_CHANNELS = udiat.IMG_CHANNELS
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        print("Building model...")
        model = udiat.build_u2net(input_shape).to(DEVICE)
        
        print("Loading state dict...")
        state = torch.load(model_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        
        new_state = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_key] = v
        model.load_state_dict(new_state, strict=False)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print("Error:")
        traceback.print_exc()

load_model()
