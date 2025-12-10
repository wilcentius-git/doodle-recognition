import torch
from src.config import DEVICE, MODEL_SAVE_PATH
from src.models_dl import TunedMLP, TunedCNN, get_resnet18, get_mobilenet_v2

def load_models_for_inference():
    """Memuat semua model DL, ML, dan Scaler dari checkpoint"""
    print(f"Loading models from: {MODEL_SAVE_PATH}")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    dl_models = {
        "MLP (Dropout)": TunedMLP(num_classes).to(DEVICE),
        "CNN (BatchNorm)": TunedCNN(num_classes).to(DEVICE),
        "ResNet18": get_resnet18(num_classes).to(DEVICE),
        "MobileNetV2": get_mobilenet_v2(num_classes).to(DEVICE)
    }
    
    for name in dl_models:
        dl_models[name].load_state_dict(checkpoint[name])
        dl_models[name].eval()
    
    ml_models = checkpoint['ml_models']
    scaler = checkpoint['scaler']
    
    return dl_models, ml_models, scaler, classes
