import torch
from src.config import DEVICE, MODEL_SAVE_PATH
from src.models_dl import TunedCNN, get_resnet18

def load_models_for_inference():
    """Memuat model CNN, ResNet18, dan Random Forest"""
    print(f"Loading models from: {MODEL_SAVE_PATH}")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    # Hanya load CNN dan ResNet
    dl_models = {
        "CNN (BatchNorm)": TunedCNN(num_classes).to(DEVICE),
        "ResNet18": get_resnet18(num_classes).to(DEVICE)
    }
    
    for name in dl_models:
        dl_models[name].load_state_dict(checkpoint[name])
        dl_models[name].eval()
    
    ml_models = checkpoint['ml_models']
    # Scaler dihapus karena RF tidak butuh
    
    return dl_models, ml_models, classes
