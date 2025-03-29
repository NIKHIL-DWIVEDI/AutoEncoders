from models.convolutionalAE import CAE
from models.variationalAE import VAE
# Add more imports as you create more models

# Model registry for factory pattern
MODEL_REGISTRY = {
    'cae': CAE,
    'dae':CAE,
    'vae':VAE
    # 'vae': VAE,  # future implementation
}

def get_model(model_name, **kwargs):
    """Get model by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)