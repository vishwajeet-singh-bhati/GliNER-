import os
import torch
import torch.nn as nn
from e8_quant import E8QATLinear

def replace_linears_with_e8qat(module: nn.Module, min_features: int = 64, scaling_mode: str = "block_max", learnable: bool = True) -> int:
    n_replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if child.in_features >= min_features and child.out_features >= 2:
                setattr(module, name, E8QATLinear.from_linear(child, scaling_mode=scaling_mode, learnable=learnable))
                n_replaced += 1
        else:
            n_replaced += replace_linears_with_e8qat(child, min_features, scaling_mode=scaling_mode, learnable=learnable)
    return n_replaced

def get_backbone_encoders(gliner_model) -> list:
    token_rep_layer = gliner_model.model.token_rep_layer
    encoders = []
    bert_layer = getattr(token_rep_layer, "bert_layer", None)
    if bert_layer is not None:
        encoders.append(("text_encoder", bert_layer.model.encoder))
    labels_encoder = getattr(token_rep_layer, "labels_encoder", None)
    if labels_encoder is not None:
        encoders.append(("labels_encoder", labels_encoder.model.encoder))
    
    if not encoders:
        raise AttributeError("Could not find bert_layer or labels_encoder under model.token_rep_layer.")
    return encoders

def quantize_gliner_encoder(gliner_model, scaling_mode: str = "block_max", learnable: bool = True, min_features: int = 64) -> int:
    total = 0
    for name, encoder in get_backbone_encoders(gliner_model):
        n = replace_linears_with_e8qat(encoder, min_features=min_features, scaling_mode=scaling_mode, learnable=learnable)
        print(f"  quantized {n} Linear layers in {name}")
        total += n
    return total

def load_quantized_gliner(base_model_name: str, checkpoint_dir: str, scaling_mode: str = "block_max", learnable: bool = True, min_features: int = 64):
    from gliner import GLiNER
    model = GLiNER.from_pretrained(base_model_name, local_files_only=True)
    n = quantize_gliner_encoder(model, scaling_mode=scaling_mode, learnable=learnable, min_features=min_features)
    print(f"Reconstructed quantized architecture ({n} layers) before loading checkpoint weights")
    
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    pytorch_bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif os.path.exists(pytorch_bin_path):
        state_dict = torch.load(pytorch_bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Found neither model.safetensors nor pytorch_model.bin in {checkpoint_dir}.")
        
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    return model
