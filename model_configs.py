# model_configs.py
import os
import torch

# Global configuration that applies to all models
MODEL_CONFIGS = {
    # Default settings for all models
    "default": {
        "torch_dtype": torch.bfloat16,  # bf16 for most models
        "device_map": "auto",
        "attn_implementation": "eager",  # Disable FlashAttention
        "use_flash_attention_2": False
    },
    
    # Model-specific configurations
    "Gemma3-4B": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },
    
    "Gemma3-12B": {
        "torch_dtype": torch.bfloat16, 
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },
    
    "Gemma3-27B": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "load_in_8bit": True,  # 8-bit quantization for memory efficiency
        "attn_implementation": "eager"
    },
    
    "Ovis1.6-Llama3.2-3B": {
        "torch_dtype": torch.float32,  # Force float32 to fix dtype mismatch
        "device_map": "auto",
        "attn_implementation": "eager"
    },
    
    # Add configurations for other models as needed
}

# Set PyTorch memory environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
