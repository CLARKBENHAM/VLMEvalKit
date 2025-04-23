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
    # Llama 3.2 Vision models
    "Llama-3.2-11B-Vision": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    "Llama-3.2-11B-Vision-Instruct": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    "Llama-3.2-90B-Vision": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "load_in_4bit": True,  # 4-bit quantization for massive models
        "attn_implementation": "eager",
        "use_cache": True
    },

    "Llama-3.2-90B-Vision-Instruct": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "load_in_4bit": True,  # 4-bit quantization for massive models
        "attn_implementation": "eager",
        "use_cache": True
    },

    # Gemma additional models
    "gemma-3-27b-it-qat-q4_0-gguf": {
        "use_safetensors": False,  # GGUF files aren't safetensors
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    # Qwen models
    "Qwen2.5-VL-72B-Instruct": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "load_in_8bit": True,  # 8-bit quantization for large models
        "attn_implementation": "eager",
        "use_cache": True
    },

    "Qwen2.5-VL-32B-Instruct": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "load_in_8bit": True,
        "attn_implementation": "eager",
        "use_cache": True
    },

    "Qwen2.5-VL-7B-Instruct": {
        "torch_dtype": torch.float16,  # fp16 is sufficient for 7B
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    "Qwen2.5-Omni-7B": {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    # Microsoft Phi models
    "Phi-3.5-vision-instruct": {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    "Phi-3.5-vision": {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    # Specialized models
    "ChemVLM-26B": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "load_in_8bit": True,
        "attn_implementation": "eager",
        "use_cache": True
    },

    "GOT-OCR2_0": {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    "InternVL2-8B": {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },
    "InternVL2-26B": {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },
    "InternVL2-40B": {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    "MiniCPM-Llama3-V-2_5": {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "use_cache": True
    },

    # Add any other Ovis models with float32 requirement
    "Ovis2-7B": {
        "torch_dtype": torch.float32,
        "device_map": "auto",
        "attn_implementation": "eager"
    },

    "Ovis2-12B": {
        "torch_dtype": torch.float32,
        "device_map": "auto",
        "attn_implementation": "eager"
    }

}

# Set PyTorch memory environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
