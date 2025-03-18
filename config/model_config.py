"""
Basic model configuration settings.
"""

MODELS = {
    "llama3.1-8b-instruct": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    "llama2-7b-chat": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "tokenizer_id": "meta-llama/Llama-2-7b-chat-hf",
    },
    "mistral-7b-instruct-v0.3": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "tokenizer_id": "mistralai/Mistral-7B-Instruct-v0.3",
    }
}

# LoRA configuration
LORA_CONFIG = {
    "r": 16,                     # Rank of the update matrices
    "lora_alpha": 32,            # Alpha parameter for LoRA scaling
    "lora_dropout": 0.05,        # Dropout probability for LoRA layers
    "bias": "none",              # Add bias to the output
    "task_type": "CAUSAL_LM",    # Task type
    "target_modules": [          # Modules to apply LoRA to
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
}

# Training configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 100,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "max_length": 1024,        # Maximum sequence length
    "load_in_8bit": True,      # Load model in 8-bit precision
} 