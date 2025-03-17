"""
Configuration settings for model finetuning.
"""

MODELS = {
    "llama3.1-8b": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B",
        "tokenizer_id": "meta-llama/Meta-Llama-3.1-8B",
    },
    "llama3.1-8b-instruct": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    # Can add other models here as needed
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
    "max_grad_norm": 0.3,
    "weight_decay": 0.001,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 100,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "bf16": True,             # Use bfloat16 precision if available
    "tf32": True,             # Use TF32 precision if available
    "max_seq_length": 1024,   # Maximum sequence length
    "load_in_8bit": True,     # Load model in 8-bit precision
}

# Hallucination mitigation specific settings
HALLUCINATION_CONFIG = {
    "eval_metrics": ["hallucination_rate", "factual_consistency", "citation_accuracy"],
    "data_mixing_ratio": {
        "factual_data": 0.6,
        "citation_data": 0.2,
        "hallucination_examples": 0.2,
    },
} 