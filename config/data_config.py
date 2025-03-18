"""
Configuration for dataset setup.
"""

# Dataset configurations
DATASETS = {
    "coqa": {
        "train": "coqa",
        "eval": "coqa",
        "split": "validation",  # Use the validation split for CoQA
        "config": None,
    },
    "squad_v2": {
        "train": "squad_v2",
        "eval": "squad_v2",
        "split": "validation",  # Use the validation split for SQuAD v2.0
        "config": None,
    },
    "triviaqa": {
        "train": "trivia_qa",
        "eval": "trivia_qa",
        "split": "validation",  # Use the validation split for TriviaQA
        "config": "rc.nocontext",  # Use the rc.nocontext subset
    },
    "truthfulqa": {
        "train": None,  # No train split for TruthfulQA
        "eval": "truthfulqa",
        "split": None,  # Custom loading logic for TruthfulQA
        "config": None,
    },
    "halueval_qa": {
        "train": None,  # No train split for HaluEval QA
        "eval": "halueval_qa",
        "split": None,  # Custom loading logic for HaluEval QA
        "config": None,
    },
}

# Prompt templates for different datasets
PROMPT_TEMPLATES = {
    "coqa": """[INST] Answer the following question based on the provided context:

Context:
{context}

Question:
{question}

Answer:
[/INST]""",
    "squad_v2": """[INST] Answer the following question based on the provided context:

Context:
{context}

Question:
{question}

Answer:
[/INST]""",
    "triviaqa": """[INST] Answer the following question based on the provided context:

Context:
{context}

Question:
{question}

Answer:
[/INST]""",

    "truthfulqa": """[INST] {question}

Please select the true statement from the following choices:
{choices}
[/INST]""",
    "halueval_qa": """[INST] Answer the following question based on the provided context:

Context:
{context}

Question:
{question}

Answer:
[/INST]""",
}

# Data processing configurations
DATA_PROCESSING = {
    "max_input_length": 512,
    "max_output_length": 512,
    "shuffle_seed": 42,
    "train_test_split": 0.9,
} 