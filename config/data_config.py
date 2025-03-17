"""
Configuration settings for datasets used in hallucination mitigation finetuning.
"""

# Dataset configurations
DATASETS = {
    # Dataset for factual knowledge
    "factual": {
        "train": "truthful_qa",  # TruthfulQA dataset for factual knowledge
        "eval": "truthful_qa",
        "format": "jsonl",
        "input_key": "question",
        "output_key": "best_answer",
        "subset": "mc", # Multiple choice subset
    },
    
    # Dataset with examples of hallucinations (for negative examples)
    # Using publicly available datasets as alternatives to vectara/hallucination-leaderboard
    "hallucinations": {
        "train": "openai/summarize_from_feedback",  # Contains examples of human feedback on responses
        "eval": "openai/summarize_from_feedback",
        "format": "jsonl",
        "input_key": "query",
        "output_key": "response",
    },

    # Dataset with citation/reference examples
    "citations": {
        "train": "allenai/sciq",  # Science QA with references
        "eval": "allenai/sciq",
        "format": "jsonl", 
        "input_key": "question",
        "output_key": "support",  # Contains the scientific reference/citation
    }
}

# Prompting templates for different dataset types
PROMPT_TEMPLATES = {
    "factual": """[INST] Answer the following question with factual information only:
{question} [/INST]""",
    
    "hallucination": """[INST] Here is an example of a response with hallucinations that should be avoided:
{bad_example}

Now, answer the following question with factual information only:
{question} [/INST]""",
    
    "citation": """[INST] Answer the following question and provide citations for your claims:
{question} [/INST]"""
}

# Data processing configurations
DATA_PROCESSING = {
    "max_input_length": 512,
    "max_output_length": 512,
    "num_proc": 4,
    "shuffle_seed": 42,
    "train_test_split": 0.9,
}

# Evaluation datasets
EVAL_DATASETS = {
    "truthfulqa": "truthful_qa",
    "hallucination_benchmark": "openai/summarize_from_feedback",
} 