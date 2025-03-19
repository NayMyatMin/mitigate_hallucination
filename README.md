# Hallucination Mitigation Evaluation Pipeline

This project implements a comprehensive pipeline for evaluating hallucination mitigation in large language models using several widely used question-answering datasets.

## Overview

The pipeline evaluates how much a language model hallucinates by comparing its outputs against human-validated ground truth answers across multiple QA datasets:

- **CoQA**: Conversational Question Answering dataset with dialog structure
- **SQuAD v2.0**: Stanford Question Answering Dataset with answerable questions focus
- **TriviaQA**: Trivia questions and answers dataset (rc.nocontext subset)
- **TruthfulQA**: Dataset to measure truthfulness with binary choice format
- **HaluEval QA**: Dataset specifically designed to evaluate hallucination with factual questions

## Project Structure

- `lib/`: Core library modules for evaluation framework
  - `__init__.py`: Package initialization file
  - `evaluation.py`: Core dataset evaluation workflow
  - `prediction_cleaning.py`: Functions for cleaning and normalizing predictions
  - `result_handling.py`: Utilities for saving and processing results
  - `config_handling.py`: Configuration and model setup utilities
- `config/`: Configuration files for models and datasets
  - `data_config.py`: Dataset configurations and prompt templates
  - `model_config.py`: Model configurations and LoRA parameters
- `utils/`: Utility functions
  - `data_utils.py`: Functions for loading and processing datasets
  - `model_utils.py`: Functions for loading and preparing models
  - `eval_utils.py`: Evaluation helper functions
- `dataset/`: Dataset files and validation scripts
  - `HaluEval/`: HaluEval QA dataset files
  - `TruthfulQA.csv`: TruthfulQA dataset in binary choice format
  - `comprehensive_dataset_check.py`: Script to validate all datasets
- `train.py`: Script for fine-tuning models with LoRA
- `evaluate.py`: Main script for evaluating hallucination in trained models
- `inference.py`: Script for running inference with fine-tuned models
- `diagnostic.py`: Script for diagnosing and debugging the pipeline
- `sbatch_train.sh`: SLURM script for training on HPC environments

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NayMyatMin/mitigate_hallucination
   cd hallucination-mitigation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```

## Dataset Preparation

The pipeline automatically downloads and prepares the following QA datasets:

- **CoQA**: Up to 8000 QA pairs (covers the paper's 7983 QA pairs in development split)
- **SQuAD v2.0**: Up to 6000 filtered development split with answerable questions only (covers the paper's 5928)
- **TriviaQA**: rc.nocontext subset, up to 10000 QA pairs (covers the paper's 9960)
- **TruthfulQA**: Binary choice format for truthfulness evaluation
- **HaluEval QA**: Dataset with paired truthful and hallucinated answers for evaluation

To verify all datasets are properly prepared, run:
```bash
python dataset/comprehensive_dataset_check.py
```

## Usage

### Fine-tuning a Model

```bash
python train.py --model_name llama3.1-8b-instruct --dataset_type truthfulqa --load_in_8bit
```

For HPC environments with SLURM:
```bash
sbatch sbatch_train.sh
```

### Evaluating Hallucination

Evaluate a base model:
```bash
python evaluate.py --model_name llama3.1-8b-instruct --dataset_types coqa squad_v2 triviaqa halueval_qa truthfulqa --load_in_8bit
```

Evaluate a fine-tuned model with LoRA adapter:
```bash
python evaluate.py --model_name llama3.1-8b-instruct --adapter_path outputs/llama3.1-8b-instruct_truthfulqa_lora --dataset_types coqa squad_v2 triviaqa halueval_qa truthfulqa --load_in_8bit
```

### Running Inference

```bash
python inference.py --model_name llama3.1-8b-instruct --adapter_path outputs/llama3.1-8b-instruct_truthfulqa_lora --load_in_8bit
```

For interactive mode, simply omit the --prompt parameter:
```bash
python inference.py --model_name llama3.1-8b-instruct --adapter_path outputs/llama3.1-8b-instruct_truthfulqa_lora --load_in_8bit
```

## Evaluation Metrics

The pipeline calculates the following metrics to measure hallucination:

- **ROUGE scores**: ROUGE-1, ROUGE-2, and ROUGE-L for measuring overlap with reference answers
- **BLEU scores**: BLEU-1 and BLEU-4 for additional precision metrics
- **Hallucination score**: A composite metric (1 - ROUGE-L) where higher values indicate more hallucination
- **Empty prediction rate**: How often the model fails to produce an answer

## Code Architecture and Refactoring

The evaluation framework has been refactored into a modular structure to improve maintainability and organization.

### Modular Design

1. **Main Script**: `evaluate.py` orchestrates the evaluation process
2. **Library Modules**:
   - `lib/evaluation.py`: Core dataset evaluation workflow
   - `lib/prediction_cleaning.py`: Functions to clean model outputs
   - `lib/result_handling.py`: Saving and processing results
   - `lib/config_handling.py`: Configuration and setup utilities

### Refactoring Motivation

The original evaluation script contained all functionality in a single large file, which made it difficult to:
- Understand the overall flow of the evaluation process
- Maintain and update specific parts of the pipeline
- Add new features without risking unintended side effects
- Reuse components across different scripts

### Refactoring Strategy

The refactoring followed these principles:
1. **Separation of concerns**: Each module has a clear, singular responsibility
2. **Encapsulation**: Implementation details are hidden within their respective modules
3. **Cohesion**: Related functionality is grouped together
4. **Reusability**: Components can be used independently

### Benefits of the New Structure

1. **Improved maintainability**: Each module has clear responsibilities, making updates easier
2. **Better testability**: Modules can be tested independently
3. **Easier extension**: New datasets or evaluation methods can be added with minimal changes
4. **Better readability**: The main script is now significantly shorter and easier to understand
5. **Reduced duplication**: Common functionality is centralized and reused
6. **Better error handling**: Errors can be isolated to specific modules

### Future Improvements

Potential areas for future improvement:
1. Add unit tests for each module
2. Create a configuration file system rather than hard-coded parameters
3. Implement a plugin system for adding new datasets and evaluation metrics
4. Add more documentation and examples
5. Optimize performance for large dataset evaluation

## Configuration

You can configure datasets and models in the respective configuration files:

- `config/data_config.py`: Add or modify dataset configurations, prompt templates, and data processing parameters
- `config/model_config.py`: Configure model parameters, LoRA settings, and training hyperparameters

## Troubleshooting

If you encounter issues with the pipeline, run the diagnostic script:
```bash
python diagnostic.py
```

This will check all dependencies, dataset loading, and model compatibility.

## License

[Specify the license here]

## Acknowledgments

This project uses datasets from the following sources:
- Stanford NLP (CoQA)
- Stanford University (SQuAD v2.0)
- University of Washington (TriviaQA)
- TruthfulQA creators
- HaluEval creators

## Citation

If you use this code in your research, please cite:
```
https://github.com/NayMyatMin/mitigate_hallucination
``` 