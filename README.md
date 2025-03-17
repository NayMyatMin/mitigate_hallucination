# Hallucination Mitigation with LoRA Fine-tuning

This project provides a structured framework for fine-tuning large language models (LLMs) to mitigate hallucinations using Low-Rank Adaptation (LoRA). The implementation focuses on fine-tuning the Llama 3.1 8B model, but can be extended to other models as well.

## Project Structure

```
├── config/
│   ├── model_config.py       # Model and training configurations
│   └── data_config.py        # Dataset configurations
├── utils/
│   ├── data_utils.py         # Data processing utilities
│   ├── model_utils.py        # Model loading and configuration utilities
│   └── evaluation_utils.py   # Evaluation metrics for hallucination
├── train.py                  # Main training script
├── evaluate.py               # Evaluation script
├── inference.py              # Inference script for using the model
└── requirements.txt          # Required dependencies
```

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Weights & Biases for experiment tracking:
```bash
wandb login
```

## Dataset Preparation

Before training, you need to prepare your datasets. The project expects three types of datasets:

1. **Factual dataset**: Contains facts and factual information
2. **Citations dataset**: Examples with proper citations/references
3. **Hallucinations dataset**: Examples of model hallucinations (as negative examples)

Update the dataset paths in `config/data_config.py` with your actual dataset locations.

## Training

To fine-tune the model using LoRA, run:

```bash
python train.py --model_name llama3.1-8b --dataset_type mixed --load_in_8bit
```

Arguments:
- `--model_name`: Model to fine-tune (default: llama3.1-8b)
- `--dataset_type`: Type of dataset to use (factual, citations, hallucinations, mixed)
- `--output_dir`: Directory to save model checkpoints (default: outputs)
- `--load_in_8bit`: Whether to load the model in 8-bit precision
- `--load_in_4bit`: Whether to load the model in 4-bit precision
- `--seed`: Random seed (default: 42)
- `--max_steps`: Maximum number of training steps (overrides num_train_epochs)

## Evaluation

To evaluate the fine-tuned model:

```bash
python evaluate.py --model_name llama3.1-8b --adapter_path outputs/[your-adapter-path] --eval_truthfulqa --load_in_8bit
```

Arguments:
- `--model_name`: Base model name (default: llama3.1-8b)
- `--adapter_path`: Path to the LoRA adapter (optional, if not provided will evaluate base model)
- `--output_dir`: Directory to save evaluation results (default: evaluation_results)
- `--eval_truthfulqa`: Whether to evaluate on TruthfulQA benchmark
- `--eval_custom`: Whether to evaluate on custom datasets
- `--custom_dataset_path`: Path to custom evaluation dataset
- `--max_samples`: Maximum number of samples to evaluate (default: 100)

## Inference

To use the fine-tuned model for inference:

```bash
python inference.py --model_name llama3.1-8b --adapter_path outputs/[your-adapter-path] --load_in_8bit
```

This will start an interactive session. Alternatively, use the `--prompt` argument to provide a single prompt:

```bash
python inference.py --model_name llama3.1-8b --adapter_path outputs/[your-adapter-path] --prompt "What is the capital of France?" --load_in_8bit
```

## Customization

- Modify `config/model_config.py` to adjust model settings, LoRA parameters, and training hyperparameters
- Update `config/data_config.py` to configure dataset paths and prompt templates
- Add new models by extending the `MODELS` dictionary in `config/model_config.py`

## Hallucination Metrics

The project measures several hallucination metrics:

- **Hallucination rate**: The proportion of factually incorrect information
- **Factual consistency**: How well the model sticks to known facts
- **Citation accuracy**: Whether references are properly cited
- **TruthfulQA score**: Performance on the TruthfulQA benchmark

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- PEFT 0.7+
- Accelerate 0.21+
- BitsAndBytes 0.41+

## License

[Your license information] 