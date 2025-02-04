# GPT Langauge Model

## Overview

This repository contains an implementation of a GPT-style language model built using PyTorch. The model utilizes a Transformer architecture with multi-head self-attention and causal attention masking, and it is trained on a custom dataset (e.g., OpenWebText). It is designed for text generation tasks, producing human-like text based on a given prompt.

## Key Features

- **Transformer-based Architecture**: Implements multi-head self-attention and feed-forward networks.
- **Custom Tokenizer**: Uses a SentencePiece model for efficient subword tokenization.
- **Dynamic Data Loading**: Retrieves random chunks from a large text file for training.
- **Text Generation**: Supports text generation with adjustable temperature, top-k, and nucleus (top-p) sampling.
- **Training & Evaluation**: Periodic evaluation during training using cross-entropy loss.

## Project Structure

The repository is organized as follows:

```
llm/
    /data
        /data extraction scripts
    /pretrained
        /pretraining scripts
    /finetuning
        /finetuning scripts

```

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```
The requirements.txt file contains all the necessary packages to run the code.

## Configuration

The training and model configuration is handled in the ModelConfig class within llm/finetuning/model.py. 
You can adjust parameters such as:
    . block_size: Length of the input sequence.
    . batch_size: Number of samples per batch.
    . max_iters: Total training iterations.
    . eval_iters: Frequency of evaluation steps.
    . learning_rate: Optimizer learning rate.
    . embedding_dim: Dimensionality of the token embeddings.
    . number_of_heads: Number of attention heads.
    . decoder_layers: Number of transformer layers.
    . dropout: Dropout rate for regularization.

The data configuration is handled in the DataConfig class within llm/data/data_config.py.
You can adjust parameters for your tokenizer such as:
    . vocab_size: Size of the vocabulary.
    . input_sentence_size: Maximum length of input sentences.
    . output_file_train: Path to the training data file.
    . output_file_val: Path to the validation data file.

    