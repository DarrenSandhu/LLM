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
    ....
    ....
    /data
        /data extraction scripts
    /finetuning
        /finetuning scripts
    /openwebtext
        /openwebtext data and sentencepiece models
    /pretrained
        /pretraining scripts
    ....
    ....

```

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```
The requirements.txt file contains all the necessary packages to run the code.

## Configuration

The training and model configuration is handled in the `ModelConfig` class within `llm/finetuning/model.py`. You can adjust parameters such as:
- **block_size**: Length of the input sequence.
- **batch_size**: Number of samples per batch.
- **max_iters**: Total training iterations.
- **eval_iters**: Frequency of evaluation steps.
- **learning_rate**: Optimizer learning rate.
- **embedding_dim**: Dimensionality of the token embeddings.
- **number_of_heads**: Number of attention heads.
- **decoder_layers**: Number of transformer layers.
- **dropout**: Dropout rate for regularization.

The data configuration is handled in the `DataConfig` class within `llm/data/data_config.py`. You can adjust parameters for your tokenizer such as:
- **vocab_size**: Size of the vocabulary.
- **input_sentence_size**: Maximum length of input sentences.
- **output_file_train**: Path to the training data file.
- **output_file_val**: Path to the validation data file.

## Creating the Dataset and Training the SentencePiece Model

To create the dataset, run the following command:

```bash
python -m data.create_spm
```

When you run this script, it performs several key tasks to prepare your text dataset and train a SentencePiece BPE tokenizer model. Here’s a breakdown of what the script does:

- **Load Data Configuration**  
  The script begins by importing required modules and initializing the data configuration via the `DataConfig` class. This configuration provides access to the training and validation datasets as well as various file paths and parameters.

- **Display Dataset Statistics**  
  It prints out the total number of rows in both the training and validation datasets, giving you an overview of your data size.

- **Clean the Full Dataset**  
  The script cleans the entire dataset by processing the raw text data and writing the cleaned output to a file. This step ensures that the data is formatted consistently and is ready for tokenizer training.

- **Train the SentencePiece BPE Model**  
  Using the cleaned dataset file as input, the script trains a SentencePiece model with the following configurations:
  - **Vocabulary Size:** Sets the size of the tokenizer vocabulary.
  - **Model Type:** Uses Byte Pair Encoding (BPE) for subword segmentation.
  - **Max Sentence Length:** Specifies the maximum sentence length to consider.
  - **Large Corpus Handling:** Optimized for extremely large datasets.
  - **Character Coverage & Normalization:** Ensures broad character coverage and applies specific normalization rules.
  - **Input Shuffling & Multi-threading:** Improves training efficiency by shuffling input sentences and utilizing multiple threads.
  
  The model is saved with a unique prefix specified in the configuration.

- **Cleanup of Intermediate Files**  
  After training the SentencePiece model, the script deletes the cleaned full dataset file to free up disk space.

- **Create Cleaned Train and Validation Files**  
  If they do not already exist, the script creates separate cleaned files for training and validation datasets. This step prepares the data for subsequent model training tasks.

- **Test the Trained SentencePiece Model**  
  Finally, the script loads the newly trained SentencePiece model and demonstrates its functionality by:
  - **Encoding a Sample Text:** Converts an example sentence into subword tokens.
  - **Decoding the Tokens:** Converts the subword tokens back into the original text, verifying that the model works as expected.

By the end of the script, you will have a trained SentencePiece BPE model ready for tokenizing text, along with cleaned train and validation files prepared for further model training.


## Training The Model

To train the model, run the following command:

```bash
python -m pretraining.train
```
This script will:
- Load random text chunks from **config.data.output_file_train**.
- Tokenize the text using SentencePiece from **config.tokenizer**.
- Train the model using the AdamW optimizer and cross-entropy loss
- Evaluate the model on the validation set every **config.model.eval_iters** iterations.

## Fine-Tuning The Model
I will add more details on fine-tuning the model in the future.

## Text Generation
After training, you can generate text using the model. 
The script includes a *generate* method that supports parameters such as:
- **Prompt**: The input text to generate from.
- **Max Tokens**: The maximum number of tokens to generate.
- **Temperature**: Controls the randomness of the generated text.
- **Top-K**: Limits the sampling to the top-k most likely tokens.
- **Top-P**: Limits the sampling to the cumulative probability of the top-p most likely tokens.

**Note**: The model must be trained before generating text.
<!-- **TODO**:
To generate text, run the following command :

```bash
python -m pretraining.generate --prompt "Once upon a time" --max_tokens 100 --temperature 0.7
```  -->

## Evaluating The Model

During training, the script periodically evaluates the model using the estimate_loss function, which computes cross-entropy loss on both training and validation sets. This helps monitor performance and detect overfitting.

## Saving and Loading Models

The model is periodically saved as a pickle file. To load a saved model:

```python
import pickle

with open('model_checkpoint.pkl', 'rb') as f:
    model = pickle.load(f)
```

## Future Work
 
- Pretrain the model down to a smaller loss value.
- Fine-tune the model on the open assistant dataset.
- Create a fine-tuning script.
- Create a chatbot using the fine-tuned model.
- Evaluate the model on the open assistant dataset.

## License
Licensed under the MIT License. See `LICENSE` for more information.