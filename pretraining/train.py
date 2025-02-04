import torch
import torch.nn as nn
import random
import mmap
import pickle
import time
import math
import sentencepiece as spm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm
from .model import ModelConfig, GPTLanguageModel


# ============================
# Data Processing
# ============================
def get_random_chunk(split, config, tokenizer, max_retries=10):
    """Get a random chunk of text from a large file with retry mechanism"""

    filename = f'{config.data_config.output_file_train}' if split == 'train' else f'{config.data_config.output_file_val}'
    retries = 0

    while retries < max_retries:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            # Get the size of the file (in bytes) to pick a random start position
            f.seek(0, 2)  # Seek to the end of the file
            filesize = f.tell()
            
            # Pick a random starting position
            start = random.randint(0, filesize - 10000)  # Read a large chunk
            # print(f"filesize: {filesize}, start: {start}")
            
            # Read the chunk from the file
            f.seek(start)
            chunk = f.read(10000)  # Read up to 10,000 characters to ensure enough text
            
            # print(f"Length of chunk: {len(chunk)}")
            # print("Chunk: ", chunk)
            
            # Tokenize the chunk
            encoded_chunk = tokenizer.encode(chunk)
            # print(f"Length of encoded chunk: {len(encoded_chunk)}")
            # print("Encoded chunk: ", encoded_chunk)
            
            # Ensure the chunk is large enough
            if len(encoded_chunk) >= config.block_size:
                return torch.tensor(encoded_chunk, dtype=torch.long)
            
            # Retry if chunk is too small
            # print("Block too small, retrying...")
            retries += 1

    # If retries exceed max_retries, raise an exception
    raise ValueError("Exceeded maximum retries to find a valid chunk")

def get_batch(split, config, tokenizer):
    """Generate a batch of data"""
    data = get_random_chunk(split, config, tokenizer)
    # print(len(data))
    # print()
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    # print(ix)
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    # print(x)
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    # print(y)
    return x.to(config.device), y.to(config.device)

# ============================
# Training
# ============================
def train_model(model, config, tokenizer):
    """Train the model"""
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=config.max_iters, eta_min=1e-5)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=config.learning_rate, 
    #     total_steps=config.max_iters, 
    #     pct_start=0.1,  # 10% warmup
    #     anneal_strategy="cos"
    # )

    try:
        start_time = time.time()
        for iter in range(config.max_iters):
            xb, yb = get_batch('train', config, tokenizer)
            optimizer.zero_grad()

            logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            # Evaluation
            if iter % config.eval_iters == 0:
                losses = estimate_loss(model, config, tokenizer)
                elapsed = time.time() - start_time
                start_time = time.time()
                print(f'Step: {iter}, Train Loss: {losses["train"]:.4f}, Val Loss: {losses["val"]:.4f}, Time: {elapsed:.2f}s')
        
        # Save model
        with open(config.model_name + '.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
        with open(config.model_name + '.pkl', 'wb') as f:
            pickle.dump(model, f)

@torch.no_grad()
def estimate_loss(model, config, tokenizer):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, config, tokenizer)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    # Initialize configuration
    config = ModelConfig()
    
    # Initialize tokenizer
    tokenizer = config.tokenizer
    
    # Initialize model
    # try:
    #     with open('model-01_spe_3.pkl', 'rb') as f:
    #         model = pickle.load(f)
    #         print("Model loaded successfully")
    # except:
    model = GPTLanguageModel(config, tokenizer).to(config.device)
    print("Model initialized successfully")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Train model
    train_model(model, config, tokenizer)
    
    # Test generation
    test_prompt = "Once upon a time"
    generated_text = model.generate(test_prompt, max_new_tokens=100, temperature=0.8)
    print("\nGenerated Text:")
    print(generated_text)