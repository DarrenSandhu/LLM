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



# ============================
# Tokenizer Wrapper
# ============================
class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
    def encode(self, text):
        """Convert text to token indices"""
        return self.sp.encode(text, out_type=int)
    
    def decode(self, tokens):
        """Convert token indices to text"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.sp.decode(tokens)
    
    @property
    def vocab_size(self):
        """Get the vocabulary size"""
        return self.sp.get_piece_size()
    
# ============================
# Configuration
# ============================
class ModelConfig:
    device = 'mps' if torch.mps.is_available() else 'cpu'
    block_size = 64
    batch_size = 32
    max_iters = 10000
    eval_iters = 100
    learning_rate = 6e-4
    dropout = 0.1
    embedding_dim = 768
    decoder_layers = 12
    number_of_heads = 12
    spm_model_path = "openwebtext/spm_bpe_vocab-32k_inputSentence-10M.model"
    tokenizer = Tokenizer(spm_model_path)
    vocab_size = tokenizer.vocab_size
    model_name = f"preTrained_gpt2_batch={batch_size}_block={block_size}_embed={embedding_dim}_layer={decoder_layers}_head={number_of_heads}"
# ============================
# Data Processing
# ============================
def get_random_chunk(split, config, tokenizer, max_retries=10):
    """Get a random chunk of text from a large file with retry mechanism"""

    filename = f'openwebtext/cleaned_openwebtext_train.txt' if split == 'train' else f'openwebtext/cleaned_openwebtext_val.txt'
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
# Model Components
# ============================
class Head(nn.Module):
    """Single Self-Attention Head"""
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.query = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.value = nn.Linear(config.embedding_dim, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Attention computation
        weights = torch.matmul(q, k.transpose(-2,-1)) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, torch.finfo(weights.dtype).min)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        # Weighted aggregation
        v = self.value(x)
        return torch.matmul(weights, v)

class MultiHeadAttention(nn.Module):
    """Multiple Self-Attention Heads in Parallel"""
    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.number_of_heads)])
        self.proj = nn.Linear(head_size * config.number_of_heads, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Debug check for NaN
        # if torch.isnan(out).any():
        #     print("NaN detected in MultiHeadAttention before projection")

        # out = self.proj(out)  # Project concatenated outputs
        
        # # Debug check for NaN after projection
        # if torch.isnan(out).any():
        #     print("NaN detected in MultiHeadAttention after projection")
        out = self.dropout(self.proj(out))
        return out
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.number_of_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=False) 
        # output projection
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.number_of_heads
        self.n_embd = config.embedding_dim
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    """Feed-Forward Neural Network"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, config):
        super().__init__()
        head_size = config.embedding_dim // config.number_of_heads
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.self_attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)
        
        

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

# ============================
# Main Model
# ============================
class GPTLanguageModel(nn.Module):
    """GPT Language Model"""
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Embedding layers
        self.token_embedding = nn.Embedding(tokenizer.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        
        # Transformer blocks
        # self.blocks = nn.Sequential(*[Block(config) for _ in range(config.decoder_layers)])
        # self.ln_final = nn.LayerNorm(config.embedding_dim)
        # self.lm_head = nn.Linear(config.embedding_dim, tokenizer.vocab_size)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim),
            wpe = nn.Embedding(config.block_size, config.embedding_dim),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.decoder_layers)]),
            ln_f = nn.LayerNorm(config.embedding_dim),
        ))
        self.lm_head = nn.Linear(config.embedding_dim, tokenizer.vocab_size)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # B, T = idx.shape
        
        # # Get embeddings
        # token_emb = self.token_embedding(idx)
        # pos_emb = self.position_embedding(torch.arange(T, device=self.config.device))
        # x = token_emb + pos_emb
        
        # # Transform
        # x = self.blocks(x)
        # x = self.ln_final(x)
        # logits = self.lm_head(x)
        
        # # Loss calculation if targets provided
        # if targets is None:
        #     return logits, None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B*T, C)
        #     targets = targets.view(B*T)
        #     return logits, F.cross_entropy(logits, targets)
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
        
    def generate(self, prompt, max_new_tokens, temperature=0.7, top_k=50, top_p=0.9):
        """Generate text from a prompt"""
        # Encode the prompt
        if isinstance(prompt, str):
            index = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.config.device).unsqueeze(0)
        else:
            index = prompt
            
        for _ in range(max_new_tokens):
            # Crop context if needed
            index_cond = index[:, -self.config.block_size:]
            # Get predictions
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :]
            
            # Apply temperature and sampling
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the sequence
            index = torch.cat((index, index_next), dim=1)
        
        # Decode the generated sequence
        return self.tokenizer.decode(index[0].tolist())
    
    # def generate(self, index, max_new_tokens):
    #     for _ in range(max_new_tokens):
    #         index_cond = index[:, -self.config.block_size:]
    #         logits, _ = self.forward(index_cond)
    #         logits = logits[:, -1, :]
    #         probs = F.softmax(logits, dim=-1)
    #         index_next = torch.multinomial(probs, num_samples=1)
    #         index = torch.cat((index_cond, index_next), dim=1)
    #     return index

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
    tokenizer = Tokenizer(config.spm_model_path)
    
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