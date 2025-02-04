import torch
import torch.nn as nn
import random
import mmap
import pickle
import time
import sentencepiece as spm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm
from typing import Dict, List

# ============================
# Configuration
# ============================
class ModelConfig:
    device = 'mps' if torch.mps.is_available() else 'cpu'
    block_size = 1024
    batch_size = 6
    max_iters = 10000
    eval_iters = 1
    learning_rate = 3e-5
    dropout = 0.1
    embedding_dim = 768
    decoder_layers = 12
    number_of_heads = 12
    dataset_name = "OpenAssistant/oasst2" # need to change this
    spm_model_path = "/Users/darrensandhu/Projects/LLM/openassistant/openasst2_tokenizer.model"
    min_quality_score = 0.5

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
# Data Processing
# ============================
def extract_conversations(message_tree: Dict) -> List[List[Dict]]:
    """
    Extract all possible conversation paths from a message tree.
    Returns a list of conversation sequences.
    """
    conversations = []
    
    def traverse_tree(messages: List[Dict], current_path: List[Dict]):
        for message in messages:
            # Add current message to path
            current_path.append({
                'role': message['role'],
                'content': message['text'],
                'quality': message.get('labels', {}).get('quality', {}).get('value', 1.0)
            })
            
            # If this message has children, traverse them
            if 'children' in message and message['children']:
                traverse_tree(message['children'], current_path.copy())
            else:
                # If it's a leaf node, save the conversation
                conversations.append(current_path.copy())
            
            # Remove current message from path before processing siblings
            current_path.pop()
    
    # Start traversal from the root messages
    traverse_tree([message_tree], [])
    return conversations

def filter_conversation(conversation: List[Dict], min_quality: float = 0.5) -> bool:
    """Filter conversations based on quality scores"""
    return all(msg.get('quality', 1.0) >= min_quality for msg in conversation)

def prepare_openassistant_data(dataset_name: str):
    """Load and prepare OpenAssistant2 dataset"""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Process each message tree into conversation sequences
    processed_data = {'train': [], 'validation': []}
    
    for split in ['train', 'validation']:
        print(f"\nProcessing {split} set...")
        
        # Print sample of raw data structure
        if len(dataset[split]) > 0:
            print("\nSample message structure:")
            print(dataset[split][0].keys())
        
        for tree in tqdm(dataset[split], desc=f"Processing {split} set"):
            try:
                conversations = extract_conversations(tree)
                # Filter conversations based on quality
                quality_conversations = [
                    conv for conv in conversations 
                    if filter_conversation(conv, ModelConfig.min_quality_score)
                ]
                processed_data[split].extend(quality_conversations)
            except Exception as e:
                print(f"Error processing tree: {e}")
                continue
    
    return processed_data

def format_conversation(conversation: List[Dict]) -> str:
    """Format a conversation sequence into a single string"""
    formatted = ""
    for msg in conversation:
        role = "Assistant: " if msg['role'] == "assistant" else "Human: "
        content = msg['content'].strip() if msg['content'] else ""
        formatted += role + content + "\n"
    return formatted.strip()

def get_batch(split: str, config: ModelConfig, tokenizer, dataset: Dict):
    """Generate a batch of data from processed conversations"""
    conversations = []
    
    # Sample random conversations
    for _ in range(config.batch_size):
        if len(dataset[split]) == 0:
            raise ValueError(f"No conversations available in {split} set")
        idx = random.randint(0, len(dataset[split]) - 1)
        conv = dataset[split][idx]
        conversations.append(format_conversation(conv))
    
    # Tokenize conversations
    tokens = []
    for conv in conversations:
        if not conv.strip():  # Skip empty conversations
            continue
        tokens.append(torch.tensor(tokenizer.encode(conv), dtype=torch.long))
    
    if not tokens:  # If no valid tokens were generated
        raise ValueError("No valid conversations were found in the batch")
    
    # Pad or truncate sequences
    x = torch.zeros((len(tokens), config.block_size), dtype=torch.long)
    y = torch.zeros((len(tokens), config.block_size), dtype=torch.long)
    
    for i, seq in enumerate(tokens):
        if len(seq) > config.block_size:
            # Random starting point for long sequences
            start_idx = random.randint(0, len(seq) - config.block_size - 1)
            x[i] = seq[start_idx:start_idx + config.block_size]
            y[i] = seq[start_idx + 1:start_idx + config.block_size + 1]
        else:
            # Pad shorter sequences
            x[i,:len(seq)] = seq
            y[i,:len(seq)-1] = seq[1:]
            if len(seq) > 1:  # Ensure there's at least one token to predict
                y[i,len(seq)-1:len(seq)] = seq[-1]
    
    return x.to(config.device), y.to(config.device)



# def prepare_openassistant_data(dataset_name):
#     """Load and prepare OpenAssistant2 dataset"""
#     dataset = load_dataset(dataset_name)
    
#     # Process each message tree into conversation sequences
#     processed_data = {'train': [], 'validation': []}
    
#     for split in ['train', 'validation']:
#         for tree in tqdm(dataset[split], desc=f"Processing {split} set"):
#             conversations = extract_conversations(tree)
#             # Filter conversations based on quality
#             quality_conversations = [
#                 conv for conv in conversations 
#                 if filter_conversation(conv, ModelConfig.min_quality_score)
#             ]
#             processed_data[split].extend(quality_conversations)
    
#     return processed_data

# def format_conversation(conversation: List[Dict]) -> str:
#     """Format a conversation sequence into a single string"""
#     formatted = ""
#     for msg in conversation:
#         role = "Assistant: " if msg['role'] == "assistant" else "Human: "
#         formatted += role + msg['content'].strip() + "\n"
#     return formatted.strip()

# def get_batch(split: str, config: ModelConfig, tokenizer, dataset: Dict):
#     """Generate a batch of data from processed conversations"""
#     conversations = []
    
#     # Sample random conversations
#     for _ in range(config.batch_size):
#         idx = random.randint(0, len(dataset[split]) - 1)
#         conv = dataset[split][idx]
#         conversations.append(format_conversation(conv))
    
#     # Tokenize conversations
#     tokens = []
#     for conv in conversations:
#         tokens.append(torch.tensor(tokenizer.encode(conv), dtype=torch.long))
    
#     # Pad or truncate sequences
#     x = torch.zeros((config.batch_size, config.block_size), dtype=torch.long)
#     y = torch.zeros((config.batch_size, config.block_size), dtype=torch.long)
    
#     for i, seq in enumerate(tokens):
#         if len(seq) > config.block_size:
#             # Random starting point for long sequences
#             start_idx = random.randint(0, len(seq) - config.block_size - 1)
#             x[i] = seq[start_idx:start_idx + config.block_size]
#             y[i] = seq[start_idx + 1:start_idx + config.block_size + 1]
#         else:
#             # Pad shorter sequences
#             x[i,:len(seq)] = seq
#             y[i,:len(seq)-1] = seq[1:]
#             if len(seq) > 1:  # Ensure there's at least one token to predict
#                 y[i,len(seq)-1:len(seq)] = seq[-1]
    
#     return x.to(config.device), y.to(config.device)


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
        weights = torch.matmul(q, k.transpose(-2,-1)) * (C ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
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
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Feed-Forward Neural Network"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, config):
        super().__init__()
        head_size = config.embedding_dim // config.number_of_heads
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.self_attention = MultiHeadAttention(config, head_size)
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
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.decoder_layers)])
        self.ln_final = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, tokenizer.vocab_size)
        
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
        B, T = idx.shape
        
        # Get embeddings
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=self.config.device))
        x = token_emb + pos_emb
        
        # Transform
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # Loss calculation if targets provided
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            return logits, F.cross_entropy(logits, targets)
        
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
        
    # def sample_next_token(self, logits, top_k=0, top_p=1.0, temperature=1.0):
    #     """
    #     Sample the next token using either top-k or nucleus (top-p) sampling.
    #     Args:
    #         logits (torch.Tensor): Raw logits from model output (batch_size x vocab_size)
    #         top_k (int): Number of highest probability tokens to keep for top-k sampling
    #         top_p (float): Cumulative probability threshold for nucleus sampling
    #         temperature (float): Temperature for softmax, higher means more random
    #     Returns:
    #         torch.Tensor: Index of the sampled token
    #     """
    #     # Apply temperature
    #     logits = logits / temperature

    #     # Top-k sampling
    #     if top_k > 0:
    #         # Get the top k logits and their indices
    #         top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
    #         # Apply softmax to convert to probabilities
    #         probs = F.softmax(top_k_logits, dim=-1)
    #         # Sample from the top-k distribution
    #         idx = torch.multinomial(probs, num_samples=1)
    #         idx_next = top_k_indices.gather(-1, idx)

    #     # Nucleus (top-p) sampling
    #     elif top_p < 1.0:
    #         # Sort logits in descending order
    #         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #         # Calculate cumulative probabilities
    #         probs = F.softmax(sorted_logits, dim=-1)
    #         cumulative_probs = torch.cumsum(probs, dim=-1)
    #         # Remove tokens with cumulative probability above the threshold
    #         sorted_indices_to_remove = cumulative_probs > top_p
    #         # Shift the indices to keep also the first token above the threshold
    #         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #         sorted_indices_to_remove[..., 0] = 0
    #         # Set removed indices to negative infinity
    #         indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    #         logits[indices_to_remove] = float('-inf')
    #         # Sample from the filtered distribution
    #         probs = F.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1)

    #     # Regular sampling
    #     else:
    #         probs = F.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1)

    #     return idx_next

    # def generate2(self, idx, max_new_tokens, top_k=50, top_p=0.95, temperature=1.0):
    #     """
    #     Generate text using the model with sampling options.
    #     Args:
    #         idx (torch.Tensor): Starting token indices
    #         max_new_tokens (int): Number of tokens to generate
    #         top_k (int): If > 0, use top-k sampling
    #         top_p (float): If < 1.0, use nucleus sampling
    #         temperature (float): Temperature for sampling
    #     """
    #     # Make sure idx is 2D if it's not already
    #     if idx.dim() == 1:
    #         idx = idx.unsqueeze(0)
            
    #     for _ in range(max_new_tokens):
    #         # Crop context if needed
    #         idx_cond = idx[:, -self.config.block_size:]
    #         # Get predictions
    #         logits, _ = self(idx_cond)
    #         logits = logits[:, -1, :] # Focus on last token
    #         # Sample next token
    #         idx_next = self.sample_next_token(
    #             logits,
    #             top_k=top_k,
    #             top_p=top_p,
    #             temperature=temperature
    #         )
    #         # Append to the sequence
    #         idx = torch.cat((idx, idx_next), dim=1)
        
    #     return idx

    # def generate(self, index, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9):
    #     for _ in range(max_new_tokens):
    #         index_cond = index[:, -self.config.block_size:]
    #         logits, _ = self.forward(index_cond)
    #         logits = logits[:, -1, :]  # Focus on the last token's logits

    #         # Apply temperature
    #         logits = logits / temperature

    #         # Apply top-k filtering
    #         if top_k > 0:
    #             values, indices = torch.topk(logits, k=top_k)
    #             logits[logits < values[:, -1, None]] = float('-inf')

    #         # Apply top-p (nucleus) sampling
    #         if top_p < 1.0:
    #             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    #             sorted_indices_to_remove = cumulative_probs > top_p
    #             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
    #             sorted_indices_to_remove[..., 0] = 0
    #             logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

    #         # Sample from final probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         index_next = torch.multinomial(probs, num_samples=1)

    #         # Append new token
    #         index = torch.cat((index, index_next), dim=1)
        
    #     return index

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
def train_model(model, config, tokenizer, dataset):
    """Train the model"""
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_iters, eta_min=1e-5)
    try:
        start_time = time.time()
        for iter in range(config.max_iters):
            xb, yb = get_batch('train', config, tokenizer, dataset)
            optimizer.zero_grad()

            logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Evaluation
            if iter % config.eval_iters == 0:
                losses = estimate_loss(model, config, tokenizer, dataset)
                elapsed = time.time() - start_time
                start_time = time.time()
                print(f'Step: {iter}, Train Loss: {losses["train"]:.4f}, Val Loss: {losses["val"]:.4f}, Time: {elapsed:.2f}s')
        
        # Save model
        with open('model-01_oasst2.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
        with open('model-01_oasst2.pkl', 'wb') as f:
            pickle.dump(model, f)

@torch.no_grad()
def estimate_loss(model, config, tokenizer, dataset):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, config, tokenizer, dataset)
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
    
    # Load and process OpenAssistant dataset
    print("Loading and processing dataset...")
    dataset = prepare_openassistant_data(config.dataset_name)
    print(f"Processed {len(dataset['train'])} training conversations")
    print(f"Processed {len(dataset['validation'])} validation conversations")
    
    # Initialize model
    model = GPTLanguageModel(config, tokenizer).to(config.device)
    
    # Train model
    train_model(model, config, tokenizer, dataset)
    
    # Test generation
    test_prompt = "Once upon a time"
    generated_text = model.generate(test_prompt, max_new_tokens=100, temperature=0.8)
    print("\nGenerated Text:")
    print(generated_text)