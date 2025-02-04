import torch
import torch.nn as nn
import random
import mmap
import pickle
import time
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm

# Set the device for computation (MPS or CPU)
device = 'mps' if torch.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Configuration parameters
class ModelConfig():
    # Set the device for computation (MPS or CPU)
    device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    block_size = 128
    batch_size = 64
    max_iters = 300
    eval_iters = 100
    learning_rate = 3e-4
    dropout = 0.2
    embedding_dim = 384
    decoder_layers = 8
    number_of_heads = 8

# ============================
# Data Processing
# ============================
def load_vocabulary(vocab_file='openwebtext/vocab.txt'):
    """Load and process vocabulary from file"""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    
    # Create mapping dictionaries
    string_to_int = {ch:i for i,ch in enumerate(chars)}
    int_to_string = {i:ch for i,ch in enumerate(chars)}
    
    return vocab_size, string_to_int, int_to_string

def encode(s, string_to_int):
    """Convert string to token indices"""
    return [string_to_int[c] for c in s]

def decode(l, int_to_string):
    """Convert token indices to string"""
    return ''.join([int_to_string[i] for i in l])

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
        return self.dropout(self.proj(x))

class FeedForward(nn.Module):
    """Feed-Forward Neural Network"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.ReLU(),
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
        self.self_attention = MultiHeadAttention(config, head_size)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

# ============================
# Main Model
# ============================
class GPTLanguageModel(nn.Module):
    """GPT Language Model"""
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.decoder_layers)])
        self.ln_final = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, vocab_size)
        
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

    
    def sample_next_token(self, logits, top_k=0, top_p=1.0, temperature=1.0):
        """
        Sample the next token using either top-k or nucleus (top-p) sampling.
        Args:
            logits (torch.Tensor): Raw logits from model output (batch_size x vocab_size)
            top_k (int): Number of highest probability tokens to keep for top-k sampling
            top_p (float): Cumulative probability threshold for nucleus sampling
            temperature (float): Temperature for softmax, higher means more random
        Returns:
            torch.Tensor: Index of the sampled token
        """
        # Apply temperature
        logits = logits / temperature

        # Top-k sampling
        if top_k > 0:
            # Get the top k logits and their indices
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            # Apply softmax to convert to probabilities
            probs = F.softmax(top_k_logits, dim=-1)
            # Sample from the top-k distribution
            idx = torch.multinomial(probs, num_samples=1)
            idx_next = top_k_indices.gather(-1, idx)

        # Nucleus (top-p) sampling
        elif top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # Calculate cumulative probabilities
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # Set removed indices to negative infinity
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Regular sampling
        else:
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        return idx_next

    def generate2(self, idx, max_new_tokens, top_k=50, top_p=0.95, temperature=1.0):
        """
        Generate text using the model with sampling options.
        Args:
            idx (torch.Tensor): Starting token indices
            max_new_tokens (int): Number of tokens to generate
            top_k (int): If > 0, use top-k sampling
            top_p (float): If < 1.0, use nucleus sampling
            temperature (float): Temperature for sampling
        """
        # Make sure idx is 2D if it's not already
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx[:, -self.config.block_size:]
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # Focus on last token
            # Sample next token
            idx_next = self.sample_next_token(
                logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -self.config.block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index_cond, index_next), dim=1)
        return index


class ChatbotInterface:
    def __init__(self, model_path: str, config: ModelConfig):
        """
        Initialize the chatbot interface.
        
        Args:
            model_path: Path to the saved model file
            config: Model configuration object
        """
        self.config = config
        self.device = config.device
        
        # Load vocabulary
        self.vocab_size, self.string_to_int, self.int_to_string = load_vocabulary()
        
        # Load model
        print("Loading model...")
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
        
    def _load_model(self, model_path: str) -> GPTLanguageModel:
        """Load and initialize the model."""
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model.to(self.device)
        except FileNotFoundError:
            print(f"Model file not found at {model_path}")
            print("Initializing new model...")
            return GPTLanguageModel(self.vocab_size, self.config).to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(self, 
                         prompt: str, 
                         max_tokens: int = 150,
                         temperature: float = 0.7,
                         top_k: int = 50,
                         top_p: float = 0.95) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: Input text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated response text
        """
        with torch.no_grad():  # Disable gradient computation for inference
            # Encode the prompt
            encoded_prompt = encode(prompt, self.string_to_int)
            
            # Convert to tensor and move to device
            context = torch.tensor(encoded_prompt, 
                                dtype=torch.long, 
                                device=self.device)
            
            # Ensure context has batch dimension
            if context.dim() == 1:
                context = context.unsqueeze(0)
            
            # Generate response
            generated_tokens = self.model.generate2(
                context,
                max_new_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            
            # Extract the generated text (remove the prompt)
            generated_text = decode(generated_tokens[0].tolist(), self.int_to_string)
            
            # Return only the new generated text (remove the prompt)
            prompt_length = len(prompt)
            return generated_text[prompt_length:]

    def chat_loop(self):
        """Run the interactive chat loop."""
        print("\nChatbot is ready! (Type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                prompt = input("\nYou: ").strip()
                
                # Check for exit command
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                # Skip empty inputs
                if not prompt:
                    continue
                
                # Generate and print response
                response = self.generate_response(prompt)
                print("\nChatbot:", response)
                
            except KeyboardInterrupt:
                print("\n\nChat session interrupted.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")

def main():
    # Initialize configuration
    config = ModelConfig()
    
    # Create chatbot interface
    chatbot = ChatbotInterface(
        model_path="model-01.pkl",
        config=config
    )
    
    # Start chat loop
    chatbot.chat_loop()

if __name__ == "__main__":
    main()