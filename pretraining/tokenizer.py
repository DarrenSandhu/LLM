import sentencepiece as spm
import torch
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