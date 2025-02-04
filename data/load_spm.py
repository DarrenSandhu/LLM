import sentencepiece as spm

# ============================
# Load SentencePiece Model
# ============================
spm_model_path = "openwebtext/spm_bpe_vocab-32k_inputSentence-5M.model"  # Ensure this is trained beforehand
sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)

# ============================
# Encode Text
# ============================
text = "Hello, this is an example sentence!"
encoded = sp.encode(text, out_type=int)  # Encode into subword tokens
print(f"Encoded: {encoded}")

# ============================
# Decode Text
# ============================
decoded = sp.decode(encoded)  # Decode subword tokens back into text
print(f"Decoded: {decoded}")

print("SentencePiece model vocabulary size:", sp.get_piece_size())