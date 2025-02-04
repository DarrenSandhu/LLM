import sentencepiece as spm
import os
import py7zr
import lzma
import os
import pyarrow.feather as feather
import mmap
import random
from tqdm import tqdm
from tqdm import tqdm
from datasets import load_dataset
from data_config import DataConfig

DIRECTORY = os.path.dirname(os.path.realpath(__file__))

data_config = DataConfig()
print(f"Total rows train: {len(data_config.train_dataset)}")
print(f"Total rows val: {len(data_config.val_dataset)}")

# Write to file
print("Cleaning full dataset to file...")
data_config.clean_file(data_config.output_file_dataset, data_config.data)


# Train SentencePiece on the chunk
print("Training SentencePiece BPE model on the full dataset...")
spm.SentencePieceTrainer.Train(
    input=data_config.output_file_dataset,                 # Provide file instead of a string
    model_prefix=f'{data_config.model_prefix}',  # Unique prefix
    vocab_size=data_config.vocab_size,                      
    model_type='bpe',                       
    max_sentence_length=6000,              
    train_extremely_large_corpus=True,      
    num_threads=16,                          
    character_coverage=0.9995,              
    shuffle_input_sentence=True,             
    max_sentencepiece_length=16,
    input_sentence_size=data_config.input_sentence_size,
    normalization_rule_name='nmt_nfkc'       
)

print("Training complete!")

print("Deleting the cleaned dataset file...")
os.remove(data_config.output_file_dataset)
print("Deleted the cleaned dataset file!")

print("Creating cleaned train and val files...")
if not (os.path.exists(data_config.output_file_train)):
    data_config.clean_file(data_config.output_file_train, data_config.train_dataset)
    print("Created cleaned train files!")
else:
    print("Train files already exist!")
if not (os.path.exists(data_config.output_file_val)):
    data_config.clean_file(data_config.output_file_val, data_config.val_dataset)
    print("Created cleaned val files!")
else:
    print("Val files already exist!")


# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file=f'{data_config.model_prefix}.model')

# Encode some text into subwords
text = "Hello, this is an example sentence!"
encoded = sp.encode(text, out_type=str)  # Encode into subword tokens
print(f"Encoded: {encoded}")

# Decode the subword tokens back into text
decoded = sp.decode(encoded)  # Decode back to the original text
print(f"Decoded: {decoded}")