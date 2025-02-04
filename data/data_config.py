import py7zr
import lzma
import os
import pyarrow.feather as feather
import mmap
import random
from tqdm import tqdm
from datasets import load_dataset

class DataConfig:
    folder_path = "/Users/darrensandhu/Projects/LLM/openwebtext/train"
    output_file_train = "cleaned_openwebtext_train.txt"
    output_file_val = "cleaned_openwebtext_val.txt"
    output_file_dataset = "cleaned_openwebtext_dataset.txt"
    print("Loading dataset...")
    dataset = load_dataset('Skylion007/openwebtext')
    data = dataset["train"]
    total_rows = len(data)
    split_index = int(total_rows * 0.9)
    train_dataset = data.select(range(0, split_index))
    val_dataset = data.select(range(split_index, total_rows))
    vocab_size = 32000
    input_sentence_size = 10000000
    model_prefix = f'spm_bpe_vocab-{vocab_size}_inputSentence-{input_sentence_size}'

    def clean_file(self, output_file, dataset):
        with open(output_file, 'w') as f:
            for row in tqdm(dataset, total=len(dataset)):
                f.write(row['text'] + '\n')