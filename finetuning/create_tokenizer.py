import sentencepiece as spm
import re
from datasets import load_dataset

ds = load_dataset("mylesmharrison/cornell-movie-dialog")

# Split the dataset into training and validation sets
data = ds['train']
print(data.num_rows)
size = data.num_rows
n_split = int(size * 0.9)
train = data.select(range(0, n_split))
val = data.select(range(n_split, size))
print(f"Training samples: {len(train)}, Validation samples: {len(val)}")
print("Dataset loaded successfully!")
print(train[1])

# Function to remove character's name from the text
def remove_character_name(text):
    # Regular expression to match the character's name followed by whitespace
    cleaned_text = re.sub(r'^[A-Za-z]+[\s]+', '', text)  # Match name followed by spaces at the start
    return cleaned_text

# Remove character names from the training and validation sets
train_data = train.map(lambda x: {'text': remove_character_name(x['text'])})
val_data = val.map(lambda x: {'text': remove_character_name(x['text'])})

print(train_data[:10])  # Preview first 10 examples

# Save the text data to a file
with open("cornell_text.txt", "w") as f:
    for row in train_data:
        text = row["text"]
        f.write(text + "\n")  # Write each example to a new line
    
    for row in val_data:
        text = row["text"]
        f.write(text + "\n")

# Train the SentencePiece model
spm.SentencePieceTrainer.train(
    input="cornell_text.txt",  # Input file with the text data
    model_prefix="cornell_tokenizer",  # Prefix for the model files
    vocab_size=32000,  # Adjust this based on your needs
    character_coverage=0.9995,  # High coverage for rare characters
    model_type="bpe",  # Byte Pair Encoding tokenizer
    input_sentence_size=1000000  # Use a large sample for training
)
