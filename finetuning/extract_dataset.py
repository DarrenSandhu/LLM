import pyarrow.feather as feather
import torch
import sentencepiece as spm
from tqdm import tqdm
import re
from datasets import load_from_disk, load_dataset
# from load_openasst2_dataset import openassistant, train, val
from collections import defaultdict
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
train = train.map(lambda x: {'text': remove_character_name(x['text'])})
val = val.map(lambda x: {'text': remove_character_name(x['text'])})
print(train[1])
print(val[1])
print("Character names removed successfully!")
# # Load SentencePiece Model
# model_path = "/Users/darrensandhu/Projects/LLM/openassistant/openasst2_tokenizer.model"

class OpenAsst2Tokenizer:
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


# def extract_conversation_chunks(node, conversation_history=[]):
#     """
#     Recursively traverse the dataset and generate input-target pairs.
    
#     Args:
#         node (dict): Current node in the conversation tree.
#         conversation_history (list): List of previous messages.

#     Returns:
#         List of (input, target) tuples.
#     """
#     chunks = []
    
#     # Append current message to the conversation history
#     new_history = conversation_history + [(node['role'], node['text'])]
#     # print(new_history)

#     # If the current node is an assistant reply, create an input-target pair
#     if node['role'] == "assistant":
#         input_text = format_conversation(new_history[:-1])  # All history before this response
#         print(input_text)
#         target_text = node["text"]  # The assistant's response
#         chunks.append((input_text, target_text))

#     # Recursively process replies
#     for reply in node.get("replies", []):
#         chunks.extend(extract_conversation_chunks(reply, new_history))

#     return chunks

# def format_conversation(history):
#     """
#     Format conversation history into training input format.
#     """
#     formatted = []
#     for role, text in history:
#         formatted.append(f"<|{role}|> {text} <|end|>")
#     return "\n".join(formatted) + "\n<|assistant|>"



# conversation_trees = defaultdict(list)
# for sample in train:
#     # English only
#     # if sample["lang"] != "en":
#     #     continue
#     conversation_trees[sample["message_tree_id"]].append(sample)

# # Process each tree
# conversation_chunks = []
# for tree_id, messages in conversation_trees.items():
#     # Sort messages by timestamp (if available)
#     # messages.sort(key=lambda x: x.get("created_date", ""))
#     print(f"Tree ID: {tree_id}")
#     for message in messages:
#         print(f"{message['role']}:\n {message['text']}")
#         print("\n")
#     print("\n")
#     print("\n")
#     print("\n")

#     # Extract conversation chunks
#     # root_message = next((msg for msg in messages if msg["parent_id"] is None), messages[0])
#     # print(messages[0])
#     conversation_chunks.extend(extract_conversation_chunks(messages[0]))  # Start from the root


# print(f"Extracted {len(conversation_chunks)} conversation chunks.")

# # Print examples
# for i, (input_text, target_text) in enumerate(conversation_chunks[:3]):
#     print(f"Chunk {i+1}:")
#     print("---- Input ----")
#     print(input_text)
#     print("---- Target ----")
#     print(target_text)
#     print("\n")

# def preprocess_messages(dataset):
#     # Group messages by message_tree_id
#     message_tree = defaultdict(list)
    
#     # Iterate through the dataset and group messages
#     for message in dataset:
#         message_tree[message['message_tree_id']].append(message)
    
#     # Now you have a dictionary where each key is a message tree ID,
#     # and the value is a list of all messages in that tree
#     return message_tree

# # Apply the preprocessing function to both train and validation splits
# train_message_trees = preprocess_messages(train)
# val_message_trees = preprocess_messages(val)

# print("Message trees preprocessed successfully!")
# print(f"Training trees: {len(train_message_trees)}, Validation trees: {len(val_message_trees)}")

# def tokenize_conversation(conversation, tokenizer, block_size=512):
#     tokens = []
#     for message in conversation:
#         tokens += tokenizer.encode(message['text'])
#         if len(tokens) >= block_size:
#             break  # Ensure that the chunk doesn't exceed the block size
#     return tokens[:block_size]

# def get_conversation_chunk(message_tree, tokenizer, config):
#     conversation = message_tree
#     tokens = tokenize_conversation(conversation, tokenizer, config)
#     return torch.tensor(tokens, dtype=torch.long)

# # Example: process a random message tree for training
# tokenizer = OpenAsst2Tokenizer(model_path)
# random_message_tree = random.choice(list(train_message_trees.values()))
# print(f"All messages in the tree: {random_message_tree}")
# chunk = get_conversation_chunk(random_message_tree, tokenizer, 8)
# print(f"Chunk encoded: {chunk}")
# print(f"Chunk decoded: {tokenizer.decode(chunk)}")
























# def get_batch(tree, config, tokenizer):
#     """Generate a batch of data"""
#     data = get_conversation_chunk(tree, config, tokenizer)
#     # print(len(data))
#     # print()
#     ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
#     # print(ix)
#     x = torch.stack([data[i:i+config.block_size] for i in ix])
#     # print(x)
#     y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
#     # print(y)
#     return x.to(config.device), y.to(config.device)

# train_folder_path = "/Users/darrensandhu/Projects/LLM/openassistant/oasst2/train"
# val_folder_path = "/Users/darrensandhu/Projects/LLM/openassistant/oasst2/validation"
# output_file_train = "openasst2_output_train.txt"
# output_file_val = "openasst2_output_val.txt"
# vocab_file = "vocab2.txt"

# # Load dataset properly
# print("Loading dataset...")
# train_dataset = load_from_disk(train_folder_path)
# val_dataset = load_from_disk(val_folder_path)

# total_rows = len(train_dataset) + len(val_dataset)

# # split_index = int(total_rows * 0.9)
# # train_dataset = dataset.select(range(0, split_index))
# # val_dataset = dataset.select(range(split_index, total_rows))
# print(f"Total rows: {total_rows}")

# # Filter only prompter
# train_dataset = train_dataset.filter(lambda x: x['role'] == 'prompter')
# print(f"Total rows: {len(train_dataset)}")


# vocab = set()

# # Process training data
# print("Processing training data...")
# with open(output_file_train, "w", encoding="utf-8") as outfile:
#     for row in tqdm(train_dataset, total=split_index):
#         text = row["text"]
#         outfile.write(text + "\n")
#         vocab.update(set(text))

# # Process validation data
# print("Processing validation data...")
# with open(output_file_val, "w", encoding="utf-8") as outfile:
#     for row in tqdm(val_dataset, total=total_rows - split_index):
#         text = row["text"]
#         outfile.write(text + "\n")
#         vocab.update(set(text))
    
# # Save vocab
# with open(vocab_file, "w", encoding="utf-8") as vocab_out:
#     for char in sorted(vocab):
#         vocab_out.write(char + "\n")

    

# print("Processing complete! Vocab saved to", vocab_file)