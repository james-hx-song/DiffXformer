import tiktoken
import requests
import numpy as np
import os

print("Downloading Shakespeare dataset...")

file_dir = os.path.join(os.path.dirname(__file__), "shakespeare.txt")

if not os.path.exists(file_dir):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open(file_dir, "w") as f:
        f.write(response.text)

with open(file_dir, "r") as f:
    text = f.read()

n = len(text)
train_text = text[:int(0.9*n)]
val_text = text[int(0.9*n):]

enc = tiktoken.get_encoding("cl100k_base") # Vocab Size: 100288
train_tokens = enc.encode(train_text)
val_tokens = enc.encode(val_text)

train_tokens = np.array(train_tokens, dtype=np.uint32)
val_tokens = np.array(val_tokens, dtype=np.uint32)

train_tokens.tofile(os.path.join(os.path.dirname(__file__), 'train.npy'))
val_tokens.tofile(os.path.join(os.path.dirname(__file__), 'val.npy'))
