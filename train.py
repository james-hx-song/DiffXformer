import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time
from models.diff_transformer import DifferentialTransformer
from config import StableLMConfig, CONFIG_ARGS, ToyTransConfig

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
# elif hasattr(torch.backends, 'mps') and torch.mps.is_available():
#     device = 'mps'
print(f"Using device: {device}")
# --------- Dataset Loading --------- #
dataset = "shakespeare"
train_data = np.memmap(f"dataset/{dataset}/train.npy", dtype=np.uint32, mode="r")
val_data = np.memmap(f"dataset/{dataset}/val.npy", dtype=np.uint32, mode="r")

print(f"Train Data: {train_data.shape} | Val Data: {val_data.shape}")
# ------ Model Configuration & Hyperparameters ------ #
# param = "830M"
# model_config = StableLMConfig(**CONFIG_ARGS[param])
model_config = ToyTransConfig()

betas = (0.9, 0.95)
min_lr = 1.28e-5
max_lr = 3.2e-4
warmup_steps = 1000
max_iters = 3000

def get_lr(step):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    
    return max_lr - (step - warmup_steps) * (max_lr - min_lr) / (max_iters - warmup_steps)

batch_size = 1
n_ctx = model_config.n_ctx

def get_batch(mode="train"):
    data = train_data if mode == "train" else val_data
    idx = np.random.randint(len(data) - n_ctx, size=batch_size)

    x = torch.from_numpy(np.stack([data[j:j+n_ctx].astype(dtype=np.int64) for j in idx]))
    y = torch.from_numpy(np.stack([data[j+1:j+n_ctx+1].astype(dtype=np.int64) for j in idx]))

    return x, y

# ------ Training Loop ------ #
model = DifferentialTransformer(model_config)
print(f"Model has: {sum(p.numel() for p in model.parameters())} parameters.")
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=betas)

model.to(device)
model.train()

for i in range(max_iters):
    x, y = get_batch("train")
    x, y = x.to(device), y.to(device)

    # print(f"Training on x: {x.shape} | y: {y.shape}")

    t0 = time.time()

    optimizer.zero_grad()

    logits = model(x)
    # print(f"Logits: {logits.shape} | y: {y.shape}")
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    

    loss.backward()

    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(i+1)
    
    optimizer.step()

    t1 = time.time()

    if device == "cuda":
        torch.cuda.synchronize()

    print(f"Iteration {i+1} | Loss: {loss.item()} | Time: {t1-t0}")

