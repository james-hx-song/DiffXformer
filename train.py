import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import time
from models.diff_transformer import DifferentialTransformer
from config import StableLMConfig, VANILLA_CONFIG_ARGS, DIFF_CONFIG_ARGS, ToyTransConfig
from datasets import load_dataset
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --------- Dataset Loading --------- #


def prepareDataset(batch_size: int):
    # Load the dataset
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        split="train",
        streaming=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/smollm-corpus")

    hugging_face_wrapper = HuggingFaceDatasetWrapper(ds, tokenizer)

    dataloader = DataLoader(hugging_face_wrapper, batch_size=batch_size, shuffle=True)
    return dataloader


class HuggingFaceDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        tokenized = self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # now get the text (x) and the label (y)
        # the label is just x shifted by one token
        x = tokenized["input_ids"].squeeze()
        y = x.clone()
        y = torch.cat((x[1:], torch.tensor([-100], dtype=x.dtype)))
        return x, y


class Trainer:
    def __init__(
        self,
        model,
        config,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        max_iters: int,
    ):
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])
        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.current_iter = 0

        def _run_batch(self, x, y):
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            self.optimizer.step()

        def train(self):
            self.model.train()
            while self.current_iter < self.max_iters:
                for x, y in self.dataloader:
                    x, y = x.to(self.gpu_id), y.to(self.gpu_id)
                    self._run_batch(x, y)
                    self.current_iter += 1
                    if self.current_iter % self.save_every == 0:
                        self.save_model()


def main():
    # Hyperpraameters
    max_iters = 200
    batch_size = 64
    # This is the context length of the model. The smoll LLM models use 2048. Longer will make training slower.
    max_length = 1024
