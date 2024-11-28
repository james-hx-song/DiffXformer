import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np

import time
import os
from config import StableLMConfig, ToyTransConfig
from datasets import load_dataset
from transformers import AutoTokenizer

from models.model import TransModel


from typing import Optional

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --------- Dataset Loading --------- #
def load_dataloader(
    batch_size: int,
    tokenizer_name: str,
    n_ctx: int,
    num_val_samples: Optional[int] = None
):
    
    if num_val_samples is None:
        num_val_samples = batch_size
    
    print("Loading Dataset...")
    # Load the dataset
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        split="train",
        streaming=True,
    )

    ds_val = ds.take(num_val_samples)
    ds_train = ds.skip(num_val_samples)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(example):
        """
        Tokenize the text into a tensor. If text is longer than context length, then
        split into smaller batches of context length. If smaller than context length,
        add padding to match context length.
        
        Args:
            example (dict): Dictionary containing 'text' key with the input text (huggingface elem)
        
        Returns:
            dict: Dictionary with 'input_ids' and 'attention_mask' tensors
         """
        tokenized = tokenizer(
            example['text'], 
            add_special_tokens=True, 
            return_tensors='pt', 
            padding=False,  # We'll handle padding manually
            truncation=False  # We'll handle truncation if needed
        )

        max_len = n_ctx + 1

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        curr_length = input_ids.shape[1]
        next_multiple = ((curr_length + max_len - 1) // max_len) * max_len

        padding_length = next_multiple - curr_length
        padded_input_ids = F.pad(
            input_ids,
            (0, padding_length),
            mode='constant',
            value=tokenizer.pad_token_id
        )

        padded_attention_mask = F.pad(
            attention_mask,
            (0, padding_length),
            mode='constant',
            value=0
        )

        reshaped_input_ids = padded_input_ids.view(-1, max_len)
        reshaped_attention_mask = padded_attention_mask.view(-1, max_len)

        input = reshaped_input_ids[:, :-1]
        target = reshaped_input_ids[:, 1:]

        mask = reshaped_attention_mask[:, 1:]

        assert input.shape[1] == n_ctx, f"Input shape: {input.shape}"
        assert target.shape[1] == n_ctx, f"Target shape: {target.shape}"

        return {
            'input_ids': input,
            'mask': mask, 
            'target_ids': target
        }
    
    ds_train = ds_train.map(
        tokenize, 
        batched=True, 
        batch_size=1, 
        remove_columns=['text', 'id', 'metadata']
    )

    ds_val = ds_val.map(
        tokenize, 
        batched=True, 
        batch_size=1, 
        remove_columns=['text', 'id', 'metadata']
    )

    ds_train_loader= DataLoader(ds_train, batch_size=batch_size,)
    ds_val_loader = DataLoader(ds_val, batch_size=batch_size,)

    return ds_train_loader, ds_val_loader


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: Optional[int],
        save_every: int,
        max_iters: int,
        eval_every: int
    ):
        self.model = model.to(gpu_id)

        if gpu_id is not None:
            self.model = DDP(model, device_ids=[gpu_id])

        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader    
        self.optimizer = optimizer
        self.gpu_id = gpu_id if gpu_id is not None else 'cpu'
        self.save_every = save_every
        self.current_iter = 0
        self.max_iters = max_iters
        self.eval_every = eval_every
        
    def eval(self):
        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            count = 0
            for batch in self.val_loader:
                x = batch['input_ids'].to(self.gpu_id)
                y = batch['target_ids'].to(self.gpu_id)

                mask = batch['mask'].to(self.gpu_id)

                output = self.model(x)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1), reduction='none')
                loss = (loss * mask.view(-1)).sum() / mask.sum()

                total_loss += loss.item()
                count += 1
            
            print(f"Validation Loss: {total_loss / count}")
    def _save_checkpoint(self):
        checkpoint = {
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_iteration": self.current_iter,
            "sampler_state": self.dataloader.sampler.state_dict(),  # Save the state of the sampler
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(
            f"Iteration {self.current_iter} | Training checkpoint saved at {self.checkpoint_path}"
        )

    def _load_checkpioint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iteration = checkpoint["current_iteration"]
        self.dataloader.sampler.load_state_dict(checkpoint["sampler_state"])
        print(
            f"Checkpoint loaded from {self.checkpoint_path} at iteration {self.current_iter}"
        )


    def train(self):
        self.model.train()
        while self.current_iter < self.max_iters:
            for batch in self.train_loader:

                if self.current_iter % self.eval_every == 0:
                    # Evaluate Validation Loss
                    self.eval()
                    self.model.train()
                
                x = batch['input_ids'].to(self.gpu_id)
                y = batch['target_ids'].to(self.gpu_id)

                mask = batch['mask'].to(self.gpu_id)
                

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1), reduction='none')
                # print(loss.shape, mask.shape)
                loss = (loss * mask.view(-1)).sum() / mask.sum()

                # loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
                loss.backward()
                self.optimizer.step()

                self.current_iter += 1
                if self.current_iter % self.save_every == 0:
                    self.save_checkpoint()

                if self.current_iter % 10 == 0:
                    print(f"Iter: {self.current_iter}, Loss: {loss.item()}")


def test():
    # Hyperpraameters
    max_iters = 200
    batch_size = 1
    # This is the context length of the model. The smoll LLM models use 2048. Longer will make training slower.
    n_ctx = 64

    config = ToyTransConfig(n_ctx=n_ctx)
    model = TransModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader, val_loader = load_dataloader(
        batch_size,
        "HuggingFaceTB/SmolLM-135M",
        n_ctx
    )


    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        gpu_id=None,
        save_every=1e7,
        max_iters=max_iters,
        eval_every=100
    )

    trainer.train()

if __name__ == "__main__":
    test()


