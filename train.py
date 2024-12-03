import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np

import time
import os
import wandb
import yaml
import sys
from config import StableLMConfig, ToyTransConfig
from datasets import load_dataset
from transformers import AutoTokenizer

from models.model import TransModel
from typing import Optional

# training_config = dict(
#     learning_rate=3e-3, 
#     architecture="DiffFormer",
#     dataset="HuggingFaceTB/smollm-corpus",
#     max_iters=400, 
#     batch_size=1,
#     save_every=100,
#     eval_every=100,
# )

with open("config.yaml", "r") as file:
    training_config = yaml.safe_load(file)

wandb.init(
    project="DiffFormer",
    config=training_config,
)



# --------- Dataset Loading --------- #
def load_dataloader(
    batch_size: int,
    tokenizer_name: str,
    n_ctx: int,
    num_val_samples: Optional[int] = None,
    train_skip_samples: Optional[int] = None
):
    
    if num_val_samples is None:
        num_val_samples = batch_size

    if train_skip_samples is None:
        train_skip_samples = 0
    
    print("Loading Dataset...")
    # Load the dataset
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        split="train",
        streaming=True,
    )

    print(f"Splitting dataset into {num_val_samples} validation samples, skipping {train_skip_samples} training samples")
    ds_val = ds.take(num_val_samples)
    ds_train = ds.skip(num_val_samples + train_skip_samples)

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
        model: TransModel,
        model_config,
        training_config,
        optimizer: torch.optim.Optimizer,
        gpu_id: Optional[int],
    ):
        self.model = model.to(gpu_id)

        if gpu_id is not None:
            self.model = DDP(model, device_ids=[gpu_id])

        self.model_config = model_config
        self.training_config = training_config
        self.train_loader = self.val_loader = None
        self.optimizer = optimizer
        self.gpu_id = gpu_id if gpu_id is not None else 'cpu'
        self.save_every = training_config['save_every']
        self.current_iter = 0
        self.max_iters = training_config['max_iters']
        self.eval_every = training_config['eval_every']
        self.batch_idx = 0

        self.model_name = "DiffFormer" if model_config.is_diff else "Transformer"
        self.checkpoint_dir = f"checkpoints/{self.model_name}"

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self._load_checkpoint()
        self._load_dataloader(training_config)

    def _load_dataloader(self, training_config):
        response = input("Load dataset from scratch? (y/n) ")

        if response.lower() == 'y':
            train_skip_samples = None
        elif response.lower() == 'n':
            train_skip_samples = self.batch_idx
        else:
            raise ValueError("Invalid input. Please enter y or n.")


        batch_size = training_config['batch_size']
        n_ctx = self.model_config.n_ctx

        self.train_loader, self.val_loader = load_dataloader(
            batch_size,
            "HuggingFaceTB/SmolLM-135M",
            n_ctx,
            num_val_samples=self.training_config['num_val_samples'],
            train_skip_samples=train_skip_samples
        )


    def _save_checkpoint(self,):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_iteration": self.current_iter,
            "batch_idx": self.batch_idx
        }

        checkpoint_pth = os.path.join(self.checkpoint_dir, f"Iteration_{self.current_iter}.pth")
        torch.save(checkpoint, checkpoint_pth)
        print(
            f"Iteration {self.current_iter} | Training checkpoint saved at {checkpoint_pth}"
        )

    def _prompt_checkpoint(self):
        '''Ask user which checkpoint to load'''

        files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("Iteration_") and f.endswith(".pth")
        ]
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.pth')[0]))

        if len(files) == 0:
            print("No checkpoints found. Training from scratch")
            return None
        
        choice = int(input("Enter the index of the checkpoint to load: (-1 for latest, -2 for from scratch) "))
        if choice == -2:
            print("Clearing all checkpoints")

            for f in files:
                os.remove(os.path.join(self.checkpoint_dir, f))
            return None
        elif choice == -1:
            return int(files[-1].split('_')[1].split('.pth')[0])
        elif 1 <= choice <= len(files):
            return int(files[choice - 1].split('_')[1].split('.pth')[0])
        else:
            raise ValueError("Invalid input. Please enter a valid number.")

    def _load_checkpoint(self):

        checkpoint_idx = self._prompt_checkpoint()

        if checkpoint_idx is not None:

            path = os.path.join(self.checkpoint_dir, f"Iteration_{checkpoint_idx}.pth")

            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_iter = checkpoint["current_iteration"]
            self.batch_idx = checkpoint["batch_idx"]
            print(
                f"Checkpoint loaded from {path} at iteration {self.current_iter}"
            )
        else:
            print("Training from scratch")


    def train(self):
        self.model.train()
        while self.current_iter < self.max_iters:
            for idx, batch in enumerate(self.train_loader):
                
                if self.current_iter > self.max_iters:
                    return

                if self.current_iter % self.eval_every == 0:
                    # Evaluate Validation Loss
                    self.eval()
                    self.model.train()
                
                x = batch['input_ids'].to(self.gpu_id)
                y = batch['target_ids'].to(self.gpu_id)

                mask = batch['mask'].to(self.gpu_id)
                
                t0 = time.time()
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1), reduction='none')
                # print(loss.shape, mask.shape)
                loss = (loss * mask.view(-1)).sum() / mask.sum()

                # loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
                loss.backward()
                self.optimizer.step()
                t1 = time.time()

                self.current_iter += 1
                if self.current_iter % self.save_every == 0:
                    self.batch_idx = idx
                    self._save_checkpoint()

                if self.current_iter % 10 == 0:
                    print(f"Iter: {self.current_iter:<12} | Loss: {loss.item():<10.6f} | Time: {t1 - t0:<8.6f}")
                    wandb.log({"loss": loss.item(), "iteration": self.current_iter})
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
            
            print(f"Validation Loss (Iteration {self.current_iter}): {total_loss / count} in {count} batches")
            wandb.log({"val_loss": total_loss / count, "iteration": self.current_iter})


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    lr = training_config['learning_rate']

    config = ToyTransConfig(is_diff=training_config['architecture'] == "DiffFormer")
    model = TransModel(config)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        model_config=config,
        training_config=training_config,
        optimizer=optimizer,
        gpu_id=None,
    )

    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    main()
    

