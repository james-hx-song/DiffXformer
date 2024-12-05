import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np

import time
import os
import wandb
import yaml
import sys
from configs.config import ToyTransConfig, LM_ARGS, LMConfig
from datasets import load_dataset
from transformers import AutoTokenizer

from models.model import TransModel
from typing import Optional

# --------- Dataset Loading --------- #
def load_dataloader(
    batch_size: int,
    tokenizer_name: str,
    n_ctx: int,
    num_val_samples: Optional[int] = None,
    train_skip_samples: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
):
    if num_val_samples is None:
        num_val_samples = batch_size

    if train_skip_samples is None:
        train_skip_samples = 0

    if rank == 0:
        print("Loading Dataset...")
    # Load the dataset with streaming=True
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        split="train",
        streaming=True,  # Use streaming dataset
    )
    
    try:
        print(ds.info)
        print(f"Number of examples: {ds.info.splits['train'].num_examples}")
    except AttributeError:
        print("Dataset info is not available in streaming mode.")

    if rank == 0:
        print(f"Splitting dataset into {num_val_samples} validation samples, skipping {train_skip_samples} training samples")
    
    # Create the validation dataset by taking the first `num_val_samples` samples
    ds_val = ds.take(num_val_samples)

    # Skip validation samples and training samples to be skipped, then shard the dataset
    ds_train = ds.skip(num_val_samples + train_skip_samples)

    # Shard the training and validation datasets across processes
    ds_train = ds_train.shard(num_shards=world_size, index=rank)
    ds_val = ds_val.shard(num_shards=world_size, index=rank)

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

    # Since we're using streaming and sharding, we don't need DistributedSampler
    # Create DataLoader without sampler
    ds_train_loader = DataLoader(ds_train, batch_size=batch_size)
    ds_val_loader = DataLoader(ds_val, batch_size=batch_size)

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
        scheduler,
        device: str,
        gpu_id: Optional[int],
        rank: int,
        world_size: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.model_config = model_config
        self.training_config = training_config
        self.train_loader = self.val_loader = None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_every = training_config['save_every']
        self.current_iter = 0
        self.max_iters = training_config['max_iters']
        self.eval_every = training_config['eval_every']
        self.batch_idx = 0

        # Wrap the model with DDP
        self.model = DDP(model, device_ids=[gpu_id], output_device=gpu_id).to(self.device)

        self.model_name = "DiffFormer" if model_config.is_diff else "Transformer"
        self.checkpoint_dir = f"checkpoints/{self.model_name}"

        if not os.path.exists(self.checkpoint_dir) and self.rank == 0:
            os.makedirs(self.checkpoint_dir)

        self._load_checkpoint()
        self._load_dataloader(training_config)

    def _load_dataloader(self, training_config):
        train_skip_samples = self.batch_idx

        batch_size = training_config['batch_size']
        n_ctx = self.model_config.n_ctx

        self.train_loader, self.val_loader = load_dataloader(
            batch_size,
            "HuggingFaceTB/SmolLM-135M",
            n_ctx,
            num_val_samples=self.training_config.get('num_val_samples', batch_size),
            train_skip_samples=train_skip_samples,
            rank=self.rank,
            world_size=self.world_size,
        )


    def _save_checkpoint(self,):
        if self.rank != 0:
            return
        checkpoint = {
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_iteration": self.current_iter,
            "batch_idx": self.batch_idx
        }

        checkpoint_pth = os.path.join(self.checkpoint_dir, f"Iteration_{self.current_iter}.pth")
        torch.save(checkpoint, checkpoint_pth)
        print(
            f"Iteration {self.current_iter} | Training checkpoint saved at {checkpoint_pth}"
        )

    def _prompt_checkpoint(self):
        '''Automatically load the latest checkpoint if available'''

        files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("Iteration_") and f.endswith(".pth")
        ]
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.pth')[0]))

        if len(files) == 0:
            if self.rank == 0:
                print("No checkpoints found. Training from scratch")
            return None
        else:
            latest_checkpoint = files[-1]
            checkpoint_idx = int(latest_checkpoint.split('_')[1].split('.pth')[0])
            if self.rank == 0:
                print(f"Loading latest checkpoint: {latest_checkpoint}")
            return checkpoint_idx

    def _load_checkpoint(self):

        checkpoint_idx = self._prompt_checkpoint()

        if checkpoint_idx is not None:

            path = os.path.join(self.checkpoint_dir, f"Iteration_{checkpoint_idx}.pth")

            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            checkpoint = torch.load(path, map_location=map_location)
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.current_iter = checkpoint["current_iteration"]
            self.batch_idx = checkpoint["batch_idx"]
            if self.rank == 0:
                print(
                    f"Checkpoint loaded from {path} at iteration {self.current_iter}"
                )
        else:
            if self.rank == 0:
                print("Training from scratch")


    def train(self):
        self.model.train()
        while self.current_iter < self.max_iters:
            for idx, batch in enumerate(self.train_loader):

                if self.current_iter >= self.max_iters:
                    return

                if self.current_iter % self.eval_every == 0:
                    # Evaluate Validation Loss
                    self.eval()
                    self.model.train()

                x = batch['input_ids'].to(self.device)
                y = batch['target_ids'].to(self.device)

                mask = batch['mask'].to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1), reduction='none')
                loss = (loss * mask.view(-1)).sum() / mask.sum()

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.current_iter += 1
                if self.current_iter % self.save_every == 0:
                    self.batch_idx = idx
                    self._save_checkpoint()

                if self.current_iter % 10 == 0 and self.rank == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Iter: {self.current_iter:<12} | Loss: {loss.item():<10.6f} | LR: {lr:.6e}")
                    wandb.log({"loss": loss.item(), "iteration": self.current_iter, "lr": lr})
    def eval(self):
        self.model.eval()
        total_loss = torch.tensor(0.0).to(self.device)
        count = torch.tensor(0).to(self.device)

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['input_ids'].to(self.device)
                y = batch['target_ids'].to(self.device)

                mask = batch['mask'].to(self.device)

                output = self.model(x)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1), reduction='none')
                loss = (loss * mask.view(-1)).sum() / mask.sum()

                total_loss += loss.item()
                count += 1

        # Aggregate the total loss and count across all processes
        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

        avg_loss = total_loss / count

        if self.rank == 0:
            print(f"Validation Loss (Iteration {self.current_iter}): {avg_loss.item()} in {count.item()} batches")
            wandb.log({"val_loss": avg_loss.item(), "iteration": self.current_iter})


def main(rank, world_size):
    
        
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--size", type=str, default="122M")
    
    
    args = parser.parse_args()

    ddp_setup(rank, world_size)

    # Import training configuration
    with open(args.config, "r") as file:
        training_config = yaml.safe_load(file)

    if rank == 0:
        wandb.init(
            project=training_config['architecture'],
            config=training_config,
        )

    device = f"cuda:{rank}"
    print(f"Using device: {device}")

    max_lr = training_config['max_learning_rate']
    min_lr = training_config['min_learning_rate']
    warmup_steps = training_config['warmup_steps']
    max_iters = training_config['max_iters']

    # config = ToyTransConfig(is_diff=training_config['architecture'] == "DiffFormer")
    config = LMConfig(**LM_ARGS[args.size], is_diff=training_config['architecture'] == "DiffFormer")
    model = TransModel(config).to(device)

    if rank == 0:
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    # Define the learning rate scheduler with warmup
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < max_iters:
            return max(
                min_lr / max_lr,
                (max_iters - current_step) / float(max(1, max_iters - warmup_steps)) * (1.0 - min_lr / max_lr) + min_lr / max_lr
            )
        else:
            return min_lr / max_lr


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = Trainer(
        model=model,
        model_config=config,
        training_config=training_config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gpu_id=rank,
        rank=rank,
        world_size=world_size,
    )

    trainer.train()

    if rank == 0:
        wandb.finish()

    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)