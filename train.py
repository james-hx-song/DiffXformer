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
from configs.config import StableLMConfig, ToyTransConfig, LMConfig, LM_ARGS
from datasets import load_dataset
from transformers import AutoTokenizer

from models.model import TransModel
from typing import Optional
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: TransModel,
        load_dataloader,
        model_config,
        training_config,
        optimizer: torch.optim.Optimizer,
        scheduler, 
        gpu_id: Optional[int],
        args,
    ):
        self.model = model.to(gpu_id)

        # if gpu_id is not None:
        #     self.model = DDP(model, device_ids=[gpu_id])

        self.model_config = model_config
        self.load_dataloader = load_dataloader
        self.training_config = training_config
        self.train_loader = self.val_loader = None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gpu_id = gpu_id if gpu_id is not None else 'cpu'
        self.save_every = training_config['save_every']
        self.current_iter = 0
        self.max_iters = training_config['max_iters']
        self.eval_every = training_config['eval_every']
        self.batch_idx = 0

        self._load_checkpoint(args)
        self._load_dataloader(training_config)

    def _load_dataloader(self, training_config):
        response = input("Load dataset from scratch? (y/n) ")

        if response.lower() == 'y':
            train_skip_samples = None
        elif response.lower() == 'n':
            train_skip_samples = self.batch_idx
        else:
            raise ValueError("Invalid input. Please enter y or n.")


        # batch_size = training_config['batch_size']
        # n_ctx = self.model_config.n_ctx

        self.train_loader, self.val_loader, self.test_loader = self.load_dataloader(
            train_skip_samples=train_skip_samples,
            **training_config,
            **self.model_config.__dict__
        )


    def _save_checkpoint(self,):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
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

    def _prompt_checkpoint(self, checkpoint_dir):
        '''Ask user which checkpoint to load'''

        files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("Iteration_") and f.endswith(".pth")
        ]
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.pth')[0]))

        if len(files) == 0:
            print("No checkpoints found. Training from scratch")
            return None
        
        choice = int(input(f"Loading checkpoints from '{checkpoint_dir}'. Enter the index of the checkpoint to load: (-1 for latest, -2 for from scratch) "))
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

    def _load_checkpoint(self, args):
        
        checkpoint_path = args.checkpoint_path
        load_weight_only = args.load_weight_only
        
        model_name = "DiffFormer" if self.model_config.is_diff else "Transformer"
        self.checkpoint_dir = f"checkpoints/{checkpoint_path}" if checkpoint_path else f"checkpoints/{model_name}"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        checkpoint_idx = self._prompt_checkpoint(self.checkpoint_dir)

        if checkpoint_idx is not None:

            path = os.path.join(self.checkpoint_dir, f"Iteration_{checkpoint_idx}.pth")

            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if not load_weight_only:
                print("Resuming training from checkpoint")
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.current_iter = checkpoint["current_iteration"]
                self.batch_idx = checkpoint["batch_idx"]
            print(
                f"Checkpoint loaded from {path} at iteration {self.current_iter}"
            )
        else:
            print("Training from scratch")


    def train(self):
        self.model.train()
        epoch = 0
        while self.current_iter < self.max_iters:
            epoch += 1
            print(f"Epoch: {epoch}")
            
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
                self.scheduler.step()
                t1 = time.time()

                self.current_iter += 1
                if self.current_iter % self.save_every == 0:
                    self.batch_idx = idx
                    self._save_checkpoint()

                if self.current_iter % 10 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Iter: {self.current_iter:<12} | Loss: {loss.item():<10.6f} | Time: {t1 - t0:<8.6f} | LR: {lr:.6e}")
                    wandb.log({"loss": loss.item(), "iteration": self.current_iter})
                    
    def fine_tuning(self):
        """
        For fine-tuning the model. Instead of prediciting the next token, we predict answer only.
        """
        self.model.train()
        epoch = 0
        while self.current_iter < self.max_iters:
            epoch += 1
            print(f"Epoch: {epoch}")
            
            for idx, batch in enumerate(self.train_loader):
                
                if self.current_iter > self.max_iters:
                    return

                if self.current_iter % self.eval_every == 0 and self.current_iter > 0:
                    # Evaluate Validation Loss
                    self.eval()
                    self.model.train()
                
                x = batch['input_ids'].to(self.gpu_id)
                y = batch['target_ids'].to(self.gpu_id)
                answer_mask = batch['answer_mask'].to(self.gpu_id)
                mask = batch['mask'].to(self.gpu_id)
                
                t0 = time.time()
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1), reduction='none')
                # print(loss.shape, mask.shape)
                loss = (loss * mask.view(-1) * answer_mask.view(-1)).sum() / (mask * answer_mask).sum()

                # loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                t1 = time.time()

                self.current_iter += 1
                if self.current_iter % self.save_every == 0:
                    self.batch_idx = idx
                    self._save_checkpoint()

                if self.current_iter % 10 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Iter: {self.current_iter:<12} | Loss: {loss.item():<10.6f} | Time: {t1 - t0:<8.6f} | LR: {lr:.6e}")
                    wandb.log({"loss": loss.item(), "iteration": self.current_iter})
                    
    def eval(self):
        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            count = 0
            for batch in tqdm(self.val_loader):
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
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--fine_tuning", action="store_true")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as file:
        training_config = yaml.safe_load(file)
    
    if training_config['dataset'] == "HuggingFaceTB/smollm-corpus":
        from dataset.smollm_corpus import load_dataloader
    elif training_config['dataset'] == "FinQA":
        from dataset.FinQA import load_dataloader
    elif training_config['dataset'] == "ICL":
        from dataset.ICL import load_dataloader
    elif training_config['dataset'] == "LogiQA":
        from dataset.LogiQA import load_dataloader
    elif training_config['dataset'] == "MSMARCO":
        from dataset.MSMARCO import load_dataloader
    else:
        raise ValueError("Invalid dataset")
        
    args.load_weight_only = training_config.get("pretrain", False)
    args.checkpoint_path = training_config['work_dir'] if training_config.get('work_dir') else None
    


    wandb.init(
        project="DiffFormer",
        config=training_config,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    max_lr = training_config['max_learning_rate']
    min_lr = training_config['min_learning_rate']
    warmup_steps = training_config['warmup_steps']
    max_iters = training_config['max_iters']

    # config = ToyTransConfig(is_diff=training_config['architecture'] == "DiffFormer")
    print(training_config['architecture'])
    config = LMConfig(**LM_ARGS[training_config['size']], is_diff=training_config['architecture'] == "DiffFormer")
    model = TransModel(config)

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
        load_dataloader=load_dataloader,
        model_config=config,
        training_config=training_config,
        optimizer=optimizer,
        scheduler=scheduler,
        gpu_id=device,
        args=args
    )
    
    if args.fine_tuning:
        trainer.fine_tuning()
    else:
        trainer.train()

    wandb.finish()

if __name__ == "__main__":
    main()
    
    # from sentence_transformers import SentenceTransformer
    # tokenizer = AutoTokenizer.from_pretrained(training_config["tokenizer_name"])  # or your model tokenizer
    # sim_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    

