import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np

import time
import os
import yaml
import sys
from configs.config import ToyTransConfig, LM_ARGS, LMConfig
from datasets import load_dataset
from transformers import AutoTokenizer

from models.model import TransModel
from typing import Optional


# ----------------------------
# New Dataset and Model Setup
# ----------------------------


class PhiMLP(nn.Module):
    def __init__(self, d_dim, hidden_d_dim, num_layers):
        super(PhiMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(d_dim, hidden_d_dim))
        layers.append(nn.LeakyReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_d_dim, hidden_d_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_d_dim, hidden_d_dim))
        self.mlp = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class InContextDataset(Dataset):
    """
    Infinite (or effectively infinite) training dataset and a finite validation dataset.

    If val=False, the dataset returns a very large number in __len__, simulating infinite samples.
    If val=True, __len__ returns num_val_samples, providing a fixed-size validation dataset.
    """

    def __init__(self, d_dim, hidden_d_dim, n_examples, tau, val=False, num_val_samples=1000):
        self.n_examples = n_examples
        self.d_dim = d_dim
        self.hidden_d_dim = hidden_d_dim
        self.tau = tau
        self.phi_mlp = PhiMLP(d_dim, hidden_d_dim, num_layers=3)
        self.val = val
        self.num_val_samples = num_val_samples

    def sample_x(self):
        # Shape: (n_examples+1, d_dim)
        return torch.rand(self.n_examples + 1, self.d_dim)

    def sample_z(self):
        # noise sample from N(0,1)
        return F.leaky_relu(torch.randn(self.n_examples + 1))

    def compute_y(self, x, z):
        with torch.no_grad():
            # w ~ N(0, tau^2)
            w = torch.randn(self.hidden_d_dim) * (self.tau**2)
            phi_x = self.phi_mlp(x)  # Shape: (n_examples+1, hidden_d_dim)
            dot_product = torch.matmul(phi_x, w)  # Shape: (n_examples+1,)
            y = dot_product + z  # Shape: (n_examples+1,)
            return y

    def build_h(self, x, y):
        # Build a "context matrix" h by interleaving x_i and a padded version of y_i
        # for i in range(n_examples). Then append x_{n_examples} at the end.
        h_matrix = []
        for i in range(self.n_examples):
            y_i = y[i].unsqueeze(0)
            y_i_padded = torch.cat([y_i, torch.zeros(self.d_dim - 1)])
            h_matrix.append(x[i])
            h_matrix.append(y_i_padded)
        # Append the last x
        h_matrix.append(x[self.n_examples])
        combined_matrix = torch.stack(h_matrix)
        # Shape: (2*n_examples + 1, d_dim)
        # We'll transpose to have shape: (d_dim, 2*n_examples + 1)
        return combined_matrix.T

    def __len__(self):
        if self.val:
            return self.num_val_samples
        else:
            # For training (val=False), return a very large number to simulate infinite samples
            return 10**12  # effectively infinite

    def __getitem__(self, idx):
        x = self.sample_x()  # (n_examples+1, d_dim)
        z = self.sample_z()  # (n_examples+1)
        y = self.compute_y(x, z)  # (n_examples+1)
        h = self.build_h(x, y)  # (d_dim, 2*n_examples+1)

        # y_target will be the last y value: y[self.n_examples]
        return {"h": h, "y": y[self.n_examples].unsqueeze(0)}


# ----------------------------
# Training Configuration
# ----------------------------

with open("config_ICL2_DX.yaml", "r") as file:
    training_config = yaml.safe_load(file)
    print(training_config)

# wandb.init(
#     project="DiffFormer",
#     config=training_config,
# )


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
        model: nn.Module,
        training_config,
        optimizer: torch.optim.Optimizer,
        gpu_id: Optional[int],
    ):
        self.model = model.to(gpu_id if gpu_id is not None else "cpu")

        if gpu_id is not None:
            self.model = DDP(model, device_ids=[gpu_id])

        self.training_config = training_config
        self.train_loader = self.val_loader = None
        self.optimizer = optimizer
        self.gpu_id = gpu_id if gpu_id is not None else "cpu"
        self.save_every = training_config["save_every"]
        self.current_iter = 0
        self.max_iters = training_config["max_iters"]
        self.eval_every = training_config["eval_every"]
        self.batch_idx = 0

        self.model_name = "DiffFormer" if training_config["architecture"] == "DiffFormer" else "Transformer"
        self.checkpoint_dir = f"checkpoints/{self.model_name}"

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self._load_dataloader(training_config)
        self._load_checkpoint()

    def _load_dataloader(self, training_config):
        d_dim = training_config.get("d_dim", 10)
        hidden_d_dim = training_config.get("hidden_d_dim", 20)
        n_examples = training_config.get("n_examples", 5)
        tau = training_config.get("tau", 1.0)
        batch_size = training_config["batch_size"]
        num_val_samples = training_config.get("num_val_samples", 1000)

        # Infinite training dataset
        train_dataset = InContextDataset(d_dim=d_dim, hidden_d_dim=hidden_d_dim, n_examples=n_examples, tau=tau, val=False)
        # Finite validation dataset
        val_dataset = InContextDataset(
            d_dim=d_dim, hidden_d_dim=hidden_d_dim, n_examples=n_examples, tau=tau, val=True, num_val_samples=num_val_samples
        )

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def _save_checkpoint(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict() if not isinstance(self.model, DDP) else self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_iteration": self.current_iter,
            "batch_idx": self.batch_idx,
        }

        checkpoint_pth = os.path.join(self.checkpoint_dir, f"Iteration_{self.current_iter}.pth")
        torch.save(checkpoint, checkpoint_pth)
        print(f"Iteration {self.current_iter} | Training checkpoint saved at {checkpoint_pth}")

    def _prompt_checkpoint(self):
        """Ask user which checkpoint to load"""
        files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("Iteration_") and f.endswith(".pth")]
        files = sorted(files, key=lambda x: int(x.split("_")[1].split(".pth")[0]))

        if len(files) == 0:
            print("No checkpoints found. Training from scratch")
            return None

        print("Found checkpoints:")
        for i, f in enumerate(files):
            iteration_num = f.split("_")[1].split(".pth")[0]
            print(f"{i+1}: {f} (Iteration {iteration_num})")
        print("-1 for latest, -2 for from scratch")
        choice = int(input("Enter the index of the checkpoint to load: "))
        if choice == -2:
            print("Clearing all checkpoints")
            for f in files:
                os.remove(os.path.join(self.checkpoint_dir, f))
            return None
        elif choice == -1:
            return int(files[-1].split("_")[1].split(".pth")[0])
        elif 1 <= choice <= len(files):
            return int(files[choice - 1].split("_")[1].split(".pth")[0])
        else:
            raise ValueError("Invalid input. Please enter a valid number.")

    def _load_checkpoint(self):
        checkpoint_idx = self._prompt_checkpoint()
        if checkpoint_idx is not None:
            path = os.path.join(self.checkpoint_dir, f"Iteration_{checkpoint_idx}.pth")
            checkpoint = torch.load(path, map_location=self.gpu_id)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_iter = checkpoint["current_iteration"]
            self.batch_idx = checkpoint["batch_idx"]
            print(f"Checkpoint loaded from {path} at iteration {self.current_iter}")
        else:
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

                h = batch["h"].to(self.gpu_id)
                y = batch["y"].to(self.gpu_id)  # shape: (batch_size, 1)

                t0 = time.time()
                self.optimizer.zero_grad()
                output = self.model(h)  # should produce (batch_size,) or (batch_size,1)
                if output.dim() == 0:
                    output = output.unsqueeze(0)
                if output.dim() == 1:
                    output = output.unsqueeze(-1)  # ensure shape (batch_size,1)

                loss = F.mse_loss(output, y)
                loss.backward()
                self.optimizer.step()
                t1 = time.time()

                self.current_iter += 1
                if self.current_iter % self.save_every == 0:
                    self.batch_idx = idx
                    self._save_checkpoint()

                if self.current_iter % 10 == 0:
                    print(f"Iter: {self.current_iter:<12} | Loss: {loss.item():<10.6f} | Time: {t1 - t0:<8.6f}")
                    # wandb.log({"loss": loss.item(), "iteration": self.current_iter})

                if self.current_iter >= self.max_iters:
                    break

    def eval(self):
        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            count = 0
            for batch in self.val_loader:
                h = batch["h"].to(self.gpu_id)
                y = batch["y"].to(self.gpu_id)
                output = self.model(h)
                if output.dim() == 0:
                    output = output.unsqueeze(0)
                if output.dim() == 1:
                    output = output.unsqueeze(-1)

                loss = F.mse_loss(output, y)
                total_loss += loss.item()
                count += 1

            val_loss = total_loss / count
            print(f"Validation Loss (Iteration {self.current_iter}): {val_loss} in {count} batches")
            # wandb.log({"val_loss": val_loss, "iteration": self.current_iter})


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    lr = training_config["learning_rate"]

    config = LMConfig(**LM_ARGS["122M"], is_diff=training_config["architecture"] == "DiffFormer")
    model = TransModel(config)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        training_config=training_config,
        optimizer=optimizer,
        gpu_id=None,
    )

    trainer.train()

    # wandb.finish()


if __name__ == "__main__":
    main()
