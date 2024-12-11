import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np

import time
import os
import wandb
import yaml
import sys
from configs.config import ToyTransConfig, LM_ARGS, LMConfig

from models.model import TransModel
from typing import Optional, Iterator

import matplotlib.pyplot as plt

import json

# ----------------------------
# New Dataset and Model Setup
# ----------------------------


class RandomMLP:
    """
    A manually implemented MLP with fixed random weights and LeakyReLU activation.
    This replaces PhiMLP but retains the multi-layer structure and nonlinearity.
    """

    def __init__(self, d_dim, hidden_d_dim, num_layers, seed=42):
        torch.manual_seed(seed)  # Ensure reproducibility
        self.num_layers = num_layers
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        for i in range(num_layers):
            in_dim = d_dim if i == 0 else hidden_d_dim
            out_dim = hidden_d_dim
            weight = torch.randn(in_dim, out_dim)
            nn.init.kaiming_normal_(weight)  # Use Kaiming initialization
            bias = torch.zeros(out_dim)
            self.weights.append(weight)
            self.biases.append(bias)

        # Convert to tensors on the appropriate device during use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, device):
        """Move weights and biases to the specified device."""
        self.device = device
        self.weights = [w.to(device) for w in self.weights]
        self.biases = [b.to(device) for b in self.biases]

    def forward(self, x):
        """
        Forward pass through the MLP.
        """
        for i in range(self.num_layers):
            x = torch.matmul(x, self.weights[i]) + self.biases[i]
            if i < self.num_layers - 1:  # Apply LeakyReLU except on the last layer
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        return x


def sample_x(n_examples, d_dim, device):
    # Shape: (n_examples+1, d_dim)
    return torch.rand(n_examples, d_dim, device=device)


def sample_z(n_examples, device, sigma):
    # noise sample from N(0,1)
    return torch.randn(n_examples, device=device) * sigma


def compute_y(x, z, random_mlp, tau, is_eval=False):
    """
    Updated compute_y to use the RandomMLP instead of PhiMLP.
    """
    # w ~ N(0, tau^2)
    w = torch.randn(random_mlp.weights[-1].size(1), device=x.device) * (tau**2)
    phi_x = random_mlp.forward(x)  # Shape: (n_examples, hidden_d_dim)
    dot_product = torch.matmul(phi_x, w)  # Shape: (n_examples,)
    if not is_eval:
        y = dot_product + z  # Shape: (n_examples,)
    else:
        y = dot_product  # Shape: (n_examples,)
    return y


@torch.no_grad()
def build_h(x, y, n_examples, d_dim, is_eval=False):
    # Vectorized build_h:
    # We have N = n_examples, D = d_dim.
    # h shape before transpose: (2N+1, D)
    # Even indices 0, 2, 4 ... contain x[i]
    # Odd indices 1, 3, 5 ... contain y[i] followed by zeros
    N = n_examples
    D = d_dim

    h_matrix = torch.zeros((2 * N, D), device=x.device, dtype=x.dtype)
    # Fill even rows with x[:N]
    h_matrix[0 : 2 * N : 2] = x[:N]
    # Fill odd rows with y[:N] in the first column
    h_matrix[1 : 2 * N : 2, 0] = y[:N]

    if is_eval:
        return h_matrix
    else:
        return h_matrix


class InContextIterableDataset(IterableDataset):
    """
    Infinite training dataset implemented as an IterableDataset.
    """

    def __init__(self, d_dim, hidden_d_dim, n_examples, tau, num_layers=2, seed=42, sigma=0.1):
        super().__init__()
        self.n_examples = n_examples
        self.d_dim = d_dim
        self.hidden_d_dim = hidden_d_dim
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma = sigma

        # Initialize a RandomMLP
        self.random_mlp = RandomMLP(d_dim, hidden_d_dim, num_layers, seed=seed)
        self.random_mlp.to(self.device)

    def __iter__(self) -> Iterator[dict]:
        while True:
            x = sample_x(self.n_examples, self.d_dim, self.device)
            z = sample_z(self.n_examples, self.device, sigma=self.sigma)
            y = compute_y(x, z, self.random_mlp, self.tau, is_eval=False)
            h = build_h(x, y, self.n_examples, self.d_dim)
            # "h" and "y" are on GPU already
            yield {"h": h, "y": y}


class InContextValDataset(Dataset):
    """
    Finite validation dataset.
    """

    def __init__(self, d_dim, hidden_d_dim, n_examples, tau, num_val_samples, num_layers=2, seed=42):
        super().__init__()
        self.d_dim = d_dim
        self.hidden_d_dim = hidden_d_dim
        self.n_examples = n_examples
        self.tau = tau
        self.num_val_samples = num_val_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize a RandomMLP
        self.random_mlp = RandomMLP(d_dim, hidden_d_dim, num_layers, seed=seed)
        self.random_mlp.to(self.device)

    def __len__(self):
        return self.num_val_samples

    @torch.no_grad()
    def __getitem__(self, idx):
        x = sample_x(self.n_examples, self.d_dim, self.device)
        z = sample_z(self.n_examples, self.device, sigma=0)
        y = compute_y(x, z, self.random_mlp, self.tau, is_eval=True)
        h = build_h(x, y, self.n_examples, self.d_dim, is_eval=True)
        return {"h": h, "y": y}


# ----------------------------
# Training Configuration
# ----------------------------


# wandb.init(
#     project="DiffFormer",
#     config=training_config,
# )


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(
        self, model: nn.Module, training_config, optimizer: torch.optim.Optimizer, gpu_id: Optional[int], model_name: str, sigma: float
    ):
        self.training_config = training_config
        self.train_loader = self.val_loader = None
        self.optimizer = optimizer
        self.save_every = training_config["save_every"]
        self.current_iter = 0
        self.max_iters = training_config["max_iters"]
        self.eval_every = training_config["eval_every"]
        self.batch_idx = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sigma = sigma

        # self.model_name = "DiffFormer" if training_config["architecture"] == "DiffFormer" else "Transformer"
        self.model_name = model_name
        self.checkpoint_dir = f"checkpoints/{self.model_name}"

        self.model = model.to(self.device)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Initialize loss tracking lists
        self.train_losses = []
        self.val_losses = []

        self._load_dataloader(training_config)
        self._load_checkpoint()

    def _load_dataloader(self, training_config):
        d_dim = training_config.get("d_dim", 10)
        hidden_d_dim = training_config.get("hidden_d_dim", 20)
        n_examples = training_config.get("n_examples", 5)
        tau = training_config.get("tau", 1.0)
        batch_size = training_config["batch_size"]
        num_layers = training_config["num_mlp_layers"]
        # num_val_samples = training_config.get("num_val_samples", 1000)

        num_val_samples = 15600

        # Infinite training dataset as IterableDataset
        train_dataset = InContextIterableDataset(
            d_dim=d_dim, hidden_d_dim=hidden_d_dim, n_examples=n_examples, tau=tau, num_layers=num_layers, sigma=self.sigma
        )
        # Finite validation dataset
        val_dataset = InContextValDataset(
            d_dim=d_dim, hidden_d_dim=hidden_d_dim, n_examples=n_examples, tau=tau, num_val_samples=num_val_samples, num_layers=num_layers
        )

        # No shuffling needed for infinite dataset, just iterative sampling
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def _save_checkpoint(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict() if not isinstance(self.model, DDP) else self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_iteration": self.current_iter,
            "batch_idx": self.batch_idx,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        checkpoint_pth = os.path.join(self.checkpoint_dir, f"Iteration_{self.current_iter}.pth")
        torch.save(checkpoint, checkpoint_pth)
        print(f"Iteration {self.current_iter} | Training checkpoint saved at {checkpoint_pth}")

    def _prompt_checkpoint(self):
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
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_iter = checkpoint["current_iteration"]
            self.batch_idx = checkpoint["batch_idx"]

            # Restore losses if available
            if "train_losses" in checkpoint:
                self.train_losses = checkpoint["train_losses"]
            if "val_losses" in checkpoint:
                self.val_losses = checkpoint["val_losses"]

            print(f"Checkpoint loaded from {path} at iteration {self.current_iter}")
        else:
            print("Training from scratch")

    def train(self):
        self.model.train()
        scaler = torch.amp.GradScaler()  # For mixed precision training
        while self.current_iter < self.max_iters:
            for idx, batch in enumerate(self.train_loader):
                if self.current_iter >= self.max_iters:
                    return

                if self.current_iter % self.eval_every == 0:
                    self.eval()
                    self.model.train()

                h = batch["h"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)

                t0 = time.time()
                self.optimizer.zero_grad()

                # Use mixed precision for forward pass and loss computation
                with torch.amp.autocast(device_type=self.device):
                    output = self.model(h)  # output shape: (batch_size, T, 1)
                    output = output[:, ::2].squeeze(-1)

                    loss = F.mse_loss(output, y)

                # Backpropagation with mixed precision
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                t1 = time.time()

                # Record training loss
                self.train_losses.append(loss.item())

                self.current_iter += 1

                if self.current_iter % 100 == 0 or self.current_iter == 1:
                    print(f"Iter: {self.current_iter:<12} | Loss: {loss.item():<10.6f} | Time: {t1 - t0:<8.6f}")

                if self.current_iter % self.save_every == 0:
                    self.batch_idx = idx
                    self._save_checkpoint()

                if self.current_iter >= self.max_iters:
                    break

    def eval(self):
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to GPU
                h = batch["h"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device):  # Mixed precision in evaluation
                    output = self.model(h)  # Pass the full input through the model
                    output = output[:, ::2].squeeze(-1)

                    # Only evaluate the very last example in the batch
                    y_last = y[-1:]  # Select the last target value
                    output_last = output[-1:]  # Select the last model output

                    # print("y_last shape:", y_last.shape)
                    # print("output_last shape:", output_last.shape)

                    loss = F.mse_loss(output_last, y_last)
                    total_loss += loss.item()
                    count += 1

        val_loss = total_loss / count
        self.val_losses.append(val_loss)
        print(f"Validation Loss (Iteration {self.current_iter}) for last examples: {val_loss} in {count} batches")

    def test(self):
        self.model.eval()

        # Extract parameters from training_config
        n_examples_max = self.training_config.get("n_examples", 5)
        tau = self.training_config.get("tau", 1.0)
        d_dim = self.training_config.get("d_dim", 10)
        hidden_d_dim = self.training_config.get("hidden_d_dim", 20)
        # Using the same number of validation samples as in _load_dataloader
        num_val_samples = 2560
        batch_size = self.training_config["batch_size"]
        num_layers = self.training_config["num_mlp_layers"]

        results = {}
        with torch.no_grad():
            for n_examples_current in range(39, n_examples_max + 1):
                # Create a new validation dataset and loader for the current n_examples
                test_dataset = InContextValDataset(
                    d_dim=d_dim,
                    hidden_d_dim=hidden_d_dim,
                    n_examples=n_examples_current,
                    tau=tau,
                    num_val_samples=num_val_samples,
                    num_layers=num_layers,
                )
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                total_loss = 0
                count = 0
                for batch in test_loader:
                    # Move data to GPU
                    h = batch["h"].to(self.device, non_blocking=True)
                    y = batch["y"].to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type=self.device):  # Mixed precision in evaluation
                        output = self.model(h)  # Pass the full input through the model
                        output = output[:, ::2].squeeze(-1)

                        # Only evaluate the very last example in the batch
                        y_last = y[-1:]  # Select the last target value
                        output_last = output[-1:]  # Select the last model output

                        # print("y_last shape:", y_last.shape)
                        # print("output_last shape:", output_last.shape)

                        loss = F.mse_loss(output_last, y_last)
                        total_loss += loss.item()
                        count += 1

                val_loss = total_loss / count if count > 0 else float("inf")
                results[n_examples_current] = val_loss
                print(f"Validation for n_examples={n_examples_current}: {val_loss}")

        results_filename = os.path.join(self.checkpoint_dir, "test_results.json")
        with open(results_filename, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {results_filename}")
        return results


def plots(file1, file2):
    """
    Reads and plots data from four JSON files with custom titles for each line.

    Args:
        file1 (str): Path to the first JSON file.
        file2 (str): Path to the second JSON file.
        file3 (str): Path to the third JSON file.
        file4 (str): Path to the fourth JSON file.
    """

    # Load JSON data
    def load_json(file):
        with open(file, "r") as f:
            return json.load(f)

    data1 = load_json(file1)
    data2 = load_json(file2)

    # Extract data for plotting
    def extract_data(data):
        x = list(map(int, data.keys()))
        y = list(data.values())
        return x, y

    x1, y1 = extract_data(data1)
    x2, y2 = extract_data(data2)

    # Plot data with custom titles
    plt.figure(figsize=(6, 5))

    plt.plot(x1, y1, label=r"Diff $\sigma = 0.1$", linewidth=3)
    plt.plot(x2, y2, label=r"Vanilla $\sigma = 0.1$", linewidth=3)

    plt.xlabel("in-context examples")
    plt.ylabel("square loss")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig("ctxlen_test_sigmas.png")


# Example usage
# plot_json_files("file1.json", "file2.json", "file3.json", "file4.json")


# Example usage
# plot_json_files("file1.json", "file2.json", "file3.json", "file4.json")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("--------------------------TRAINING DIFF TRANSFORMER 01 ---------------------------------")

    with open("config_ICL2_DX.yaml", "r") as file:
        training_config = yaml.safe_load(file)

    lr = training_config["learning_rate"]

    config = LMConfig(**LM_ARGS["122M"], is_diff=training_config["architecture"] == "DiffFormer")
    model = TransModel(config)
    model = torch.compile(model, mode="default", backend="inductor")

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model=model, training_config=training_config, optimizer=optimizer, gpu_id=None, model_name="diff_01", sigma=0.1)

    trainer.train()

    print("--------------------------TRAINING VANILLIA TRANSFORMER---------------------------------")

    with open("config_ICL2_D.yaml", "r") as file:
        training_config = yaml.safe_load(file)

    lr = training_config["learning_rate"]

    config = LMConfig(**LM_ARGS["122M"], is_diff=training_config["architecture"] == "DiffFormer")
    model = TransModel(config)
    model = torch.compile(model, mode="default", backend="inductor")

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model=model, training_config=training_config, optimizer=optimizer, gpu_id=None, model_name="trans_01", sigma=0.1)

    trainer.train()


if __name__ == "__main__":
    main()
