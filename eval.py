import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import wandb
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Optional
from models.model import TransModel
from configs.config import LMConfig, LM_ARGS
import json

class Evaluator:
    def __init__(
        self,
        model: TransModel,
        load_dataloader,
        model_config,
        eval_config,
        gpu_id: Optional[int],
        args,
    ):
        self.model = model.to(gpu_id)
        self.model_config = model_config
        self.load_dataloader = load_dataloader
        self.eval_config = eval_config
        self.gpu_id = gpu_id if gpu_id is not None else 'cpu'
        self.current_iter = 0

        self._load_checkpoint(args)
        self._load_dataloader(eval_config)

    def _load_dataloader(self, eval_config):
        # For evaluation, we assume no prompt for scratch loading
        # Adjust as needed if you want the same behavior
        train_skip_samples = None
        self.train_loader, self.val_loader, self.test_loader = self.load_dataloader(
            train_skip_samples=train_skip_samples,
            **eval_config,
            **self.model_config.__dict__
        )

    def _prompt_checkpoint(self, checkpoint_dir):
        '''Ask user which checkpoint to load'''

        files = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("Iteration_") and f.endswith(".pth")
        ]
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.pth')[0]))

        if len(files) == 0:
            print("No checkpoints found.")
            return None
        
        choice = int(input(f"Loading checkpoints from '{checkpoint_dir}'. Enter the index of the checkpoint to load: (-1 for latest) "))
        if choice == -1:
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
            checkpoint = torch.load(path, map_location=self.gpu_id)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.current_iter = checkpoint.get("current_iteration", 0)
            print(f"Checkpoint loaded from {path} at iteration {self.current_iter}")
        else:
            print("No checkpoint loaded.")

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
            
            avg_val_loss = total_loss / count if count > 0 else float('inf')
            print(f"Validation Loss (Iteration {self.current_iter}): {avg_val_loss}")
            wandb.log({"val_loss": avg_val_loss, "iteration": self.current_iter})

    def eval_output(self, similarity=False, sim_model=None, tokenizer=None):
        self.model.eval()

        # Statistics for multiple choice evaluation
        mc_correct = 0
        mc_total = 0

        # Statistics for open-ended evaluation
        oe_sim_sum = 0.0
        oe_count = 0
        
        groundtruth_json = []
        predicted_json = []
        questions_json = []
        
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.val_loader)):
                if idx >= 100:
                    break
                x = batch['query'].to(self.gpu_id)
                y = batch['answer'].to(self.gpu_id)
                # mask = batch['mask'].to(self.gpu_id)
                
                # Assume generate function is implemented in the model for evaluation
                # Adjust max_length and eos_token_id as appropriate
                
                # logits = self.model(output_ids)  # Recompute logits for the generated sequence
                # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
                # loss = (loss * mask.view(-1)).sum() / mask.sum()
                if not similarity:
                    output_ids, logits_list = self.model.generate(x, 
                                                                  max_length=1, 
                                                                  eos_token_id=tokenizer.eos_token_id, 
                                                                  return_logits=True)
                    # Multiple choice scenario
                    # We assume correct_answer exists in batch and is one of 'a','b','c','d'
                    answer_tokens = [tokenizer.encode(c, add_special_tokens=False)[0] for c in ['A','B','C','D']]
                    logits_choices = logits_list[0][0, answer_tokens]
                    pred_indices = torch.argmax(logits_choices, dim=-1)
                    preds = ['A','B','C','D'][pred_indices]
                    mc_total += 1
                    if preds == tokenizer.decode(y[0]):
                        mc_correct += 1
                    
                else:
                    # Open-ended evaluation with similarity
                    output_ids, _ = self.model.generate(x, 
                                                        max_length=100, 
                                                        eos_token_id=tokenizer.eos_token_id)
                    # Decode predicted text
                    predicted_text = tokenizer.decode(output_ids[0][x.shape[1]:], skip_special_tokens=True)
                    predicted_text = predicted_text.split(".")[0].strip() if len(predicted_text.split(".")) > 1 else predicted_text
                    groundtruth_text = tokenizer.decode(batch['answer'][0], skip_special_tokens=True)
                    if "No Answer Present" in groundtruth_text:
                        continue
                    emb_pred = sim_model.encode(predicted_text, convert_to_tensor=True)
                    emb_gt = sim_model.encode(groundtruth_text, convert_to_tensor=True)
                    from sentence_transformers.util import cos_sim
                    similarities = cos_sim(emb_pred, emb_gt)
                    sim = similarities.item()
                    oe_sim_sum += sim
                    oe_count += 1
                    print(f"overall similarity: {oe_sim_sum/oe_count}")
                    
                    groundtruth_json.append({"query_id": idx, "answers": [groundtruth_text]})
                    predicted_json.append({"query_id": idx, "answers": [predicted_text]})
                    questions_json.append({"query_id": idx, "question": tokenizer.decode(x[0], skip_special_tokens=True)})
                    

            if mc_total > 0:
                mc_accuracy = mc_correct / mc_total
                print(f"Multiple Choice Accuracy: {mc_accuracy}")
                wandb.log({"mc_accuracy": mc_accuracy, "iteration": self.current_iter})

            if oe_count > 0:
                oe_avg_sim = oe_sim_sum / oe_count
                print(f"Open-Ended Average Similarity: {oe_avg_sim}")
                wandb.log({"oe_avg_sim": oe_avg_sim, "iteration": self.current_iter})
                
            # Save the results to a json file
            if similarity:
                
                with open(os.path.join("checkpoints", self.eval_config['work_dir'], "groundtruth.json"), "w") as f:
                    json.dump(groundtruth_json, f)
                with open(os.path.join("checkpoints", self.eval_config['work_dir'], "predicted.json"), "w") as f:
                    json.dump(predicted_json, f)
                with open(os.path.join("checkpoints", self.eval_config['work_dir'], "questions.json"), "w") as f:
                    json.dump(questions_json, f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--similarity", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, "r") as file:
        eval_config = yaml.safe_load(file)
    
    # Load the appropriate dataset loader
    if eval_config['dataset'] == "HuggingFaceTB/smollm-corpus":
        from dataset.smollm_corpus import load_dataloader
    elif eval_config['dataset'] == "FinQA":
        from dataset.FinQA import load_dataloader
    elif eval_config['dataset'] == "ICL":
        from dataset.ICL import load_dataloader
    elif eval_config['dataset'] == "LogiQA":
        from dataset.LogiQA import load_dataloader
    elif eval_config['dataset'] == "MSMARCO":
        from dataset.MSMARCO import load_dataloader
    else:
        raise ValueError("Invalid dataset")
        
    args.load_weight_only = eval_config.get("pretrain", False)
    args.checkpoint_path = eval_config['work_dir'] if eval_config['work_dir'] else None

    wandb.init(
        project="DiffFormer-Evaluation",
        config=eval_config,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model config and model
    eval_config['batch_size'] = 1  # For evaluation, we use batch size 1
    config = LMConfig(**LM_ARGS[eval_config['size']], is_diff=eval_config['architecture'] == "DiffFormer")
    model = TransModel(config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    evaluator = Evaluator(
        model=model,
        load_dataloader=load_dataloader,
        model_config=config,
        eval_config=eval_config,
        gpu_id=device,
        args=args
    )

    # Perform evaluation
    # evaluator.eval()

    # If needed, setup similarity model and tokenizer for eval_output
    tokenizer = AutoTokenizer.from_pretrained(eval_config["tokenizer_name"])
    from sentence_transformers import SentenceTransformer
    sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Evaluate outputs (multiple choice or open-ended)
    evaluator.eval_output(similarity=args.similarity, sim_model=sim_model, tokenizer=tokenizer)

    wandb.finish()

if __name__ == "__main__":
    main()
    