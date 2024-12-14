import argparse
from transformers import AutoTokenizer
from models.model import TransModel
from models.transformerWrapper import TransformerWrapper
from configs.config import LMConfig, ToyTransConfig, LM_ARGS
from lm_eval.api.instance import Instance
import lm_eval
import torch
import json
import datasets

parser = argparse.ArgumentParser(description="Evaluate a model")
parser.add_argument("--task", type=str, default="openbookqa")
parser.add_argument("--model", type=str, default="DiffFormer")
args = parser.parse_args()

task = args.task
model = args.model

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

device = "cuda" if torch.cuda.is_available() else "cpu"
is_diff = model == "DiffFormer"
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

print("Starting")

print(f"Task: {task}")

config = LMConfig(**LM_ARGS["122M"], is_diff=is_diff, n_ctx=2048)

name = "DiffFormer" if is_diff else "Transformer"

model = TransModel(config)
checkpoint = torch.load(f"checkpoints/{name}/Iteration_40000_bk.pth", map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
print("Model ready")


tokenizer_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

wrapper = TransformerWrapper(
    model=model, 
    tokenizer=tokenizer,
    batch_size=4,
    device=device,
    model_type="custom"
)

# requests = [("context1", "continuation1"), ("context2", "continuation2")]
# requests = [Instance("loglikelihood", {}, ( "context1", "continuation1"), 0), Instance("loglikelihood", {},( "context1", "continuation1"), 1)]
# res = wrapper.loglikelihood(requests)
# print(res)

# requests = [Instance("generate_until", {}, ("input1", {"until": ["stop1"], "max_gen_toks": 10}), 3), Instance("generate_until", {}, ("input2", {"until": ["stop2"], "max_gen_toks": 20}), 4)]
# res = wrapper.generate_until(requests)
# print(res)

# requests = ["input1", "input2"]
# res = wrapper.loglikelihood_rolling(requests)
# print(res)

task_manager = lm_eval.tasks.TaskManager()

results = lm_eval.simple_evaluate(
    model=wrapper,
    tasks=[task],
    num_fewshot=0,
    task_manager=task_manager, 
    device=device,
    cache_requests=True
)

with open(f'{task}_{name}_results.json', 'w') as f:
    json.dump(results, f, indent=4)
