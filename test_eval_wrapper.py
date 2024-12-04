from transformers import AutoTokenizer
from models.model import TransModel
from models.transformerWrapper import TransformerWrapper
from config import StableLMConfig, ToyTransConfig

config = ToyTransConfig(n_ctx=n_ctx)
model = TransModel(config)
tokenizer_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

wrapper = TransformerWrapper(model, tokenizer_name)

requests = [("context1", "continuation1"), ("context2", "continuation2")]
res = wrapper.loglikelihood(requests)
print(res)

requests = [("input1", {"until": "stop1"}), ("input2", {"until": "stop2"})]
res = wrapper.greedy_until(requests)
print(res)

requests = ["input1", "input2"]
res = lm.loglikelihood_rolling(requests)
print(res)