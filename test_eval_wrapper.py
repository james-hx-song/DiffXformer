from transformers import AutoTokenizer
from models.model import TransModel
from models.transformerWrapper import TransformerWrapper
from config import StableLMConfig, ToyTransConfig
from lm_eval.api.instance import Instance

print("Starting")
n_ctx=64
config = ToyTransConfig(n_ctx=n_ctx)
model = TransModel(config)
print("Model ready")
tokenizer_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

wrapper = TransformerWrapper(model, tokenizer_name)

# requests = [("context1", "continuation1"), ("context2", "continuation2")]
# requests = [Instance("loglikelihood", {}, ( "context1", "continuation1"), 0), Instance("loglikelihood", {},( "context1", "continuation1"), 1)]
# res = wrapper.loglikelihood(requests)
# print(res)

requests = [Instance("generate_until", {}, ("input1", {"until": ["stop1"], "max_gen_toks": 10}), 3), Instance("generate_until", {}, ("input2", {"until": ["stop2"], "max_gen_toks": 20}), 4)]
res = wrapper.generate_until(requests)
print(res)

requests = ["input1", "input2"]
res = wrapper.loglikelihood_rolling(requests)
print(res)