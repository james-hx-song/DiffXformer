from models.model import MultiHeadDiffAttention, TransModel
import torch.nn as nn
from transformers import AutoTokenizer
import torch.nn.functional as F
import LM

@register_model("TransformerWrapper")
class TransformerWrapper(LM):
    #...
    def __init__(self, model:TransModel, tokenizer_name):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        #...
        toret = []
        for request in requests:
            log_probability = 0
            input, expected_output = request.args
            input = self.tokenizer(input, return_tensors="pt")
            expected_output = self.tokenizer(expected_output, return_tensors="pt")
            output_logits = nn.LogSoftmax(self.model.forward(input))
            is_greedy = True
            for expected_token in expected_output:
                # Find probability that expected_token would be chosen from output_logits
                token_id = self.tokenizer.convert_tokens_to_ids(expected_token)
                log_probability += output_logits[token_id].item()
                if is_greedy:
                    for prob in output_logits:
                        if prob.item() > output_logits[token_id].item():
                            is_greedy = False
                input.append(expected_token)
                output_logits = nn.LogSoftmax(self.model.forward(input))
            toret.append((log_probability, is_greedy))
        return toret


    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        #...
        toret = []
        for request in requests:
            log_probability = 0
            expected_output = request.args[0]
            input = []
            expected_output = self.tokenizer(expected_output, return_tensors="pt")
            output_logits = nn.LogSoftmax(self.model.forward(input))
            for expected_token in expected_output:
                # Find probability that expected_token would be chosen from output_logits
                token_id = self.tokenizer.convert_tokens_to_ids(expected_token)
                log_probability += output_logits[token_id].item()
                input.append(expected_token)
                output_logits = nn.LogSoftmax(self.model.forward(input))
            toret.append(log_probability)
        return toret


    def generate_until(self, requests: list[Instance]) -> list[str]:
        toret = []
        for request in requests:
            input, gen_controls = request.args
            keep_generating = True
            generated_tokens = []
            while keep_generating:
                # If we have exceeded max gen tokens
                if len(generated_tokens) >= gen_controls["max_gen_toks"]:
                    keep_generating = False
                # If we have made any of the stop sequences
                generated_text = self.tokenizer.convert_tokens_to_string(generated_tokens)
                for stop_seq in gen_controls["until"]:
                    if len(generated_text) > len(stop_seq) and generated_text[-len(stop_seq):] == stop_seq:
                        keep_generating = False
                        continue
                if not keep_generating:
                    continue

                # Get logits based on what we generated so far and pick next token based off of those probabilities
                output_logits = nn.LogSoftmax(self.model.forward(generated_tokens))
                probs = F.softmax(output_logits, dim=-1)
                next_token = self.transformer.convert_ids_to_tokens(torch.multinomial(probs, num_samples=1).item())
                # next_token = ""
                # highest_prob = float('-inf')
                # for token_id in range(len(output_logits)):
                #     if output_logits[token_id].item() >= highest_prob:
                #         highest_prob = output_logits[token_id].item()
                #         next_token = self.transformer.convert_ids_to_tokens(token_id)

                # Append generated token to sequence
                generated_tokens.append(next_token)
            # Append text to results
            toret.append(self.tokenizer.convert_tokens_to_string(generated_tokens))
        return toret