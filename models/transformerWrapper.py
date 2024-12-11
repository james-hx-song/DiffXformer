from models.model import MultiHeadDiffAttention, TransModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from tqdm import tqdm


# @register_model("TransformerWrapper")
class TransformerWrapper(LM):
    #...
    def __init__(self, model:TransModel, tokenizer_name, device):

        super().__init__()
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        self.model.eval()
        #...
        toret = []

        for request in tqdm(requests):
            log_probability = 0
            input, expected_output = request.args
            input = self.tokenizer(input, return_tensors="pt")['input_ids']
            expected_output = self.tokenizer(expected_output, return_tensors="pt")['input_ids']
            # print(expected_output.numpy())
            log_softmax = nn.LogSoftmax(dim = 1)
            model_input = torch.cat((input, expected_output), dim = 1) # self.tokenizer.convert_ids_to_tokens(

            model_input = model_input.to(self.device)
            results = self.model.forward(model_input)

            output_logits = log_softmax(results)
            # print(output_logits.shape)
            is_greedy = True
            for index, expected_token in enumerate(expected_output[0][:]):
                # Find probability that expected_token would be chosen from output_logits
                # print(expected_token.numpy())
                token_id = expected_token #self.tokenizer.convert_tokens_to_ids(expected_token)
                # for x in range(output_logits.shape[2]):
                #     log_probability += output_logits[0][index+input.shape[0]][x].item()
                log_probability += output_logits[0][index+input.shape[0] - 1][token_id].item()
                if is_greedy:
                    for prob in output_logits[0][index]:
                        if prob.item() > output_logits[0][index][token_id].item():
                            is_greedy = False
                # input = torch.cat(input, expected_token, dim = 0)
                # output_logits = log_softmax(self.model.forward(input))
            toret.append((log_probability, is_greedy))
        return toret


    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        toret = []
        for request in tqdm(requests):
            log_probability = 0
            input, expected_output = request.args
            input = self.tokenizer(input, return_tensors="pt")['input_ids']
            expected_output = self.tokenizer(expected_output, return_tensors="pt")['input_ids']
            # print(expected_output.numpy())
            log_softmax = nn.LogSoftmax(dim = 1)
            # results = self.model.forward(input)
            results = self.model.forward(torch.cat((input, expected_output), dim = 1))

            output_logits = log_softmax(results)
            # print(output_logits.shape)
            is_greedy = True
            for index, expected_token in enumerate(input[0][1:]):
                # print(expected_token.numpy())
                token_id = expected_token #self.tokenizer.convert_tokens_to_ids(expected_token)
                log_probability += output_logits[0][index-1][token_id].item()
                if is_greedy:
                    for prob in output_logits[0][index]:
                        if prob.item() > output_logits[0][index - 1][token_id].item():
                            is_greedy = False
            for index, expected_token in enumerate(expected_output[0]):
                # Find probability that expected_token would be chosen from output_logits
                print(expected_token.numpy())
                token_id = expected_token #self.tokenizer.convert_tokens_to_ids(expected_token)
                log_probability += output_logits[0][index+input.shape[0] - 1][token_id].item()
                if is_greedy:
                    for prob in output_logits[0][index]:
                        if prob.item() > output_logits[0][index][token_id].item():
                            is_greedy = False
            toret.append(log_probability)
        return toret


    def generate_until(self, requests: list[Instance]) -> list[str]:
        toret = []
        for request in tqdm(requests):
            input, gen_controls = request.args
            keep_generating = True
            generated_tokens = self.tokenizer(input, return_tensors="pt")['input_ids'].flatten().tolist()
            while keep_generating:
                # print(generated_tokens)
                generated_token_text = self.tokenizer.convert_ids_to_tokens(generated_tokens)
                # print(generated_token_text)
                # If we have exceeded max gen tokens
                if len(generated_tokens) >= gen_controls.get("max_gen_toks", 100):
                    keep_generating = False
                 # TODO: Last token is None sometimes, not sure if this is bug or because last token in generated_tokens is stop token
                elif generated_token_text[-1] == None:
                    generated_tokens.pop()
                    keep_generating = False
                # If we have made any of the stop sequences
                else:
                    generated_text = self.tokenizer.convert_tokens_to_string(generated_token_text)
                    for stop_seq in gen_controls["until"]:
                        if len(generated_text) > len(stop_seq) and generated_text[-len(stop_seq):] == stop_seq:
                            print(stop_seq)
                            print(generated_text[-len(stop_seq):])
                            keep_generating = False
                            continue
                if not keep_generating:
                    continue

                # Get logits based on what we generated so far and pick next token based off of those probabilities
                model_input = torch.tensor(generated_tokens).reshape(-1, 1)
                log_softmax = nn.LogSoftmax(dim = -1)
                output_logits = log_softmax(self.model.forward(model_input))
                probs = F.softmax(output_logits, dim=-1)
                next_token = torch.multinomial(probs[-1][0], num_samples=1).item()
                # Append generated token to sequence
                generated_tokens.append(next_token)
            # Append text to results
            print("finished one request")
            toret.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(generated_tokens)))
        return toret