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
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        self.model.eval()
        #...
        toret = []
        BATCH_SIZE = 16
        for index in tqdm(range((len(requests) - 1) // BATCH_SIZE + 1)):
            curr_requests = requests[index * BATCH_SIZE: min((index + 1) * BATCH_SIZE, len(requests))]
            inputs = [request.args[0] for request in curr_requests]
            expected_outputs = [request.args[1] for request in curr_requests]
            input_plus_output = [inputs[x] + expected_outputs[x] for x in range(len(inputs))]
            expected_outputs = [self.tokenizer(expected_output, return_tensors="pt")['input_ids'] for expected_output in expected_outputs]
            # print(expected_outputs)
            tokenizer_output = self.tokenizer(inputs, return_tensors="pt", padding=True)
            inputs = tokenizer_output['input_ids']
            inputs_attn = tokenizer_output['attention_mask']

            tokenizer_output = self.tokenizer(input_plus_output, return_tensors="pt", padding=True)
            input_plus_output = tokenizer_output['input_ids'].to(self.device)
            input_plus_output_attn = tokenizer_output['attention_mask']

            results = self.model.forward(input_plus_output)
            log_softmax = nn.LogSoftmax(dim = 1)
            output_logits = log_softmax(results).cpu()
            for request in range(len(expected_outputs)):
                is_greedy = True
                input_len = torch.sum(inputs_attn[request])
                # input_len = inputs_attn[request].index(0)
                # if input_len == inputs_attn.size[1]:
                #     input_len = inputs.shape[1]
                # input_plus_output_len = input_plus_output_attn[request].index(0)
                input_plus_output_len = torch.sum(input_plus_output[request])
                # if input_plus_output_len < 0:
                #     input_plus_output_len = input_plus_output.shape[1]
                log_probability = 0
                for index, expected_token in enumerate(expected_outputs[request][0]):
                    if index < input_plus_output_len:
                        log_probability += output_logits[request][index + input_len - 1][expected_token].item()
                        if is_greedy:
                            for prob in output_logits[request][index]:
                                if prob.item() > output_logits[request][index][expected_token].item():
                                    is_greedy = False
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