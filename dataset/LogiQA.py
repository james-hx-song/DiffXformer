from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer
from typing import Optional
import torch.nn.functional as F
import torch

# --------- Helper Function to Load Data --------- #
def load_data_from_file(file_path):
    """
    Reads a dataset file and parses it into a list of examples.
    Each example is a dictionary with 'text' and 'label' keys.
    """
    examples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_example = {}
    current_text = ''
    ans = ''

    for line in lines:
        line = line.strip()
        if len(line) == 1 and line.lower() in ['a', 'b', 'c', 'd']:
            # Save the previous example if it exists
            if current_text and ans:
                current_example['text'] = f"{current_text.strip()} {ans.strip()}"
                examples.append(current_example)
            
            # Start a new example
            current_example = {}
            current_text = ''
            ans = f"Answer: {line.lower()}"
        else:
            current_text += line + ' '

    # Add the last example if it exists
    if current_text and ans:
        current_example['text'] = f"{current_text.strip()} {ans.strip()}"
        examples.append(current_example)
    return examples

# --------- Dataset Loading Function --------- #
def load_dataloader(
    batch_size: int,
    tokenizer_name: str,
    n_ctx: int,
    train_file: str,
    val_file: str,
    test_file: str,
    train_skip_samples: Optional[int] = None,
    **kwargs
):
    # if num_val_samples is None:
    #     num_val_samples = batch_size

    if train_skip_samples is None:
        train_skip_samples = 0

    print("Loading Dataset...")

    # Load the data from the files
    train_examples = load_data_from_file(train_file)
    val_examples = load_data_from_file(val_file)
    test_examples = load_data_from_file(test_file)

    # Create datasets from the examples
    ds_train = HFDataset.from_list(train_examples)
    ds_val = HFDataset.from_list(val_examples)
    ds_test = HFDataset.from_list(test_examples)

    # Optionally skip and take samples
    # ds_val = ds_val.select(range(min(num_val_samples, len(ds_val))))
    # ds_test = ds_test.select(range(min(num_val_samples, len(ds_val))))
    # ds_train = ds_train.select(range(train_skip_samples, len(ds_train)))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(example):
        """
        Tokenize the text into tensors. If text is longer than context length, then
        split into smaller batches of context length. If smaller than context length,
        add padding to match context length.

        Args:
            example (dict): Dictionary containing 'text' key with the input text (string)

        Returns:
            dict: Dictionary with 'input_ids', 'mask', and 'target_ids' tensors
        """
        tokenized = tokenizer(
            example['text'],
            add_special_tokens=True,
            return_tensors='pt',
            padding=False,
            truncation=False
        )

        max_len = n_ctx + 1

        input_ids = tokenized['input_ids']  # shape [1, seq_len]
        attention_mask = tokenized['attention_mask']  # shape [1, seq_len]

        curr_length = input_ids.shape[1]
        next_multiple = ((curr_length + max_len - 1) // max_len) * max_len

        padding_length = next_multiple - curr_length
        padded_input_ids = F.pad(
            input_ids,
            (0, padding_length),
            mode='constant',
            value=tokenizer.pad_token_id
        )

        padded_attention_mask = F.pad(
            attention_mask,
            (0, padding_length),
            mode='constant',
            value=0
        )

        reshaped_input_ids = padded_input_ids.view(-1, max_len)
        reshaped_attention_mask = padded_attention_mask.view(-1, max_len)

        input = reshaped_input_ids[:, :-1]
        target = reshaped_input_ids[:, 1:]

        mask = reshaped_attention_mask[:, 1:]

        assert input.shape[1] == n_ctx, f"Input shape: {input.shape}"
        assert target.shape[1] == n_ctx, f"Target shape: {target.shape}"

        return {
            'input_ids': input,
            'mask': mask,
            'target_ids': target
        }

    ds_train = ds_train.map(
        tokenize,
        remove_columns=['text']
    )

    ds_val = ds_val.map(
        tokenize,
        remove_columns=['text']
    )
    
    ds_test = ds_test.map(
        tokenize,
        remove_columns=['text']
    )

    def collate_fn(batch):
        """
        Collate function to handle batching of variable-length sequences.
        """
        
        input_ids = torch.cat([torch.tensor(item['input_ids']) for item in batch], dim=0)
        masks = torch.cat([torch.tensor(item['mask']) for item in batch], dim=0)
        target_ids = torch.cat([torch.tensor(item['target_ids']) for item in batch], dim=0)
        return {'input_ids': input_ids, 'mask': masks, 'target_ids': target_ids}
    
    ds_train_loader = DataLoader(ds_train, batch_size=batch_size, collate_fn=collate_fn)
    ds_val_loader = DataLoader(ds_val, batch_size=batch_size, collate_fn=collate_fn)
    ds_test_loader = DataLoader(ds_test, batch_size=batch_size, collate_fn=collate_fn)

    return ds_train_loader, ds_val_loader, ds_test_loader


if __name__ == '__main__':
    batch_size = 8
    tokenizer_name = 'bert-base-uncased'
    n_ctx = 512
    train_file = 'Train.txt'
    val_file = 'Test.txt'

    ds_train_loader, ds_val_loader = load_dataloader(
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        n_ctx=n_ctx,
        train_file=train_file,
        val_file=val_file
    )   