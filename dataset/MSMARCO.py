import json
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer
from typing import Optional
import torch.nn.functional as F
import torch

# --------- Helper Function to Load Data --------- #
def load_data_from_json(file_path):
    """
    Reads a JSON file and returns a list of examples.
    Each example is a dictionary containing various keys.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    import pdb; pdb.set_trace()
    return data

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

    if train_skip_samples is None:
        train_skip_samples = 0

    print("Loading Dataset...")

    # Load the data from the JSON files
    train_examples = load_data_from_json(train_file)
    val_examples = load_data_from_json(val_file)
    test_examples = load_data_from_json(test_file)

    # Create datasets from the examples
    ds_train = HFDataset.from_list(train_examples)
    ds_val = HFDataset.from_list(val_examples)
    ds_test = HFDataset.from_list(test_examples)

    # Optionally skip and take samples
    ds_train = ds_train.select(range(train_skip_samples, len(ds_train)))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(example):
        """
        Tokenize the text into tensors. If text is longer than context length, then
        split into smaller batches of context length. If smaller than context length,
        add padding to match context length.

        Args:
            example (dict): Dictionary containing 'qa' key with 'question' and 'answer'.

        Returns:
            dict: Dictionary with 'input_ids', 'mask', and 'target_ids' tensors.
        """
        # Combine the question and answer for training
        text = f"Question: {example['qa']['question']} Answer: {example['qa']['answer']}"
        tokenized = tokenizer(
            text,
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
        remove_columns=ds_train.column_names
    )

    ds_val = ds_val.map(
        tokenize,
        remove_columns=ds_val.column_names
    )

    ds_test = ds_test.map(
        tokenize,
        remove_columns=ds_test.column_names
    )

    def collate_fn(batch):
        """
        Collate function to handle batching of variable-length sequences.
        """
        input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
        masks = torch.cat([item['mask'] for item in batch], dim=0)
        target_ids = torch.cat([item['target_ids'] for item in batch], dim=0)
        return {'input_ids': input_ids, 'mask': masks, 'target_ids': target_ids}

    ds_train_loader = DataLoader(ds_train, batch_size=batch_size, collate_fn=collate_fn)
    ds_val_loader = DataLoader(ds_val, batch_size=batch_size, collate_fn=collate_fn)
    ds_test_loader = DataLoader(ds_test, batch_size=batch_size, collate_fn=collate_fn)

    return ds_train_loader, ds_val_loader, ds_test_loader


if __name__ == '__main__':
    batch_size = 8
    tokenizer_name = 'bert-base-uncased'  # Replace with your tokenizer
    n_ctx = 512
    train_file = 'train.json'
    val_file = 'dev.json'
    test_file = 'test.json'

    ds_train_loader, ds_val_loader, ds_test_loader = load_dataloader(
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        n_ctx=n_ctx,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file
    )
    
    for batch in ds_train_loader:
        print(batch['input_ids'].shape)
        print(batch['mask'].shape)
        print(batch['target_ids'].shape)
        break