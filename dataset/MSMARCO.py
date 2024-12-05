import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch
from typing import Optional
from tqdm import tqdm

# --------- Custom Dataset Class --------- #
class CustomJSONDataset(Dataset):
    def __init__(self, file_path, tokenizer_name, n_ctx):
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.n_ctx = n_ctx

        # Read the JSON file and store the data references
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.ans = self.data['answers']
        self.query = self.data['query']
        self.passages = self.data['passages']

        # Store the keys for indexing
        self.keys = list(self.ans.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        # Process the sample on-the-fly
        text = ''
        passage_idx = 1
        for p in self.passages[key]:
            if p['is_selected'] == 1:
                text += f"Passage {passage_idx}: {p['passage_text']} "
                passage_idx += 1
        
        text += f"Query: {self.query[key]} Answer: {self.ans[key]}"
        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding=False,
            truncation=False
        )

        max_len = self.n_ctx + 1

        input_ids = tokenized['input_ids']  # shape [1, seq_len]
        attention_mask = tokenized['attention_mask']  # shape [1, seq_len]

        curr_length = input_ids.shape[1]
        next_multiple = ((curr_length + max_len - 1) // max_len) * max_len

        padding_length = next_multiple - curr_length
        padded_input_ids = F.pad(
            input_ids,
            (0, padding_length),
            mode='constant',
            value=self.tokenizer.pad_token_id
        )

        padded_attention_mask = F.pad(
            attention_mask,
            (0, padding_length),
            mode='constant',
            value=0
        )

        reshaped_input_ids = padded_input_ids.view(-1, max_len)
        reshaped_attention_mask = padded_attention_mask.view(-1, max_len)

        input_ids = reshaped_input_ids[:, :-1]
        target_ids = reshaped_input_ids[:, 1:]

        mask = reshaped_attention_mask[:, 1:]

        assert input_ids.shape[1] == self.n_ctx, f"Input shape: {input_ids.shape}"
        assert target_ids.shape[1] == self.n_ctx, f"Target shape: {target_ids.shape}"

        return {
            'input_ids': input_ids,
            'mask': mask,
            'target_ids': target_ids
        }

# --------- DataLoader Function --------- #
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

    print("Loading Datasets...")

    # Create dataset instances
    ds_train = CustomJSONDataset(train_file, tokenizer_name, n_ctx)
    ds_val = CustomJSONDataset(val_file, tokenizer_name, n_ctx)
    # ds_test = CustomJSONDataset(test_file, tokenizer_name, n_ctx)

    # Optionally skip samples in the training set
    if train_skip_samples > 0:
        ds_train = torch.utils.data.Subset(ds_train, range(train_skip_samples, len(ds_train)))

    def collate_fn(batch):
        """
        Collate function to handle batching of variable-length sequences.
        """
        input_ids = []
        masks = []
        target_ids = []

        for item in batch:
            input_ids.append(item['input_ids'])
            masks.append(item['mask'])
            target_ids.append(item['target_ids'])

        # Concatenate along batch dimension
        input_ids = torch.cat(input_ids, dim=0)
        masks = torch.cat(masks, dim=0)
        target_ids = torch.cat(target_ids, dim=0)

        return {'input_ids': input_ids, 'mask': masks, 'target_ids': target_ids}

    ds_train_loader = DataLoader(ds_train, batch_size=batch_size, num_workers=8, collate_fn=collate_fn)
    ds_val_loader = DataLoader(ds_val, batch_size=batch_size, num_workers=8, collate_fn=collate_fn)
    # ds_test_loader = DataLoader(ds_test, batch_size=batch_size, collate_fn=collate_fn)

    return ds_train_loader, ds_val_loader, None

# Example usage
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
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Mask shape:", batch['mask'].shape)
        print("Target IDs shape:", batch['target_ids'].shape)
        break