from torch.utils.data import Dataset, DataLoader
from configs.config import StableLMConfig, ToyTransConfig
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional
import torch.nn.functional as F


# --------- Dataset Loading --------- #
def load_dataloader(
    batch_size: int,
    tokenizer_name: str,
    n_ctx: int,
    num_val_samples: Optional[int] = None,
    train_skip_samples: Optional[int] = None,
    **kwargs
):
    
    if num_val_samples is None:
        num_val_samples = batch_size

    if train_skip_samples is None:
        train_skip_samples = 0
    
    print("Loading Dataset...")
    # Load the dataset
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        split="train",
        streaming=True,
    )
    
    try:
        print(ds.info)
        print(f"Number of examples: {ds.info.splits['train'].num_examples}")
    except AttributeError:
        print("Dataset info is not available in streaming mode.")

    print(f"Splitting dataset into {num_val_samples} validation samples, skipping {train_skip_samples} training samples")
    ds_val = ds.take(num_val_samples)
    ds_train = ds.skip(num_val_samples + train_skip_samples)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(example):
        """
        Tokenize the text into a tensor. If text is longer than context length, then
        split into smaller batches of context length. If smaller than context length,
        add padding to match context length.
        
        Args:
            example (dict): Dictionary containing 'text' key with the input text (huggingface elem)
        
        Returns:
            dict: Dictionary with 'input_ids' and 'attention_mask' tensors
         """
        tokenized = tokenizer(
            example['text'], 
            add_special_tokens=True, 
            return_tensors='pt', 
            padding=False,  # We'll handle padding manually
            truncation=False  # We'll handle truncation if needed
        )

        max_len = n_ctx + 1

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

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
        batched=True, 
        batch_size=1, 
        remove_columns=['text', 'id', 'metadata']
    )

    ds_val = ds_val.map(
        tokenize, 
        batched=True, 
        batch_size=1, 
        remove_columns=['text', 'id', 'metadata']
    )

    ds_train_loader= DataLoader(ds_train, batch_size=batch_size,)
    ds_val_loader = DataLoader(ds_val, batch_size=batch_size,)

    return ds_train_loader, ds_val_loader, None
