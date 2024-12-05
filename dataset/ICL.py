import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from typing import Optional, Tuple

# --------- Custom Dataset Class --------- #
class InContextLearningDataset(Dataset):
    def __init__(
        self,
        num_examples: int,
        sequence_length: int,
        d_input: int,
        tokenizer_name: str,
        n_ctx: int,
        split: str = 'train',
    ):
        """
        Initialize the dataset.

        Args:
            num_examples (int): Number of examples in the dataset.
            sequence_length (int): Number of (xi, yi) pairs per sequence (N).
            d_input (int): Dimension of input xi.
            tokenizer_name (str): Name of the tokenizer to use.
            n_ctx (int): Context length for the model.
            split (str): Dataset split ('train', 'val', 'test').
        """
        self.num_examples = num_examples
        self.sequence_length = sequence_length  # N
        self.d_input = d_input  # Dimension of xi
        self.n_ctx = n_ctx
        self.split = split

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.model_max_length = n_ctx

        # Generate data
        self.data = [self._generate_sequence() for _ in range(num_examples)]

    def _generate_sequence(self) -> Tuple[str, str]:
        """
        Generate a single sequence of (xi, yi) pairs in the format specified in the paper.

        Returns:
            input_sequence (str): The input sequence to the model.
            target_sequence (str): The target sequence for the model to predict.
        """
        N = self.sequence_length

        # Generate a random linear model w
        w = torch.randn(self.d_input)

        # Generate xi and yi = w^T xi + noise
        xi_list = [torch.randn(self.d_input) for _ in range(N)]
        yi_list = [torch.dot(w, xi) + 0.1 * torch.randn(1) for xi in xi_list]

        # Build the input sequence H as per the paper
        # Concatenate xi and yi tokens with appropriate positional encodings

        tokens = []
        for i in range(N):
            xi_str = ' '.join([f"x{i}_{j}:{xi_list[i][j].item():.4f}" for j in range(self.d_input)])
            yi_str = f"y{i}:{yi_list[i].item():.4f}"

            tokens.append(f"{xi_str}")
            tokens.append(f"{yi_str}")

        input_sequence = ' '.join(tokens)

        # The target sequence is shifted by one position
        target_tokens = tokens[1:] + ['']  # Shifted by one position
        target_sequence = ' '.join(target_tokens)

        return input_sequence, target_sequence

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        input_sequence, target_sequence = self.data[idx]

        # Tokenize the sequences
        tokenized_input = self.tokenizer(
            input_sequence,
            add_special_tokens=True,
            truncation=True,
            max_length=self.n_ctx,
            return_tensors='pt'
        )

        tokenized_target = self.tokenizer(
            target_sequence,
            add_special_tokens=True,
            truncation=True,
            max_length=self.n_ctx,
            return_tensors='pt'
        )

        input_ids = tokenized_input['input_ids'].squeeze(0)
        mask = tokenized_input['attention_mask'].squeeze(0)
        target_ids = tokenized_target['input_ids'].squeeze(0)

        # Ensure input_ids and target_ids are the same length
        # Pad target_ids if necessary
        if target_ids.size(0) < input_ids.size(0):
            padding_length = input_ids.size(0) - target_ids.size(0)
            target_ids = F.pad(target_ids, (0, padding_length), value=-100)
        else:
            target_ids = target_ids[:input_ids.size(0)]

        return {
            'input_ids': input_ids,
            'mask': mask,
            'target_ids': target_ids
        }

# --------- Collate Function --------- #
def collate_fn(batch):
    """
    Custom collate function to pad sequences to the maximum length in the batch.
    """
    input_ids = [item['input_ids'] for item in batch]
    masks = [item['mask'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]

    # Pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    masks_padded = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    target_ids_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids_padded,
        'mask': masks_padded,
        'target_ids': target_ids_padded
    }

# --------- DataLoader Function --------- #
def load_dataloader(
    batch_size: int,
    tokenizer_name: str,
    n_ctx: int,
    num_train_examples: int,
    num_val_examples: int,
    num_test_examples: int,
    sequence_length: int,
    d_input: int,
    **kwargs
):
    """
    Load DataLoaders for in-context learning datasets.

    Args:
        batch_size (int): Batch size.
        tokenizer_name (str): Name of the tokenizer to use.
        n_ctx (int): Context length for the model.
        num_train_examples (int): Number of training examples.
        num_val_examples (int): Number of validation examples.
        num_test_examples (int): Number of test examples.
        sequence_length (int): Number of (xi, yi) pairs per sequence.
        d_input (int): Dimension of input xi.
    """
    # Create datasets
    print("Loading training dataset...")
    train_dataset = InContextLearningDataset(
        num_examples=num_train_examples,
        sequence_length=sequence_length,
        d_input=d_input,
        tokenizer_name=tokenizer_name,
        n_ctx=n_ctx,
        split='train'
    )

    print("Loading validation dataset...")
    val_dataset = InContextLearningDataset(
        num_examples=num_val_examples,
        sequence_length=sequence_length,
        d_input=d_input,
        tokenizer_name=tokenizer_name,
        n_ctx=n_ctx,
        split='val'
    )

    print("Loading test dataset...")
    test_dataset = InContextLearningDataset(
        num_examples=num_test_examples,
        sequence_length=sequence_length,
        d_input=d_input,
        tokenizer_name=tokenizer_name,
        n_ctx=n_ctx,
        split='test'
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    batch_size = 8
    tokenizer_name = 'bert-base-uncased'  # Replace with your tokenizer
    n_ctx = 20
    num_train_examples = 1000
    num_val_examples = 200
    num_test_examples = 200
    sequence_length = 20  # Number of (xi, yi) pairs per sequence
    d_input = 10  # Dimension of xi

    train_loader, val_loader, test_loader = load_dataloader(
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        n_ctx=n_ctx,
        num_train_examples=num_train_examples,
        num_val_examples=num_val_examples,
        num_test_examples=num_test_examples,
        sequence_length=sequence_length,
        d_input=d_input,
    )
    
    for batch in train_loader:
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention Mask shape:", batch['mask'].shape)
        print("Target IDs shape:", batch['target_ids'].shape)
        break
