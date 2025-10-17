"""
FIXED Modular Arithmetic Dataset

Key fixes:
1. Simpler format: input = full sequence, target = just the answer
2. No complex label shifting
3. Clear separation of input and target
"""

import torch
from torch.utils.data import Dataset
from typing import Optional


class ModularArithmeticDataset(Dataset):
    """
    Simple modular arithmetic dataset.
    
    Task: Given [SEP] x1 ... xN [SEP] y1 ... yN [SEP],
    predict (sum(x) * sum(y)) % modulo
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        sequence_length: int = 10,
        modulo: int = 97,
        value_range: int = 50,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.modulo = modulo
        self.value_range = value_range
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.SEP_TOKEN = 1
        self.VALUE_OFFSET = 2  # Values start at 2
        
        self.vocab_size = self.VALUE_OFFSET + max(value_range, modulo)
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Pre-generate all samples
        self.samples = []
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        
        for _ in range(num_samples):
            x_seq = torch.randint(0, value_range, (sequence_length,), generator=generator)
            y_seq = torch.randint(0, value_range, (sequence_length,), generator=generator)
            
            # Compute target
            sum_x = x_seq.sum().item()
            sum_y = y_seq.sum().item()
            target = (sum_x * sum_y) % modulo
            
            # Build input sequence: [SEP] x1 ... xN [SEP] y1 ... yN [SEP]
            input_tokens = [self.SEP_TOKEN]
            input_tokens.extend([x.item() + self.VALUE_OFFSET for x in x_seq])
            input_tokens.append(self.SEP_TOKEN)
            input_tokens.extend([y.item() + self.VALUE_OFFSET for y in y_seq])
            input_tokens.append(self.SEP_TOKEN)
            
            self.samples.append({
                'input_ids': torch.LongTensor(input_tokens),
                'target': target,  # Just the number, not tokenized yet
                'sum_x': sum_x,
                'sum_y': sum_y,
            })
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
    
    def get_vocab_size(self) -> int:
        return self.vocab_size


class SimpleSumDataset(Dataset):
    """
    Simple sum dataset - just sum a single sequence.

    Task: Given [SEP] x1 ... xN [SEP],
    predict sum(x) % modulo
    """

    def __init__(
        self,
        num_samples: int = 10000,
        sequence_length: int = 10,
        modulo: int = 97,
        value_range: int = 50,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.modulo = modulo
        self.value_range = value_range

        # Special tokens
        self.PAD_TOKEN = 0
        self.SEP_TOKEN = 1
        self.VALUE_OFFSET = 2  # Values start at 2

        self.vocab_size = self.VALUE_OFFSET + max(value_range, modulo)

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Pre-generate all samples
        self.samples = []
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        for _ in range(num_samples):
            x_seq = torch.randint(0, value_range, (sequence_length,), generator=generator)

            # Compute target - just sum of the sequence
            target = x_seq.sum().item() % modulo

            # Build input sequence: [SEP] x1 ... xN [SEP]
            input_tokens = [self.SEP_TOKEN]
            input_tokens.extend([x.item() + self.VALUE_OFFSET for x in x_seq])
            input_tokens.append(self.SEP_TOKEN)

            self.samples.append({
                'input_ids': torch.LongTensor(input_tokens),
                'target': target,  # Just the number, not tokenized yet
                'sequence_sum': x_seq.sum().item(),
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def get_vocab_size(self) -> int:
        return self.vocab_size


def collate_fn(batch):
    """Collate function with proper padding."""
    # Find max length
    max_len = max([item['input_ids'].size(0) for item in batch])

    # Pad sequences
    input_ids = []
    targets = []

    for item in batch:
        input_len = item['input_ids'].size(0)
        padding_len = max_len - input_len

        # Pad input_ids
        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(padding_len, dtype=torch.long)  # PAD_TOKEN = 0
        ])

        input_ids.append(padded_input)
        targets.append(item['target'])

    return {
        'input_ids': torch.stack(input_ids),
        'targets': torch.LongTensor(targets),
    }


if __name__ == "__main__":
    # Test the datasets
    print("Testing Modular Arithmetic Dataset (Product of Sums)")
    print("=" * 80)

    dataset = ModularArithmeticDataset(
        num_samples=5,
        sequence_length=3,
        modulo=97,
        value_range=10,
        seed=42,
    )

    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"PAD: {dataset.PAD_TOKEN}, SEP: {dataset.SEP_TOKEN}, VALUE_OFFSET: {dataset.VALUE_OFFSET}")
    print()

    for i in range(3):
        sample = dataset[i]
        input_ids = sample['input_ids'].tolist()
        target = sample['target']
        sum_x = sample['sum_x']
        sum_y = sample['sum_y']

        print(f"Sample {i + 1}:")
        print(f"  Input IDs: {input_ids}")
        print(f"  sum_x: {sum_x}, sum_y: {sum_y}")
        print(f"  Target: ({sum_x} * {sum_y}) % 97 = {target}")
        print(f"  Target as token: {target + dataset.VALUE_OFFSET}")
        print()

    print()
    print("Testing Simple Sum Dataset")
    print("=" * 50)

    simple_dataset = SimpleSumDataset(
        num_samples=5,
        sequence_length=3,
        modulo=97,
        value_range=10,
        seed=42,
    )

    print(f"Vocabulary size: {simple_dataset.vocab_size}")
    print()

    for i in range(3):
        sample = simple_dataset[i]
        input_ids = sample['input_ids'].tolist()
        target = sample['target']
        seq_sum = sample['sequence_sum']

        print(f"Sample {i + 1}:")
        print(f"  Input IDs: {input_ids}")
        print(f"  sequence_sum: {seq_sum}")
        print(f"  Target: {seq_sum} % 97 = {target}")
        print(f"  Target as token: {target + simple_dataset.VALUE_OFFSET}")
        print()

    # Test collate_fn with both datasets
    from torch.utils.data import DataLoader

    print("Batch tests:")
    print("-" * 30)

    # Test ModularArithmeticDataset batching
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("ModularArithmeticDataset batch:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  targets shape: {batch['targets'].shape}")
    print(f"  targets: {batch['targets'].tolist()}")

    # Test SimpleSumDataset batching
    simple_loader = DataLoader(simple_dataset, batch_size=2, collate_fn=collate_fn)
    simple_batch = next(iter(simple_loader))
    print("SimpleSumDataset batch:")
    print(f"  input_ids shape: {simple_batch['input_ids'].shape}")
    print(f"  targets shape: {simple_batch['targets'].shape}")
    print(f"  targets: {simple_batch['targets'].tolist()}")

    print()
    print("=" * 80)
    print("✓ Both datasets look good!")
