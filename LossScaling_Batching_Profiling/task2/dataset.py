from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from pathlib import Path
import torch.nn.functional as F


MAX_LENGTH = 640
TOKENIZER_NAME = "bert-base-uncased"

class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.padding_idx = self.tokenizer.pad_token_id
        self.tokens_list = []
        path_names = [f for f in Path(data_path).iterdir() if f.is_file() and ("train" in str(f))]
        for filename in path_names:
            print(f"Loading {filename}...")
            with open(filename, 'r') as file:
                lines = [f.strip() for f in file.readlines()]
            self.tokens_list.extend(lines)

    def __getitem__(self, idx: int):
        line = self.tokenizer(self.tokens_list[idx], max_length=self.max_length, truncation=True)
        tokens = line['input_ids']
        mask = line['attention_mask']
        padding_size = self.max_length - len(tokens)
        tokens.extend([self.padding_idx] * padding_size)
        mask.extend([0] * padding_size)
        return torch.tensor(tokens), torch.tensor(mask)
    
    def __len__(self):
        return len(self.tokens_list)


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.padding_idx = self.tokenizer.pad_token_id
        self.tokens_list = []
        path_names = [f for f in Path(data_path).iterdir() if f.is_file() and ("train" in str(f))]
        for filename in path_names:
            print(f"Loading {filename}...")
            with open(filename, 'r') as file:
                lines = [f.strip() for f in file.readlines()]
            self.tokens_list.extend(lines)

    def __getitem__(self, idx: int):
        line = self.tokenizer(self.tokens_list[idx], max_length=self.max_length, truncation=True)
        tokens = line['input_ids']
        mask = line['attention_mask']
        return torch.tensor(tokens), torch.tensor(mask)
    
    def __len__(self):
        return len(self.tokens_list)


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        self.max_length = max_length
        self.n_bins = n_bins
        self.len2id = {}
        self.id2len = {}
        self.tokens = []
        
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.padding_idx = self.tokenizer.pad_token_id
        
        path_names = [f for f in Path(data_path).iterdir() if f.is_file() and ("train" in str(f))]
        for filename in path_names:
            print(f"Loading {filename}...")
            with open(filename, 'r') as file:
                lines = [f.strip() for f in file.readlines()]
            lines = self.tokenizer(self.tokens_list[idx], max_length=self.max_length, truncation=True)
            tokens.update(lines)
        
        for idx, token in enumerate(self.tokens):
            if len(token) in self.len2id:
                self.len2id[len(token)].append(idx)
            else:
                self.len2id[len(token)] = [idx]
            
            if idx in self.id2len:
                self.id2len[idx].append(len(token))
            else:
                self.id2len[idx] = [len(token)]

    def __getitem__(self, idx: int):
        tokens = line['input_ids']
        mask = line['attention_mask']
        return torch.tensor(tokens), torch.tensor(mask)
    
    def __len__(self):
        return len(self.tokens_list)
    

class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __getitem__(self, idx: int):
        pass


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :return: tuple of padded sequences and corresponding training targets
    """
    batch = list(zip(*batch))
    max_len = max(map(lambda x : len(x), batch[0]))
    pad_tokens = lambda x : F.pad(x, (0, max_len - len(x)), 'constant', pad_id)
    pad_mask = lambda x : F.pad(x, (0, max_len - len(x)), 'constant', 0)
    batch[0] = torch.stack(tuple(map(pad_tokens, batch[0])), dim=0)
    batch[1] = torch.stack(tuple(map(pad_tokens, batch[1])), dim=0)
    return tuple(batch)
    

class UltraBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, dataset: UltraBigBrainDataset, close_range : int = 5, max_length: Optional[int] = MAX_LENGTH):
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset = dataset
        self.close_range = close_range

    def __len__(self):
        return len(self.dataset) // batch_size

    def __iter__(self):
        # weighted selection to uniformly choose ids
        base_len = self.dataset.id2len[torch.randint(0, len(self.dataset))]
        base_range = torch.randint(0, self.close_range)
        l_border = base_len - base_range + 1
        r_border = l + self.close_range
        
        weights = [
            len(self.dataset.len2id[l] for l in range(l_border, r_border))
        ]
        len_ids = torch.multinomial(weights, num_samples=2, replacement=True)
        
        batch_ids = [
            self.dataset.len2id[torch.]
        ]