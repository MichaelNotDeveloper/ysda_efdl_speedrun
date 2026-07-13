from enum import Enum

import torch
from torch import nn
from transformer import PositionalEncoding
from dataset import BrainDataset, BigBrainDataset, collate_fn
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import gc


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4

class GPT2Ripoff(torch.nn.Module):
    def __init__(self, ntokens : int, hdim : int = 1024, nhead : int = 8, padding_idx : int = 0):
        super().__init__()
        self.embed = nn.Embedding(ntokens, hdim, padding_idx=padding_idx)
        self.pe = PositionalEncoding(hdim)
        self.decoder = nn.TransformerDecoderLayer(d_model=hdim, nhead=nhead, batch_first=True)
        
    def forward(self, tokens, mask=None):
        embeds = self.embed(tokens)
        embeds = self.pe(embeds)
        out = self.decoder(embeds, embeds, mask, mask)
        return out
        
def get_gpt2_model(tokenizer, hdim : int = 1024, nhead : int = 8) -> torch.nn.Module:
    return GPT2Ripoff(len(tokenizer), hdim, nhead, tokenizer.pad_token_id)


def run_epoch(data_mode: DataMode, batch_size: int = 64, warmup_iter: int = 5) -> None:
    device = torch.device("cuda")
    gc.collect()
    torch.cuda.empty_cache()
    if data_mode is DataMode.BRAIN:
        dataset = BrainDataset('./data/wikitext-103-raw-v1/')
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    if data_mode is DataMode.BIG_BRAIN:
        dataset = BigBrainDataset('./data/wikitext-103-raw-v1/')
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, collate_fn=collate_fn)
    tokenizer = dataset.tokenizer
    model = get_gpt2_model(tokenizer)
    
    delta_times = []
    for tokens, mask in tqdm(dataloader):
        chk1 = time.perf_counter()
        tokens = tokens.to(device)
        mask = mask.to(device)
        torch.cuda.synchronize(device)
        chk2 = time.perf_counter()
        delta_times.append(chk2 - chk1)
    # remove first N iterations as warmup
    delta_times = delta_times[warmup_iter:]
    return {
        "mean" : np.mean(delta_times),
        "max" : np.max(delta_times),
        "min" : np.min(delta_times),
        "median" : np.median(delta_times),
        "deltas" : delta_times,
    }
