import numpy as np
import torch
import torch.nn as nn
from src.pipeline import (catalog_attention_sites, ActivationCapture,
                          select_probe_sites, pool_activation)


def test_select_probe_sites():
    sites = [
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2",
        "down_blocks.1.attentions.0.transformer_blocks.1.attn1",  # block 1 -> dropped
        "mid_block.attentions.0.transformer_blocks.0.attn1",
    ]
    got = select_probe_sites(sites)
    assert got == [sites[0], sites[1], sites[3]]


def test_pool_activation_cond_half_mean():
    # (B=2, T=3, C=2): uncond all 0, cond all 5 -> pooled = [5, 5]
    act = np.stack([np.zeros((3, 2)), np.full((3, 2), 5.0)])
    out = pool_activation(act, cond_index=1)
    assert out.shape == (2,)
    assert np.allclose(out, [5.0, 5.0])

class Attn(nn.Module):
    def __init__(self): super().__init__(); self.lin = nn.Linear(4, 4)
    def forward(self, x): return self.lin(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = Attn()
        self.attn2 = Attn()
    def forward(self, x): return self.attn2(self.attn1(x))

def test_catalog_finds_attn_sites():
    m = Block()
    sites = catalog_attention_sites(m)
    assert set(sites) == {"attn1", "attn2"}

def test_activation_capture_records_and_cleans_up():
    m = Block()
    sites = catalog_attention_sites(m)
    with ActivationCapture(m, sites) as cap:
        m(torch.zeros(1, 4))
        assert set(cap.acts.keys()) == {"attn1", "attn2"}
        assert cap.acts["attn1"].shape == (1, 4)
    # handles removed: a second forward must not add new captures
    cap.acts.clear()
    m(torch.zeros(1, 4))
    assert cap.acts == {}
