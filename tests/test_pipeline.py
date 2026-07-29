import torch
import torch.nn as nn
from src.pipeline import catalog_attention_sites, ActivationCapture

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
