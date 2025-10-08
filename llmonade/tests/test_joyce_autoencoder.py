import torch
import torch.nn as nn

from fla.models.joyce.autoencoder import JoyceAutoEncoder, JoyceConfig


class DummyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x, attn_mask=None):
        return self.ln(x)


class DummyLM(nn.Module):
    def __init__(self, n_layers=16, d=1024, vocab=32000):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([DummyBlock(d) for _ in range(n_layers)])

    def forward(self, input_ids=None, attention_mask=None):
        x = self.tok_embeddings(input_ids)
        for blk in self.layers:
            x = blk(x, attention_mask)
        return x


def test_forward_shapes():
    base = DummyLM()
    cfg = JoyceConfig(d_model=1024, seq_len=128, num_compressed=16, layer_L=3)
    model = JoyceAutoEncoder(base, cfg)

    B, T = 2, 128
    ids = torch.randint(0, 32000, (B, T))
    am  = torch.ones(B, T)

    loss, aux = model(ids, am)
    assert loss.dim() == 0
    assert aux["compressed"].shape == (B, cfg.num_compressed, cfg.d_model)
    assert aux["recon"].shape == (B, T, cfg.d_model)
    assert aux["hL"].shape == (B, T, cfg.d_model)
