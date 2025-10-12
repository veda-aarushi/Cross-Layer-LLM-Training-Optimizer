import math, torch, torch.nn as nn

V = 4096
SEQ = 1024

def toy_batch(bs=8, seq=SEQ, device="cpu", pinned=False):
    x = torch.randint(0, V, (bs, seq), device=device)
    y = torch.randint(0, V, (bs, seq), device=device)
    if pinned and device == "cpu":
        x = x.pin_memory(); y = y.pin_memory()
    return x, y

class TinyBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, vocab=V):
        super().__init__()
        self.tok  = nn.Embedding(vocab, d_model)
        self.ln   = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, 4*d_model)
        self.fc2  = nn.Linear(4*d_model, d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        h = self.tok(x)            # [B,S,D]
        h = self.ln(h)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        return self.head(h)        # [B,S,V]

def xent(logits, y):
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )
