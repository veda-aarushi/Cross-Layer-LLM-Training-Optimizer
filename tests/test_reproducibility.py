import torch, random, numpy as np

def test_seed_reproducibility():
    seed = 1234
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    a = torch.randn(5)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    b = torch.randn(5)
    assert torch.allclose(a, b)
