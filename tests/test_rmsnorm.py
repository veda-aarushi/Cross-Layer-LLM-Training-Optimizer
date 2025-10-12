import torch
from kernels.rmsnorm_triton import rmsnorm_ref, rmsnorm_triton

def test_rmsnorm_max_delta():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping GPU test.")
        return
    x = torch.randn(2, 4, 512, device="cuda", dtype=torch.float32)
    w = torch.ones(512, device="cuda", dtype=torch.float32)
    y_ref = rmsnorm_ref(x, w, eps=1e-5)
    y_tr  = rmsnorm_triton(x, w, eps=1e-5)
    max_err = (y_ref - y_tr).abs().max().item()
    print("RMSNorm Δ:", max_err)
    assert max_err < 1e-1  # loose tolerance for demo
