import torch, triton, triton.language as tl

def rmsnorm_ref(x, w, eps=1e-5):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * w

@triton.jit
def _rmsnorm_fwd(x_ptr, w_ptr, y_ptr, D, eps, stride_x, stride_y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_x = x_ptr + pid * stride_x
    row_y = y_ptr + pid * stride_y

    acc = tl.zeros((), dtype=tl.float32)
    for offs in range(0, D, BLOCK_SIZE):
        idx  = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x    = tl.load(row_x + idx, mask=mask, other=0.).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    inv = tl.rsqrt(acc / D + eps)

    for offs in range(0, D, BLOCK_SIZE):
        idx  = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x    = tl.load(row_x + idx, mask=mask, other=0.).to(tl.float32)
        w    = tl.load(w_ptr + idx, mask=mask, other=1.).to(tl.float32)
        tl.store(row_y + idx, (x * inv) * w, mask=mask)

def rmsnorm_triton(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda and w.is_cuda
    B,S,D = x.shape
    y = torch.empty_like(x)
    x2d, y2d = x.view(-1, D), y.view(-1, D)
    grid = (x2d.shape[0],)
    _rmsnorm_fwd[grid](x2d, w, y2d, D, eps, x2d.stride(0), y2d.stride(0),
                       num_warps=4, num_stages=2)
    return y
