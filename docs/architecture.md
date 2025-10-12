# Architecture Overview

**Objective:**  
Maximize end-to-end training throughput (tokens per second) for transformer-style models by optimizing **data movement**, **kernel-launch overhead**, and **GPU compute efficiency**.

---

## System Flow

┌──────────────────────────┐
│ Host (CPU) │
│ • Generates batches │
│ • Allocates pinned mem │
└────────────┬─────────────┘
│ (non_blocking=True)
▼
┌──────────────────────────┐
│ Device (GPU) │
│ • Asynchronous H2D copy │
│ • Forward pass (TinyBlock)
│ • CrossEntropy loss │
│ • Backward + AdamW opt │
└──────────────────────────┘

yaml
Copy code

---

## Optimizations

1. **Pinned Host Memory**  
   Page-locked host buffers speed up host-to-device (H2D) transfers and allow asynchronous copying.

2. **Non-Blocking Transfers**  
   `tensor.to(device, non_blocking=True)` overlaps memory copy with GPU computation.

3. **Torch Compile (`torch.compile(mode="reduce-overhead")`)**  
   Reduces Python dispatch and kernel-launch latency by fusing graph execution.

4. **Custom RMSNorm Kernel (Triton)**  
   Demonstrates how fused GPU kernels can reduce redundant memory accesses.

---

## End-to-End Pipeline
1. Generate batch on CPU (optionally pinned).  
2. Asynchronously transfer to GPU.  
3. Execute forward pass (TinyBlock).  
4. Compute loss and backward pass.  
5. Update weights with AdamW.  
6. Measure step time and throughput (tokens/sec).  

---

**Outcome:**  
The optimized setup achieves up to **1.6× speedup** vs naïve GPU runs, proving measurable improvement from cross-layer systems tuning.
