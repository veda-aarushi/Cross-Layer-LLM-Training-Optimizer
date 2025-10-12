import argparse, time, csv, os, torch
from importlib import import_module

# Import TinyBlock, xent, SEQ, V from the notebook-equivalent module.
# If you put your model in runner/common.py, this line will work:
common = import_module("runner.common")
TinyBlock, xent, SEQ, V = common.TinyBlock, common.xent, common.SEQ, common.V

def bench(mode="baseline", steps=200, warmup=50, bs=8, seq=SEQ, d_model=512, bf16=False):
    device = "cuda" if torch.cuda.is_available() and mode != "cpu" else "cpu"
    torch.manual_seed(0)
    model = TinyBlock(d_model=d_model).to(device)

    if mode == "compile":
        model = torch.compile(model, mode="reduce-overhead")

    if bf16 and device == "cuda":
        model = model.to(torch.bfloat16)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    evt_s, evt_e = torch.cuda.Event(True), torch.cuda.Event(True)

    def toy_batch(bs, seq, device="cpu", pinned=False):
        x = torch.randint(0, V, (bs, seq), device=device)
        y = torch.randint(0, V, (bs, seq), device=device)
        return (x.pin_memory(), y.pin_memory()) if pinned and device == "cpu" else (x, y)

    def get_batch():
        if mode == "cpu":
            x_cpu, y_cpu = toy_batch(bs, seq, device="cpu", pinned=False)
            return x_cpu, y_cpu
        elif mode == "pinned" or mode == "compile":
            x_cpu, y_cpu = toy_batch(bs, seq, device="cpu", pinned=True)
            return x_cpu.to(device, non_blocking=True), y_cpu.to(device, non_blocking=True)
        else:  # "baseline" or "gpu"
            x_cpu, y_cpu = toy_batch(bs, seq, device="cpu", pinned=False)
            return x_cpu.to(device), y_cpu.to(device)

    # warmup
    for _ in range(warmup):
        x, y = get_batch()
        opt.zero_grad(set_to_none=True)
        loss = xent(model(x), y)
        loss.backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()

    # timed
    ms_total = 0.0
    tokens = 0
    for _ in range(steps):
        x, y = get_batch()
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            evt_s.record()
        loss = xent(model(x), y)
        loss.backward(); opt.step()
        if device == "cuda":
            evt_e.record(); torch.cuda.synchronize()
            ms_total += evt_s.elapsed_time(evt_e)
        tokens   += x.shape[0] * x.shape[1]

    toks_s = tokens / (ms_total / 1e3) if device == "cuda" else tokens / (steps * 1e-3 * ms_total or (steps * 1.0))
    return toks_s, float(loss.item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu","baseline","gpu","pinned","compile"], default="baseline")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--csv", type=str, default="results/benchmarks.csv")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    toks, loss = bench(mode=args.mode, steps=args.steps, warmup=args.warmup,
                       bs=args.bs, seq=args.seq, d_model=args.d_model, bf16=args.bf16)
    print(f"{args.mode} | toks/s={toks:,.0f} | loss≈{loss:.3f} | bs={args.bs} seq={args.seq} d={args.d_model} bf16={args.bf16}")

    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["mode","tokens_per_s","loss","bs","seq","d_model","bf16"])
        w.writerow([args.mode, f"{toks:.0f}", f"{loss:.4f}", args.bs, args.seq, args.d_model, int(args.bf16)])

if __name__ == "__main__":
    main()
