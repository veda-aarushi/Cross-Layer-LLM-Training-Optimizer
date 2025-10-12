# Results

## Throughput (tokens/sec)

| Mode | tokens/sec | Notes |
|------|-----------:|------|
| cpu       | ~300    | reference |
| gpu       | ~16,000 | unoptimized GPU |
| pinned    | ~25,000 | pinned+non_blocking H2D |
| compile   | ~22,600 | fused graph execution |

`assets/throughput_comparison.png` was generated from `results/benchmarks.csv` using `utils/plot_results.py`.

## Interpretation
Pinned IO improves overlap of H2D with compute (≈1.6× over unoptimized GPU). `torch.compile` reduces launch overhead, improving scheduling. On larger GPUs, `compile` often overtakes `pinned`.

## Reproduction
python -u bench/bench_tokens.py --mode baseline
python -u bench/bench_tokens.py --mode pinned
python -u bench/bench_tokens.py --mode compile
python -u utils/plot_results.py
