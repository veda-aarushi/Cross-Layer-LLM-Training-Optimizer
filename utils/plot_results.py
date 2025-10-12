import pandas as pd, matplotlib.pyplot as plt, os

def main(csv_path="results/benchmarks.csv", out_png="assets/throughput_comparison.png"):
    assert os.path.exists(csv_path), f"Missing CSV: {csv_path}"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    df = pd.read_csv(csv_path)
    df["tokens_per_s"] = df["tokens_per_s"].astype(float)
    order = ["cpu","baseline","gpu","pinned","compile"]
    df["mode"] = pd.Categorical(df["mode"], categories=[m for m in order if m in set(df["mode"])], ordered=True)
    ax = df.groupby("mode")["tokens_per_s"].mean().plot.bar(rot=0, figsize=(6,3))
    ax.set_title("Throughput comparison")
    ax.set_ylabel("tokens/sec")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"Saved {out_png}")

if __name__ == "__main__":
    main()
