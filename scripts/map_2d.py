import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_metric_files(input_dir, pattern):
    """Find the metric batch files """

    files = sorted(input_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files found in {input_dir} matching pattern {pattern}"
        )

    return files


def load_metric_file(path):

    data = np.load(path)

    if "cpg_counts" not in data:
        raise KeyError(f"{path} does not contain cpg_counts")

    if "gc_counts" not in data:
        raise KeyError(f"{path} does not contain gc_counts")
    
    cpg_counts = np.asarray(data["cpg_counts"], dtype=np.int64)
    gc_counts = np.asarray(data["gc_counts"], dtype=np.int64)

    if cpg_counts.ndim != 1:
        raise ValueError(f"{path}: cpg_counts must be 1D, got {cpg_counts.shape}")

    if gc_counts.ndim != 1:
        raise ValueError(f"{path}: gc_counts must be 1D, got {gc_counts.shape}")

    if cpg_counts.shape[0] != gc_counts.shape[0]:
        raise ValueError(
            f"{path}: cpg_counts and gc_counts have different lengths: "
            f"{cpg_counts.shape[0]} vs {gc_counts.shape[0]}")

    return cpg_counts, gc_counts


def infer_density_shape(metric_files):
    """pass over the files to infer density_map shape:
    (max_xpg_count + 1, max_gc_count + 1)
    
    """

    max_cpg = 0
    max_gc = 0
    total_sequences = 0

    for path in metric_files:
        cpg_counts, gc_counts = load_metric_file(path)
    
        if cpg_counts.size == 0:
            continue

        file_max_cpg = int(np.max(cpg_counts))
        file_max_gc = int(np.max(gc_counts))

        max_cpg = max(max_cpg, file_max_cpg)
        max_gc = max(max_gc, file_max_gc)
        total_sequences += int(cpg_counts.shape[0])

    return max_cpg, max_gc, total_sequences


def update_density_map(density_map, cpg_counts, gc_counts):
    
    """ 
    x-axis: GC count
    y-axis: CpG count

    """

    valid = (
        (cpg_counts >= 0)
        & (cpg_counts < density_map.shape[0])
        & (gc_counts >= 0)
        & (gc_counts < density_map.shape[1]))
    
    np.add.at(
        density_map,
        (cpg_counts[valid], gc_counts[valid]),
        1,)
    
    return int(np.sum(valid)), int(np.sum(~valid))


def build_density_map(metric_files, max_cpg=None, max_gc=None):
    """
    Build the 2D CpG x GC density map from metric batch files.

    density_map indexing:
        density_map[cpg_count, gc_count]

    Plot interpretation:
        x-axis = GC count
        y-axis = CpG count
    """

    inferred_max_cpg, inferred_max_gc, total_sequences = infer_density_shape(metric_files)

    if max_cpg is None:
        max_cpg = inferred_max_cpg

    if max_gc is None:
        max_gc = inferred_max_gc

    density_map = np.zeros(
        (max_cpg + 1, max_gc + 1),
        dtype=np.uint64,
    )

    file_summaries = []
    total_valid = 0
    total_invalid = 0

    for path in metric_files:
        cpg_counts, gc_counts = load_metric_file(path)

        valid_count, invalid_count = update_density_map(
            density_map,
            cpg_counts,
            gc_counts,
        )

        total_valid += valid_count
        total_invalid += invalid_count

        summary = {
            "file": str(path),
            "n_sequences": int(cpg_counts.shape[0]),
            "valid_sequences": valid_count,
            "invalid_sequences": invalid_count,
            "cpg_min": int(np.min(cpg_counts)) if cpg_counts.size else None,
            "cpg_mean": float(np.mean(cpg_counts)) if cpg_counts.size else None,
            "cpg_max": int(np.max(cpg_counts)) if cpg_counts.size else None,
            "gc_min": int(np.min(gc_counts)) if gc_counts.size else None,
            "gc_mean": float(np.mean(gc_counts)) if gc_counts.size else None,
            "gc_max": int(np.max(gc_counts)) if gc_counts.size else None,
        }

        file_summaries.append(summary)

        print(
            f"Loaded {path.name}",
            f"n={summary['n_sequences']}",
            f"CpG={summary['cpg_min']}/{summary['cpg_mean']:.3f}/{summary['cpg_max']}",
            f"GC={summary['gc_min']}/{summary['gc_mean']:.3f}/{summary['gc_max']}",
        )

    global_summary = {
        "n_files": len(metric_files),
        "total_sequences_in_files": total_sequences,
        "total_valid_sequences": total_valid,
        "total_invalid_sequences": total_invalid,
        "inferred_max_cpg": inferred_max_cpg,
        "inferred_max_gc": inferred_max_gc,
        "used_max_cpg": int(max_cpg),
        "used_max_gc": int(max_gc),
        "density_shape": list(density_map.shape),
        "nonzero_bins": int(np.count_nonzero(density_map)),
        "max_bin_density": int(np.max(density_map)) if density_map.size else 0,
    }

    return density_map, global_summary, file_summaries


def plot_density_map(density_map, output_png, title):
    """ 
    
    """

    plot_data = np.log1p(density_map)

    plt.figure(figsize=(11, 8))

    plt.imshow(
        plot_data,
        origin="lower",
        aspect="auto",
        interpolation="nearest")
    
    plt.xlabel("GC count")
    plt.ylabel("CpG count")
    plt.title(title)
    plt.colorbar(label="log(1 + sequence count)")

    plt.tight_layout()
    plt.savefig(output_png, dpi=250)
    plt.close()

def save_nonzero_bins_csv(density_map, output_csv):
    """
    Save only occupied bins as CSV.

    Columns:
        cpg_count,gc_count,density
    """
    cpg_indices, gc_indices = np.nonzero(density_map)
    densities = density_map[cpg_indices, gc_indices]

    stacked = np.column_stack(
        [cpg_indices, gc_indices, densities]
    )

    header = "cpg_count,gc_count,density"

    np.savetxt(
        output_csv,
        stacked,
        fmt="%d",
        delimiter=",",
        header=header,
        comments="",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build and plot a 2D GC-count vs CpG-count density map from metric batch files."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing metric .npz files with cpg_counts and gc_counts.",
    )

    parser.add_argument(
        "--pattern",
        default="metrics_*.npz",
        help="Glob pattern for metric files inside input-dir.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/map_2d",
        help="Directory to save density map outputs.",
    )

    parser.add_argument(
        "--max-cpg",
        type=int,
        default=None,
        help="Optional max CpG count for density map y-axis. If omitted, inferred from data.",
    )

    parser.add_argument(
        "--max-gc",
        type=int,
        default=None,
        help="Optional max GC count for density map x-axis. If omitted, inferred from data.",
    )

    parser.add_argument(
        "--title",
        default="KRAS synonymous landscape: GC count vs CpG count",
        help="Plot title.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = find_metric_files(input_dir, args.pattern)

    print(f"Found {len(metric_files)} metric files in {input_dir}")

    density_map, global_summary, file_summaries = build_density_map(
        metric_files,
        max_cpg=args.max_cpg,
        max_gc=args.max_gc,
    )

    density_npz = output_dir / "density_map_2d_gc_x_cpg_y.npz"
    density_png = output_dir / "density_map_2d_gc_x_cpg_y.png"
    nonzero_csv = output_dir / "density_map_2d_nonzero_bins.csv"
    summary_json = output_dir / "density_map_2d_summary.json"
    file_summary_jsonl = output_dir / "density_map_2d_file_summaries.jsonl"

    np.savez_compressed(
        density_npz,
        density_map=density_map,
    )

    plot_density_map(
        density_map,
        density_png,
        args.title,
    )

    save_nonzero_bins_csv(
        density_map,
        nonzero_csv,
    )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    with open(file_summary_jsonl, "w", encoding="utf-8") as f:
        for item in file_summaries:
            f.write(json.dumps(item) + "\n")

    print("\nSaved outputs:")
    print(f"  Density map NPZ: {density_npz}")
    print(f"  Plot PNG:        {density_png}")
    print(f"  Nonzero CSV:     {nonzero_csv}")
    print(f"  Summary JSON:    {summary_json}")
    print(f"  File summaries:  {file_summary_jsonl}")

    print("\nGlobal summary:")
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()