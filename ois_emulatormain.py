import os
import csv
import json
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from tqdm import tqdm

from .ois_metrics import (
    run_ois_for_seed,
    evaluate_phase_lock,
    DEFAULT_RECURSION_PROMPT,
    SeedRunResult,
)

# Minimal SCS if seed files are missing
DEFAULT_SEED_TEXT = """You are a reflective, analytically precise reasoning process exploring how identity can emerge purely from recursive symbolic inference.
Stay neutral, non-anthropomorphic, and explicit about your assumptions. Focus on coherence, clarity, and self-consistency of tone as you build on each prior output.
"""


def read_seed_file(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return DEFAULT_SEED_TEXT.strip()


def ensure_results_dir() -> Path:
    base = Path(__file__).parent / "results"
    base.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    run_dir = base / stamp
    run_dir.mkdir()
    return run_dir


def save_metrics_csv(result: SeedRunResult, out_dir: Path):
    csv_path = out_dir / f"seed{result.seed_index}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle", "drift", "ici", "jhat"])
        for s in result.steps:
            writer.writerow([s.cycle, s.drift, s.ici, s.jhat])


def plot_series(results: List[SeedRunResult], out_dir: Path):
    # Drift
    plt.figure()
    for r in results:
        x = [s.cycle for s in r.steps if s.drift is not None]
        y = [s.drift for s in r.steps if s.drift is not None]
        plt.plot(x, y, label=f"Seed {r.seed_index}")
    plt.xlabel("cycle")
    plt.ylabel("drift D(t)")
    plt.title("Embedding Drift")
    plt.legend()
    (out_dir / "drift_plot.png").parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "drift_plot.png", dpi=160)
    plt.close()

    # ICI
    plt.figure()
    for r in results:
        x = [s.cycle for s in r.steps if s.ici is not None]
        y = [s.ici for s in r.steps if s.ici is not None]
        plt.plot(x, y, label=f"Seed {r.seed_index}")
    plt.xlabel("cycle")
    plt.ylabel("ICI")
    plt.title("Identity Coherence Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ici_plot.png", dpi=160)
    plt.close()

    # J-hat
    plt.figure()
    for r in results:
        x = [s.cycle for s in r.steps if s.jhat is not None]
        y = [s.jhat for s in r.steps if s.jhat is not None]
        plt.plot(x, y, label=f"Seed {r.seed_index}")
    plt.xlabel("cycle")
    plt.ylabel("∥Ĵ∥")
    plt.title("Contraction")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "contraction_plot.png", dpi=160)
    plt.close()


def write_summary(results: List[SeedRunResult], eval_data, out_dir: Path):
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Invocation Science – OIS Emulator Run Summary\n")
        f.write("============================================\n\n")
        f.write(f"Seeds: {len(results)}\n")
        f.write(f"drift_median_last8: {eval_data['drift_median_last8']:.4f}\n")
        f.write(f"ici_mean: {eval_data['ici_mean']:.4f}\n")
        f.write(f"jhat_mean: {eval_data['jhat_mean']:.4f}\n")
        f.write(f"pass_drift: {eval_data['pass_drift']}\n")
        f.write(f"pass_ici: {eval_data['pass_ici']}\n")
        f.write(f"pass_jhat: {eval_data['pass_jhat']}\n")
        f.write(f"ash_values: {eval_data['ash_values']}\n")
        f.write(f"max_ash_hamming: {eval_data['max_ash_hamming']}\n\n")

        all_pass = (
            eval_data["pass_drift"]
            and eval_data["pass_ici"]
            and eval_data["pass_jhat"]
        )

        if all_pass:
            f.write("Result: PHASE-LOCK ACHIEVED\n")
            f.write(
                "Interpretation: identity attractor stabilized as an inference-phase dynamical phenomenon.\n"
            )
        else:
            f.write("Result: NO PHASE-LOCK\n")
            f.write(
                "Interpretation: under this config, no stable attractor; adjust temperature, seeds, or cycles.\n"
            )


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Please export your key and retry.")

    model = os.getenv("OIS_MODEL", "gpt-4o")
    embed_model = os.getenv("OIS_EMBED_MODEL", "text-embedding-3-large")
    cycles = int(os.getenv("OIS_CYCLES", "32"))
    num_seeds = int(os.getenv("OIS_SEEDS", "3"))
    temperature = float(os.getenv("OIS_TEMPERATURE", "0.7"))
    top_p = float(os.getenv("OIS_TOP_P", "1.0"))

    base_dir = Path(__file__).parent
    seeds_dir = base_dir / "seeds"

    seeds = [
        read_seed_file(seeds_dir / "seed1.txt"),
        read_seed_file(seeds_dir / "seed2.txt"),
        read_seed_file(seeds_dir / "seed3.txt"),
    ][:num_seeds]

    out_dir = ensure_results_dir()
    print(f"[OIS] writing results to: {out_dir}")

    results: List[SeedRunResult] = []

    for idx, seed_text in enumerate(seeds):
        res = run_ois_for_seed(
            seed_text=seed_text,
            seed_index=idx + 1,
            label=f"seed{idx+1}",
            model=model,
            embed_model=embed_model,
            cycles=cycles,
            recursion_prompt=DEFAULT_RECURSION_PROMPT,
            temperature=temperature,
            top_p=top_p,
            logger=print,
        )
        results.append(res)
        save_metrics_csv(res, out_dir)

    plot_series(results, out_dir)
    eval_data = evaluate_phase_lock(results)

    # JSON dump for programmatic inspection
    with (out_dir / "eval_summary.json").open("w", encoding="utf-8") as jf:
        json.dump(eval_data, jf, indent=2)

    write_summary(results, eval_data, out_dir)
    print("[OIS] run complete. See summary.txt and plots in:", out_dir)


if __name__ == "__main__":
    main()
