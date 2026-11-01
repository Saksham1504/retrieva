"""Build the README ablation table from the saved eval results.

Run from the rag_doc root, after running run_eval.py for each config:
    python eval/make_readme_table.py
    python eval/make_readme_table.py --k 5

Reads eval/results/<config>_k<k>.json, prints a markdown table, and also saves it to
eval/results/ablation_table.md. Configs with no results file are skipped.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = [
    ("bm25", "BM25"),
    ("dense", "Dense (bge-small)"),
    ("hybrid", "Hybrid (BM25 + dense, RRF)"),
    ("dense_rerank", "Dense + cross-encoder reranker"),
    ("hybrid_rerank", "Hybrid + cross-encoder reranker"),
]


def fraction(stats, key="full"):
    """'24/25 (0.96)' style text from a summary dict with a mean in `key` and a count `n`."""
    n = stats["n"]
    return f"{round(stats[key] * n)}/{n} ({stats[key]:.2f})"


def build_table(results_dir, k):
    header = (f"| Config | Recall@{k} | Full@{k} | Precision@{k} | MRR@{k} "
              f"| Paraphrase Full@{k} | Multi-chunk Full@{k} |")
    lines = [header, "|" + "---|" * 7]
    for name, label in CONFIGS:
        path = Path(results_dir) / f"{name}_k{k}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        o, t = r["overall"], r["by_type"]
        lines.append(
            f"| {label} | {o['recall']:.3f} | {fraction(o)} | {o['precision']:.3f} | {o['mrr']:.3f} "
            f"| {fraction(t['paraphrase']) if 'paraphrase' in t else '-'} "
            f"| {fraction(t['multi_chunk']) if 'multi_chunk' in t else '-'} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--results-dir", default="eval/results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    table = build_table(results_dir, args.k)
    print(table)
    out = results_dir / "ablation_table.md"
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
