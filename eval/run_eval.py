"""Score a retriever on eval/golden_set.jsonl.

Run from the rag_doc root:
    python eval/run_eval.py                          # BM25, k=5
    python eval/run_eval.py --retriever bm25 --k 5

Metrics (answerable questions only)
    Recall@k     average fraction of a question's answer groups covered by the top-k chunks
    Full@k       fraction of questions where EVERY answer group is covered
    Precision@k  fraction of the k retrieved chunks that are labelled relevant
                 (the "context precision" number for the README)
    MRR@k        mean of 1/rank of the first relevant chunk (0 if none is in the top k)

Unanswerable questions have no relevant chunks, so they are not scored here. We record
the top-1 retrieval score for them, to see whether a score threshold could power the
abstention gate later.

Results are saved to eval/results/<retriever>_k<k>.json for the README table.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # lets `python eval/run_eval.py` import the retrieval package

METRICS = ("recall", "full", "precision", "mrr")
TYPE_ORDER = ["lookup", "multi_chunk", "paraphrase", "keyword"]


# ---------------------------------------------------------------- metrics (pure)
def evaluate_question(entry, retrieved_ids, k):
    """Score one answerable question. `entry` is a golden-set row; `retrieved_ids`
    are chunk IDs, best first. Equivalent copies of a passage live in the same
    answer group, so returning several of them never raises recall above 1."""
    top = list(retrieved_ids[:k])
    relevant = set(entry["relevant_chunks"])
    covered = [bool(set(group) & set(top)) for group in entry["answer_groups"]]
    rr = next((1.0 / rank for rank, cid in enumerate(top, start=1) if cid in relevant), 0.0)
    return {
        "recall": sum(covered) / len(covered),
        "full": float(all(covered)),
        "precision": sum(cid in relevant for cid in top) / k,
        "mrr": rr,
        "covered": covered,
        "retrieved": top,
    }


def summarize(rows):
    out = {m: mean(r[m] for r in rows) for m in METRICS}
    out["n"] = len(rows)
    return out


# ----------------------------------------------------------------------- helpers
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def make_retriever(name, chunks):
    """Add new retrievers here (dense, hybrid, ...) so --retriever can pick them."""
    if name == "bm25":
        from retrieval.bm25 import BM25Retriever
        return BM25Retriever(chunks)
    if name in ("dense", "dense_rerank"):
        first_stage = _dense(chunks)
    elif name in ("hybrid", "hybrid_rerank"):
        first_stage = _hybrid(chunks)
    else:
        raise SystemExit(f"Unknown retriever {name!r}. "
                         "Available: bm25, dense, hybrid, dense_rerank, hybrid_rerank")
    if name.endswith("_rerank"):
        from retrieval.rerank import RerankRetriever
        return RerankRetriever(first_stage, chunks, candidates=20)
    return first_stage


def _dense(chunks):
    from retrieval.dense import DenseRetriever
    return DenseRetriever(chunks, cache_dir=str(ROOT / "data" / "processed"))


def _hybrid(chunks):
    from retrieval.bm25 import BM25Retriever
    from retrieval.hybrid import HybridRetriever
    return HybridRetriever([BM25Retriever(chunks), _dense(chunks)])


def print_table(by_type, overall, k):
    head = f"{'type':14}{'n':>3}  {'Recall@'+str(k):>9} {'Full@'+str(k):>8} {'Prec@'+str(k):>8} {'MRR@'+str(k):>8}"
    print(head)
    print("-" * len(head))
    for t in TYPE_ORDER:
        if t in by_type:
            s = by_type[t]
            print(f"{t:14}{s['n']:>3}  {s['recall']:>9.3f} {s['full']:>8.3f} {s['precision']:>8.3f} {s['mrr']:>8.3f}")
    print("-" * len(head))
    print(f"{'ALL answerable':14}{overall['n']:>3}  {overall['recall']:>9.3f} {overall['full']:>8.3f} "
          f"{overall['precision']:>8.3f} {overall['mrr']:>8.3f}")


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--chunks", default=None,
                    help="chunks file (default: the one recorded in golden_set.meta.json)")
    ap.add_argument("--out-dir", default="eval/results")
    args = ap.parse_args()

    golden_path = resolve(args.golden)
    golden = load_jsonl(golden_path)
    meta_path = golden_path.with_name(golden_path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    if args.chunks:
        chunks_path = resolve(args.chunks)
    elif meta:
        chunks_path = resolve(meta["chunks_file"])
    else:
        chunks_path = resolve("data/processed/chunks.frozen.jsonl")

    # Labels are chunk IDs, so they are only valid for the exact chunks file they were built from.
    if meta and sha256_of(chunks_path) != meta["chunks_sha256"]:
        sys.exit("The chunks file has changed since the golden set was built, so the labels are "
                 "stale.\nRe-run python eval/build_golden_set.py, or point --chunks at the original file.")

    from retrieval.base import load_chunks
    chunks = load_chunks(str(chunks_path))
    retriever = make_retriever(args.retriever, chunks)
    print(f"Retriever: {args.retriever}   k={args.k}   chunks: {len(chunks)}   questions: {len(golden)}\n")

    rows, unanswerable = [], []
    for entry in golden:
        results = retriever.retrieve(entry["question"], k=args.k)
        ids = [cid for cid, _ in results]
        top1 = float(results[0][1]) if results else 0.0
        if not entry["answerable"]:
            unanswerable.append({"id": entry["id"], "question": entry["question"],
                                 "top1_score": top1, "retrieved": ids})
            continue
        scored = evaluate_question(entry, ids, args.k)
        scored.update({"id": entry["id"], "type": entry["type"], "question": entry["question"],
                       "top1_score": top1, "scores": [float(s) for _, s in results]})
        rows.append(scored)

    by_type = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
    for t, rs in groups.items():
        by_type[t] = summarize(rs)
    overall = summarize(rows)
    print_table(by_type, overall, args.k)

    # Where did it fall short? (a question is a miss if any answer group was not covered)
    misses = [r for r in rows if r["full"] < 1.0]
    if misses:
        print(f"\nMISSES ({len(misses)} questions with an uncovered answer group)")
        by_id = {e["id"]: e for e in golden}
        for r in misses:
            entry = by_id[r["id"]]
            missing = [g[:3] for g, ok in zip(entry["answer_groups"], r["covered"]) if not ok]
            print(f"  {r['id']} [{r['type']}] recall={r['recall']:.2f}  {r['question'][:70]}")
            print(f"      missing group(s): {missing}")
            print(f"      retrieved: {r['retrieved']}")

    if unanswerable:
        ans_top1 = [r["top1_score"] for r in rows]
        un_top1 = [u["top1_score"] for u in unanswerable]
        print("\nTOP-1 SCORE (for the abstention gate later)")
        print(f"  answerable   : min {min(ans_top1):.2f}  mean {mean(ans_top1):.2f}  max {max(ans_top1):.2f}")
        print(f"  unanswerable : min {min(un_top1):.2f}  mean {mean(un_top1):.2f}  max {max(un_top1):.2f}")
        print("  If these ranges overlap a lot, a raw retrieval score alone cannot decide when to abstain.")

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.retriever}_k{args.k}.json"
    out_path.write_text(json.dumps({
        "retriever": args.retriever, "k": args.k,
        "chunks_sha256": sha256_of(chunks_path), "n_chunks": len(chunks),
        "overall": overall, "by_type": by_type,
        "questions": rows, "unanswerable": unanswerable,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
