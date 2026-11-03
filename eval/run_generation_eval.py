"""Run the full pipeline on the golden set and score the answers.

    question -> retrieve + rerank -> abstention gate -> grounded answer -> judges

Needs an OpenAI API key in the OPENAI_API_KEY environment variable (PowerShell:
$env:OPENAI_API_KEY = "sk-..."). Type it in the terminal only; never put it in code or the repo.

Run from the rag_doc root:
    python eval/run_generation_eval.py --only q01,q09,q26     # a few questions first (cheap)
    python eval/run_generation_eval.py                        # all questions
    python eval/run_generation_eval.py --abstain-below 1.4 --model gpt-5-mini

Scores
    answered / refused       how the gate and the model behave on answerable vs unanswerable questions
    faithfulness             share of answer claims supported by the retrieved sources (judge LLM)
    correctness              answer vs the reference answer (judge LLM)
    valid citations          answers that cite at least one real source and no invented source numbers
Answers and judge reasoning are saved to eval/results/generation_<retriever>.json so you can read them.
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_eval                                                  # noqa: E402  (same folder)
from generation.judges import judge_correctness, judge_faithfulness   # noqa: E402
from generation.openai_llm import OpenAILLM                      # noqa: E402
from generation.pipeline import RagPipeline                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", default="hybrid_rerank",
                    help="use a rerank config: the abstention gate needs reranker scores")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--abstain-below", type=float, default=1.4,
                    help="refuse when the top-1 reranker score is below this (from the --sweep run)")
    ap.add_argument("--model", default=None, help="generation model (default: OPENAI_MODEL or gpt-5-mini)")
    ap.add_argument("--judge-model", default=None,
                    help="judge model (default: same as --model; a different or stronger one is fairer)")
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--only", default=None, help="comma-separated question ids, e.g. q01,q09,q26")
    ap.add_argument("--no-judge", action="store_true", help="skip the judge calls (cheaper)")
    ap.add_argument("--out-dir", default="eval/results")
    args = ap.parse_args()

    golden_path = run_eval.resolve(args.golden)
    golden = run_eval.load_jsonl(golden_path)
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        golden = [g for g in golden if g["id"] in wanted]

    meta_path = golden_path.with_name(golden_path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    chunks_path = run_eval.resolve(args.chunks or (meta["chunks_file"] if meta else
                                                   "data/processed/chunks.frozen.jsonl"))
    if meta and run_eval.sha256_of(chunks_path) != meta["chunks_sha256"]:
        sys.exit("The chunks file has changed since the golden set was built; labels are stale. "
                 "Re-run python eval/build_golden_set.py.")

    from retrieval.base import load_chunks
    chunks = load_chunks(str(chunks_path))
    by_id = {c["chunk_id"]: c for c in chunks}
    retriever = run_eval.make_retriever(args.retriever, chunks)

    cache = ROOT / "data" / "processed" / "llm_cache.json"
    gen_llm = OpenAILLM(model=args.model, cache_path=cache)
    judge_llm = OpenAILLM(model=args.judge_model or args.model, cache_path=cache)
    pipeline = RagPipeline(retriever, chunks, gen_llm, abstain_below=args.abstain_below, k=args.k)
    print(f"Retriever: {args.retriever}   gate: refuse below {args.abstain_below:g}   "
          f"generator: {gen_llm.model}   judge: {judge_llm.model}   questions: {len(golden)}\n")

    records = []
    print(f"{'id':4} {'type':12} {'outcome':14} {'top1':>7} {'faith':>6} {'correct':>8}")
    for entry in golden:
        rec = {"id": entry["id"], "type": entry["type"], "answerable": entry["answerable"]}
        try:
            rec.update(pipeline.answer(entry["question"]))
            if entry["answerable"] and not rec["abstained"] and not args.no_judge:
                used = [by_id[s["chunk_id"]] for s in rec["sources"]]
                try:
                    rec["faithfulness"] = judge_faithfulness(judge_llm, rec["answer"], used)
                    rec["correctness"] = judge_correctness(judge_llm, entry["question"], rec["answer"],
                                                           entry["reference_answer"])
                except Exception as e:                                  # one bad judge reply must not stop the run
                    rec["judge_error"] = f"{type(e).__name__}: {e}"
        except Exception as e:
            rec.update({"error": f"{type(e).__name__}: {e}", "abstained": None})
        records.append(rec)

        outcome = ("error" if "error" in rec else
                   f"refused({rec['abstained_by']})" if rec["abstained"] else "answered")
        faith = rec.get("faithfulness", {}).get("faithfulness")
        corr = rec.get("correctness", {}).get("score")
        print(f"{rec['id']:4} {rec['type']:12} {outcome:14} {rec.get('top1_score', float('nan')):>7.2f} "
              f"{'-' if faith is None else format(faith, '.2f'):>6} "
              f"{'-' if corr is None else format(corr, '.1f'):>8}")
        if "error" in rec:
            print(f"     error: {rec['error'][:400]}")
        if "judge_error" in rec:
            print(f"     judge error: {rec['judge_error'][:400]}")

    # ------------------------------------------------------------------ summary
    answerable = [r for r in records if r["answerable"] and "error" not in r]
    unanswerable = [r for r in records if not r["answerable"] and "error" not in r]
    answered = [r for r in answerable if not r["abstained"]]
    summary = {"retriever": args.retriever, "abstain_below": args.abstain_below,
               "generator": gen_llm.model, "judge": judge_llm.model}

    print("\nRESULTS")
    if answerable:
        by_gate = sum(r["abstained_by"] == "gate" for r in answerable if r["abstained"])
        by_model = sum(r["abstained_by"] == "model" for r in answerable if r["abstained"])
        wrong_refusals = [r["id"] for r in answerable if r["abstained"]]
        print(f"  answerable ({len(answerable)}): answered {len(answered)}, "
              f"refused {len(wrong_refusals)} (gate {by_gate}, model {by_model})"
              + (f"   refused: {wrong_refusals}" if wrong_refusals else ""))
        summary["answerable"] = {"n": len(answerable), "answered": len(answered),
                                 "wrongly_refused": wrong_refusals}
        faith = [r["faithfulness"]["faithfulness"] for r in answered
                 if r.get("faithfulness") and r["faithfulness"]["faithfulness"] is not None]
        corr = [r["correctness"]["score"] for r in answered if r.get("correctness")]
        if faith:
            full = sum(f == 1.0 for f in faith)
            print(f"  faithfulness: mean {mean(faith):.3f}   fully supported {full}/{len(faith)}")
            summary["faithfulness_mean"] = mean(faith)
        if corr:
            print(f"  correctness : mean {mean(corr):.3f}   "
                  f"correct {sum(c == 1.0 for c in corr)}  partial {sum(c == 0.5 for c in corr)}  "
                  f"incorrect {sum(c == 0.0 for c in corr)}")
            summary["correctness_mean"] = mean(corr)
        if answered:
            ok = sum(bool(r["cited"]) and not r["invalid_citations"] for r in answered)
            print(f"  valid citations: {ok}/{len(answered)} answers cite a real source and invent none")
            summary["valid_citation_answers"] = ok
        judge_errors = [r["id"] for r in answerable if "judge_error" in r]
        if judge_errors:
            print(f"  judge errors (excluded from scores): {judge_errors}")
    if unanswerable:
        refused = [r for r in unanswerable if r["abstained"]]
        by_gate = sum(r["abstained_by"] == "gate" for r in refused)
        missed = [r["id"] for r in unanswerable if not r["abstained"]]
        print(f"  unanswerable ({len(unanswerable)}): correctly refused {len(refused)} "
              f"(gate {by_gate}, model {len(refused) - by_gate})"
              + (f"   answered by mistake: {missed}" if missed else ""))
        summary["unanswerable"] = {"n": len(unanswerable), "refused": len(refused), "answered_by_mistake": missed}
    errors = [r["id"] for r in records if "error" in r]
    if errors:
        print(f"  errors (excluded): {errors}")

    shown = [r for r in answered][:2] + [r for r in unanswerable if not r["abstained"]]
    if shown:
        print("\nEXAMPLE ANSWERS (read these: do they look right?)")
        for r in shown:
            tag = "ANSWERED AN UNANSWERABLE QUESTION" if not r["answerable"] else "answerable"
            print(f"  {r['id']} ({tag}): {r['question']}")
            print(f"    {r['answer'][:400]}")

    out_dir = run_eval.resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generation_{args.retriever}.json"
    out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(f"\nAPI calls this run: generator {gen_llm.calls}, judge {judge_llm.calls} (cached calls are free)")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
