"""RAG evaluation: compare retrieval strategies on a fixed corpus and question set.

What it measures
  * Retrieval (all methods, no LLM needed): Hit@1/3/5, MRR, evidence recall and latency.
    A retrieved chunk is a "hit" when it contains every evidence string for the question.
  * Answers (the methods in --answer-methods): the full RAG pipeline answers each question;
    an LLM judge (a different Groq model) scores correctness vs. a reference answer and
    groundedness vs. the retrieved chunks, 1-5. Response time is measured end to end.

Usage (from the repo root, with GROQ_API_KEY set):
    python -m eval.run_eval                        # retrieval for all methods + answers for 2 methods
    python -m eval.run_eval --answer-methods all   # answers for all 4 methods (slower, more tokens)
    python -m eval.run_eval --retrieval-only       # no LLM calls at all

Outputs: eval/results/results.json and eval/results/report.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from notebooklm.config import RETRIEVAL_METHODS, get_settings  # noqa: E402
from notebooklm.llm import LLMError, parse_json  # noqa: E402
from notebooklm.llm.prompts import JUDGE_SYSTEM, format_context, judge_prompt  # noqa: E402
from notebooklm.services import build_services, warm_up  # noqa: E402

EVAL_DIR = ROOT / "eval"
CORPUS = EVAL_DIR / "corpus"
RESULTS = EVAL_DIR / "results"
WORK = EVAL_DIR / ".eval_data"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_groups(q: dict) -> list[list[str]]:
    """A question is answerable from a chunk that holds ALL strings of ANY evidence group.
    Most questions have one group; some facts are stated in two places in the corpus."""
    if not q.get("evidence"):
        return []
    return [q["evidence"], *q.get("alt_evidence", [])]


def hit_rank(items: list[tuple[int, str]], q: dict) -> int | None:
    """items: (rank, chunk text) best first. Returns the rank of the first chunk with the evidence."""
    groups = [[norm(e) for e in g] for g in evidence_groups(q)]
    if not groups:
        return None
    for rank, text in items:
        text = norm(text)
        if any(all(w in text for w in g) for g in groups):
            return rank
    return None


def evidence_recall(items: list[tuple[int, str]], q: dict) -> float | None:
    groups = evidence_groups(q)
    if not groups:
        return None
    joined = " ".join(norm(t) for _, t in items)
    return max(sum(norm(e) in joined for e in g) / len(g) for g in groups)


def chunk_label(rc) -> str:
    return f"{rc.chunk.source_name} · {rc.chunk.locator}"


def judge(services, q: dict, answer: str, chunks) -> dict:
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_prompt(q["question"], q["reference"], answer,
                                                         format_context(chunks, 900))}]
    for _ in range(2):
        try:
            resp = services.llm.complete(messages, max_tokens=400, temperature=0.0, json_mode=True,
                                         models=services.settings.judge_models)
            data = parse_json(resp.text)
            return {"correctness": int(data.get("correctness", 0)), "groundedness": int(data.get("groundedness", 0)),
                    "reason": str(data.get("reason", ""))[:300], "judge_model": resp.model}
        except (ValueError, TypeError, LLMError) as exc:
            last = exc
            time.sleep(5)
    return {"correctness": None, "groundedness": None, "reason": f"judge failed: {last}", "judge_model": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="chunks passed to the LLM (top-k)")
    parser.add_argument("--answer-methods", default="vector,hybrid_rerank",
                        help="comma list of methods to run the LLM for, or 'all'")
    parser.add_argument("--retrieval-only", action="store_true", help="skip LLM answers and judging")
    parser.add_argument("--questions", default=str(EVAL_DIR / "questions.json"))
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between LLM calls (rate limits)")
    parser.add_argument("--rescore", action="store_true",
                        help="recompute retrieval metrics for the saved results (after editing questions.json)")
    args = parser.parse_args()
    if args.rescore:
        rescore(Path(args.questions))
        return

    methods = list(RETRIEVAL_METHODS)
    answer_methods = [] if args.retrieval_only else (
        methods if args.answer_methods == "all" else [m.strip() for m in args.answer_methods.split(",") if m.strip()])
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    # Fresh, isolated data folder so the evaluation never touches real notebooks.
    shutil.rmtree(WORK, ignore_errors=True)
    settings = dataclasses.replace(get_settings(), data_dir=WORK.resolve(), vector_dir=None, hf_dataset_repo=None)
    services = build_services(settings)
    print(f"Embeddings: {services.embedder.name} | reranker: {services.reranker.name} | LLM: {settings.llm_backend} "
          f"{settings.chat_models[0]}")
    warm_up(services)

    nb = services.store.create_notebook("RAG evaluation")
    t0 = time.perf_counter()
    for path in sorted(CORPUS.iterdir()):
        if path.suffix.lower() in {".pdf", ".pptx", ".txt"}:
            r = services.ingestion.add_file(nb.id, path, path.name)
            print(f"  ingested {path.name}: {r.source.n_chunks} chunks in {r.seconds:.2f}s")
    n_chunks = len(services.store.load_chunks(nb.id))
    print(f"Corpus: {n_chunks} chunks, ingested in {time.perf_counter() - t0:.1f}s\n")

    records = []
    for q in questions:
        for method in methods:
            # median of 3 runs for stable latency numbers
            runs = [services.retriever.retrieve(nb.id, q["question"], method, args.k) for _ in range(3)]
            result = runs[-1]
            latency = statistics.median(r.timings_ms["total"] for r in runs)
            items = [(rc.rank, rc.chunk.text) for rc in result.chunks]
            rank = hit_rank(items, q)
            rec = {
                "id": q["id"], "type": q["type"], "question": q["question"], "method": method,
                "retrieval_ms": round(latency, 1), "timings_ms": result.timings_ms,
                "hit_rank": rank, "evidence_recall": evidence_recall(items, q),
                "retrieved": [{"rank": rc.rank, "label": chunk_label(rc), "score": rc.score, "scores": rc.scores,
                               "chunk_id": rc.chunk.id, "snippet": norm(rc.chunk.text)[:220], "text": rc.chunk.text}
                              for rc in result.chunks],
            }
            if method in answer_methods:
                try:
                    turn = services.chat.ask(nb.id, q["question"], method=method, k=args.k, save=False)
                    rec.update({"answer": turn.answer, "model": turn.model, "llm_s": round(turn.llm_seconds, 2),
                                "response_s": round(turn.total_seconds, 2),
                                "cited": [c["n"] for c in turn.citations]})
                    time.sleep(args.pause)
                    rec["judge"] = judge(services, q, turn.answer, turn.retrieval.chunks)
                    time.sleep(args.pause)
                except LLMError as exc:
                    rec.update({"answer": None, "error": str(exc)})
                    print(f"  ! {q['id']} {method}: {exc}")
            score = rec.get("judge", {})
            print(f"{q['id']:>4} {method:<14} hit@{args.k}={'-' if rank is None else rank:<2} "
                  f"{rec['retrieval_ms']:7.1f} ms"
                  + (f" | answer {rec.get('response_s', 0):5.2f}s  correct={score.get('correctness')} "
                     f"grounded={score.get('groundedness')}" if "answer" in rec else ""))
            records.append(rec)

    summary = summarize(records, methods, questions, args.k)
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = {"date": datetime.now().isoformat(timespec="seconds"), "k": args.k, "chunks": n_chunks,
            "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
            "candidate_k": settings.candidate_k, "embedding_model": services.embedder.name,
            "reranker_model": services.reranker.name, "chat_model": settings.chat_models[0],
            "judge_model": settings.judge_models[0], "answer_methods": answer_methods}
    (RESULTS / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "records": records},
                                                     indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "report.md").write_text(render_report(meta, summary, records, questions), encoding="utf-8")
    services.vstore.close(nb.id)
    print(f"\nWrote {RESULTS / 'results.json'} and {RESULTS / 'report.md'}")


def rescore(questions_path: Path) -> None:
    """Re-apply the evidence labels to a saved run without calling any model again."""
    data = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in questions}
    texts: dict[str, str] = {}
    for path in WORK.glob("notebooks/*/chunks/*.jsonl"):  # older runs did not store chunk text
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = json.loads(line)
                texts[chunk["id"]] = chunk["text"]
    for rec in data["records"]:
        for item in rec["retrieved"]:
            item.setdefault("text", texts.get(item["chunk_id"], item["snippet"]))
        items = [(x["rank"], x["text"]) for x in rec["retrieved"]]
        rec["hit_rank"] = hit_rank(items, by_id[rec["id"]])
        rec["evidence_recall"] = evidence_recall(items, by_id[rec["id"]])
    data["meta"]["rescored"] = datetime.now().isoformat(timespec="seconds")
    data["summary"] = summarize(data["records"], list(RETRIEVAL_METHODS), questions, data["meta"]["k"])
    (RESULTS / "results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "report.md").write_text(render_report(data["meta"], data["summary"], data["records"], questions),
                                       encoding="utf-8")
    print(json.dumps(data["summary"], indent=1))


def summarize(records, methods, questions, k) -> dict:
    out = {}
    answerable = {q["id"] for q in questions if q["evidence"]}
    for m in methods:
        rows = [r for r in records if r["method"] == m]
        ranked = [r for r in rows if r["id"] in answerable]
        ranks = [r["hit_rank"] for r in ranked]
        lat = [r["retrieval_ms"] for r in rows]
        answered = [r for r in rows if r.get("judge") and r["judge"].get("correctness")]
        # A correct "not in the sources" answer has nothing to cite, so groundedness only
        # counts answerable questions.
        grounded = [r for r in answered if r["id"] in answerable]
        out[m] = {
            "hit@1": round(sum(1 for x in ranks if x == 1) / len(ranks), 3) if ranks else None,
            "hit@3": round(sum(1 for x in ranks if x and x <= 3) / len(ranks), 3) if ranks else None,
            f"hit@{k}": round(sum(1 for x in ranks if x) / len(ranks), 3) if ranks else None,
            "mrr": round(sum(1 / x for x in ranks if x) / len(ranks), 3) if ranks else None,
            "evidence_recall": round(statistics.mean(r["evidence_recall"] for r in ranked), 3) if ranked else None,
            "retrieval_ms_mean": round(statistics.mean(lat), 1),
            "retrieval_ms_p95": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 1),
            "answered": len(answered),
            "correctness_mean": round(statistics.mean(r["judge"]["correctness"] for r in answered), 2) if answered else None,
            "groundedness_mean": round(statistics.mean(r["judge"]["groundedness"] for r in grounded), 2) if grounded else None,
            "response_s_mean": round(statistics.mean(r["response_s"] for r in answered), 2) if answered else None,
            "llm_s_mean": round(statistics.mean(r["llm_s"] for r in answered), 2) if answered else None,
        }
    return out


def render_report(meta, summary, records, questions) -> str:
    k = meta["k"]
    L = ["# RAG evaluation results\n",
         f"_Run {meta['date']} · {meta['chunks']} chunks (size {meta['chunk_size']}, overlap {meta['chunk_overlap']}) · "
         f"top-k {k} · candidates {meta['candidate_k']} · embeddings `{meta['embedding_model']}` · reranker "
         f"`{meta['reranker_model']}` · answer model `{meta['chat_model']}` · judge `{meta['judge_model']}`_\n",
         "## Summary by method\n",
         f"| Method | Hit@1 | Hit@3 | Hit@{k} | MRR | Evidence recall | Retrieval ms (mean / p95) | "
         "Answer correctness (1-5) | Groundedness (1-5) | Response time (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, s in summary.items():
        def f(x):
            return "–" if x is None else x
        L.append(f"| {RETRIEVAL_METHODS[m]} | {f(s['hit@1'])} | {f(s['hit@3'])} | {f(s[f'hit@{k}'])} | {f(s['mrr'])} | "
                 f"{f(s['evidence_recall'])} | {s['retrieval_ms_mean']} / {s['retrieval_ms_p95']} | "
                 f"{f(s['correctness_mean'])} | {f(s['groundedness_mean'])} | {f(s['response_s_mean'])} |")
    L.append("\nHit@n = share of the answerable questions where a chunk containing the evidence was in the top n. "
             "MRR = mean of 1/rank of that chunk. Groundedness is averaged over answerable questions only (a correct "
             "refusal has nothing to cite). Response time = retrieval + LLM, end to end.\n")

    L.append("## Per question\n")
    L.append("| Question | Method | Top retrieved chunks | Evidence rank | Retrieval | Response time | Answer quality (correct / grounded) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in records:
        top = "<br>".join(f"{x['rank']}. {x['label']}" for x in r["retrieved"][:3])
        rank = "n/a" if not next(q for q in questions if q["id"] == r["id"])["evidence"] else (r["hit_rank"] or "miss")
        j = r.get("judge") or {}
        quality = f"{j.get('correctness')} / {j.get('groundedness')}" if j else "–"
        resp = f"{r['response_s']} s" if r.get("response_s") is not None else "–"
        L.append(f"| {r['id']}: {r['question']} | {r['method']} | {top} | {rank} | {r['retrieval_ms']} ms | {resp} | {quality} |")

    L.append("\n## Answers and retrieved chunks\n")
    for q in questions:
        L.append(f"### {q['id']} ({q['type']}): {q['question']}\n")
        L.append(f"**Reference:** {q['reference']}\n")
        for r in [r for r in records if r["id"] == q["id"]]:
            L.append(f"**{RETRIEVAL_METHODS[r['method']]}**: retrieval {r['retrieval_ms']} ms, evidence rank "
                     f"{r['hit_rank'] or ('n/a' if not q['evidence'] else 'miss')}")
            for x in r["retrieved"]:
                L.append(f"  {x['rank']}. `{x['label']}` score {x['score']} — {x['snippet'][:160]}…")
            if r.get("answer"):
                j = r.get("judge", {})
                answer = r["answer"].replace("\n", " ")
                L.append(f"\n  > **Answer ({r['response_s']} s):** {answer}\n  >\n  > _Judge: correctness "
                         f"{j.get('correctness')}, groundedness {j.get('groundedness')}: {j.get('reason')}_")
            L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
