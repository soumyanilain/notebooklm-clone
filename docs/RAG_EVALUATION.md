# RAG evaluation: comparing retrieval strategies

**Bottom line:** on this corpus all four strategies found the right chunk, and answer quality was the same within the judge's noise. Reranking with a cross-encoder made retrieval about 25× slower on CPU (+2.6 s per question) without improving accuracy. The app therefore uses **hybrid search (BM25 + vector, fused with RRF)** by default and keeps reranking as an option.

Raw output, with every retrieved chunk and every answer: [`eval/results/report.md`](../eval/results/report.md) and [`eval/results/results.json`](../eval/results/results.json).

## What was tested

All four strategies use the same chunks (1000 characters, 150 overlap, 30 chunks), the same embedding model (`BAAI/bge-small-en-v1.5`, ONNX), and the same LLM (`openai/gpt-oss-120b` on Groq), with top-k = 5.

| Method | Pipeline |
|---|---|
| **Vector** (baseline) | Embed the question → cosine top-5 from Chroma |
| **Hybrid** | Dense top-20 + BM25 top-20 → Reciprocal Rank Fusion (k = 60) → top-5 |
| **Rerank** | Dense top-20 → cross-encoder `ms-marco-MiniLM-L-6-v2` scores every (question, chunk) pair → top-5 |
| **Hybrid + rerank** | Hybrid top-20 → cross-encoder → top-5 |

## How it was tested

- **Corpus** (`eval/corpus/`): a 9-page PDF handbook, a 10-slide PPTX quarterly review (some facts appear only in the speaker notes) and two incident postmortems as TXT files. They describe a fictional company on purpose, so the LLM can't answer from what it learned in training: a correct answer means retrieval worked. The corpus also has deliberate look-alikes: several form numbers (OPS-12/19/23), several hour counts, three certification levels, and two postmortems with the same structure.
- **Questions** (`eval/questions.json`): 12 questions of different types: exact-term lookups, paraphrases that share few words with the source, a fact that exists only in speaker notes, a question with a distractor document, arithmetic over two numbers, a two-fact question, and one question the sources can't answer.
- **Retrieval metrics** (all 4 methods, no LLM involved): a retrieved chunk is a *hit* when it contains the evidence strings for the question. Hit@1, Hit@3 and MRR are reported. Latency is the median of 3 runs per question.
- **Answer metrics** (vector vs. hybrid + rerank, the cheapest and the most expensive pipelines): the full chat pipeline answers each question. A different model (`openai/gpt-oss-20b`) then judges each answer: *correctness* against a hand-written reference answer and *groundedness* against the retrieved chunks, each from 1 to 5. Response time is measured end to end.
- **Hardware:** a Windows laptop CPU for the local models, Groq's API for the LLM, run on 8 Oct 2026.

**One labeling correction.** The fact behind Q5 ("why was the outage noticed late?") appears twice in the postmortem: once in the timeline ("the alert only fires after the error rate has stayed high for 15 minutes") and once in *What went wrong* ("the alert threshold of 15 minutes was too slow"). The first run's label accepted only the timeline sentence, so every method was scored as a miss, even though all of them retrieved the second passage and answered correctly. The label now accepts either passage. The saved run was rescored with `python -m eval.run_eval --rescore`; no model was called again and no answer or judge score changed.

## Results

### Summary

| Method | Hit@1 | Hit@3 | MRR | Retrieval time, mean (p95) | Answer correctness (1–5) | Groundedness (1–5) | Response time (end to end) |
|---|---|---|---|---|---|---|---|
| Vector | 0.91 | 1.00 | 0.955 | **115 ms** (177 ms) | 4.50 | 4.91 | **0.75 s** |
| Hybrid | 0.91 | 1.00 | 0.955 | **104 ms** (169 ms) | – | – | ≈ 0.75 s * |
| Rerank | 0.91 | 1.00 | 0.955 | 2,562 ms (3,273 ms) | – | – | ≈ 3.2 s * |
| Hybrid + rerank | 0.91 | 1.00 | 0.955 | 2,630 ms (3,222 ms) | 4.67 | 4.82 | 3.30 s |

Hit and MRR are over the 11 answerable questions. Groundedness is averaged over those same 11, because a correct "the sources don't cover this" answer has nothing to cite. \* Not run through the LLM; estimated as retrieval time + the measured LLM time (~0.6 s), which doesn't depend on the method.

**Where the time goes** (medians per question): query embedding ~100 ms, Chroma search ~12 ms, BM25 ~0.5 ms, cross-encoder over 20 candidates **~2,600 ms**, Groq LLM call 0.55–0.67 s. With reranking on, retrieval takes about four times as long as generation.

### Per question

Ranks are listed as Vector / Hybrid / Rerank / Hybrid + rerank. Answer columns are Vector / Hybrid + rerank.

| # | Question (type) | Evidence rank V / H / R / HR | Response time | Correctness | Groundedness |
|---|---|---|---|---|---|
| Q1 | Which form reports a field incident, and how soon? (exact term) | 1 / 1 / 1 / 1 | 1.74 s / 3.95 s | 3 / 5 | 5 / 5 |
| Q2 | How long can the drone stay in the air on one battery? (paraphrase) | 1 / 1 / 1 / 1 | 0.74 s / 4.96 s | 5 / 5 | 5 / 5 |
| Q3 | Supervised practice needed for the second certification tier? (paraphrase) | 1 / 1 / 1 / 1 | 0.77 s / 3.40 s | 3 / 3 | 5 / 4 |
| Q4 | Root cause of INC-2047? (exact term) | 1 / 1 / 1 / 1 | 0.68 s / 3.60 s | 5 / 5 | 5 / 5 |
| Q5 | Why was the outage noticed late? (paraphrase + distractor) | **2 / 2 / 1 / 1** | 0.78 s / 3.55 s | 5 / 5 | 5 / 5 |
| Q6 | Lead time for the LX-200 LiDAR module? (exact term) | 1 / 1 / 1 / 1 | 0.55 s / 2.47 s | 5 / 5 | 5 / 5 |
| Q7 | How much of the Kestrel budget was unspent? (arithmetic) | **1 / 1 / 2 / 2** | 0.76 s / 3.77 s | 5 / 5 | 5 / 5 |
| Q8 | Who is the November pilot customer? (speaker notes only) | 1 / 1 / 1 / 1 | 0.44 s / 3.38 s | 3 / 3 | 4 / 4 |
| Q9 | Robots hit by the bad firmware, and the rollback version? (distractor doc) | 1 / 1 / 1 / 1 | 0.64 s / 3.33 s | 5 / 5 | 5 / 5 |
| Q10 | What must happen to batteries before shipping? (paraphrase) | 1 / 1 / 1 / 1 | 0.65 s / 2.42 s | 5 / 5 | 5 / 5 |
| Q11 | Who owns the failover action item, and when is it due? (two facts) | 1 / 1 / 1 / 1 | 0.66 s / 2.70 s | 5 / 5 | 5 / 5 |
| Q12 | Halcyon's annual revenue? (unanswerable) | n/a | 0.63 s / 2.06 s | 5 / 5 | n/a |

### Retrieved chunks where the methods disagreed

Top 3 for each method, with scores (cosine for vector, RRF for hybrid, cross-encoder logit for rerank). ✅ marks the chunk with the evidence.

**Q5: "Why did it take so long for anyone to notice the telemetry outage?"**

| Rank | Vector | Hybrid + rerank |
|---|---|---|
| 1 | project_kestrel_q3_review.pptx · slide 7 (0.714) | postmortem_INC-2047.txt · lines 40-65 (1.85) ✅ |
| 2 | postmortem_INC-2047.txt · lines 40-65 (0.696) ✅ | handbook.pdf · p. 6 (0.95) |
| 3 | postmortem_INC-2047.txt · lines 1-38 (0.676) | project_kestrel_q3_review.pptx · slide 7 (0.62) |

The vector search ranked the slide titled *Incident follow-up: INC-2047* first because it is on-topic, but that slide doesn't explain the delay. The cross-encoder read the question and the chunk together and promoted the passage that actually answers "why".

**Q7: "How much of the Kestrel program budget was still unspent after Q3?"**

| Rank | Vector | Hybrid + rerank |
|---|---|---|
| 1 | project_kestrel_q3_review.pptx · slide 5, *Budget* (0.697) ✅ | slide 1, *Project Kestrel: Q3 2026 Review* (−2.02) |
| 2 | slide 2, *Q3 goals* (0.683) | slide 5, *Budget* (−2.35) ✅ |
| 3 | slide 1 (0.681) | slide 2 (−3.09) |

This time the reranker demoted the right slide. All three slides scored low (negative logits), and the title slide won on word overlap with "Kestrel program". MS MARCO cross-encoders are trained on web search queries and aren't good at matching "unspent" to "$3.1M approved / $2.4M spent". The answer was still correct, because the right slide was in the top 5.

**Q9: the distractor-document question (INC-1983 vs. INC-2047)**

| Rank | Vector | Hybrid | Hybrid + rerank |
|---|---|---|---|
| 1 | INC-1983 · lines 1-39 ✅ | INC-1983 · lines 1-39 ✅ | INC-1983 · lines 1-39 ✅ |
| 2 | INC-2047 · lines 1-38 | INC-1983 · lines 1-39 | INC-1983 · lines 41-42 |
| 3 | INC-2047 · lines 1-38 | INC-2047 · lines 1-38 | handbook.pdf · p. 3 |

All methods got rank 1 right. The difference is in what fills the rest of the context: vector search filled ranks 2–3 with the other postmortem, which has the same structure. BM25 matched "firmware" and "rolled back" and kept the second INC-1983 chunk at rank 2. Cleaner context gives the LLM less to confuse.

## What we learned

1. **The baseline is already strong on a small, clean corpus.** With only 30 chunks and distinctive wording, bge-small put the evidence in the top 3 for every answerable question and at rank 1 for 10 of 11. There was little left for the more expensive methods to fix.
2. **Reranking fixed one question and hurt another.** It helped on the "why" question (Q5) and hurt on the numeric one (Q7), so Hit@1 and MRR came out identical. What it did reliably was separate relevant chunks from the rest: the median gap between the top two scores was 6.7 logits for the cross-encoder, against 0.05 for cosine and 0.001 for RRF. That makes its score usable as a confidence signal, for example to say "nothing relevant found". The other scores can't do that.
3. **Reranking is expensive on CPU.** Scoring 20 chunks of about 250 tokens takes ~2.6 s on a laptop CPU, which is 25× the rest of retrieval and four times the LLM call. On the 1-vCPU Cloud Run deployment it would likely be slower.
4. **Answer quality didn't differ.** Correctness was 4.50 vs. 4.67 and groundedness 4.91 vs. 4.82, and these gaps come from judge noise. On Q1 the two methods gave nearly word-for-word identical answers ("OPS-19 … within 24 hours [1]") and got scores of 3 and 5. Both methods correctly refused the unanswerable question (Q12) without making anything up.
5. **The remaining errors come from generation, not retrieval.** Every score of 3 (Q1, Q3, Q8) was for an answer that was correct but left out a secondary detail from the reference: the 72-hour near-miss rule, the practical exam requirement, the "220 pumping stations". In each case the chunk with that detail was already at rank 1. The fix belongs in the prompt (e.g. "include conditions and exceptions stated in the excerpt"), not in retrieval.
6. **Hybrid costs almost nothing.** BM25 adds about half a millisecond, and its ranks matched vector search on every question while keeping distractor chunks out of the context (Q9).

## Decision

**Default: hybrid (BM25 + vector, RRF).** It tied for the best retrieval accuracy at about 100 ms. The evaluation couldn't separate it from plain vector search on this corpus, so the choice comes down to a design judgment. Real notebooks are full of exact identifiers: form numbers, incident IDs, part numbers, names. Dense embeddings can blur these as a notebook grows, and BM25 matches them exactly for almost no cost.

**Reranking stays available** in the chat's retrieval dropdown for large or noisy notebooks, or when a GPU is available. It isn't the default because, on CPU, it adds about 2.5 s to every question with no measured accuracy gain here.

## Limitations of this evaluation

- The corpus is small: 30 chunks and 12 questions, so one question moves Hit@1 by 9 points. A larger, noisier corpus (hundreds of chunks with near-duplicate pages) is where hybrid search and reranking are expected to matter, and it's the natural next test.
- The LLM judge is noisy at the level of a single item (see Q1), so the answer scores are only meaningful as averages, and differences under about 0.3 shouldn't be read as real.
- Each configuration was run once, and latency was measured on a Windows laptop, not on the Cloud Run deployment.
- Only one chunking setup (1000/150) and one embedding model were tested. Chunk size is a likely next variable, because several slides and pages are shorter than one chunk.
