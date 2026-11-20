# VaxiS evaluation

## Working data

- `eval_dataset_300_final_fixed.json` is the default for live evaluation. It contains 300 working goldens, including 210 answerable and 90 partially answerable records. Sixty records are labeled table reasoning.
- `eval_dataset_300_final.json` is the earlier version. Ordered source-chunk lists differ for 279 questions.
- `eval_dataset_300_generation_results.json` contains 300 saved responses matching the corrected goldens. Each response contains four retrieved IDs. Score these rankings at cutoffs up to four.
- `../test_dataset.json` is a legacy 460-record artifact retained for unanswerable-query testing. Its labels include 222 unanswerable, 164 partially answerable and 74 answerable records. The additional pass loads all supplied records. [PLACEHOLDER: identify the explicitly selected unanswerable subset used for the report.]

[PLACEHOLDER: identify the final adjudicated benchmark export, corpus revision and manifest for the saved responses.]

## Score the saved rankings

Run from the repository root. This path reads existing ranked IDs and gold labels.

```bash
uv run vacxis-eval --retrieval-only \
  --dataset evaluation/eval_dataset_300_final_fixed.json \
  --results evaluation/eval_dataset_300_generation_results.json \
  --ir-k-values 1 3 4 --skip-unanswerable \
  --out-json evaluation/saved_rankings_ir.json --out-md evaluation/saved_rankings_ir.md
```

## Run matched comparisons

Run on the machine with the corpus, Qdrant and local model weights. These commands use the same dataset, final retrieval depth and cutoffs.

```bash
uv run --extra retrieval vacxis-eval --retrieval-only --baseline --top-k 10 \
  --dataset evaluation/eval_dataset_300_final_fixed.json --chunks chunks_storage/chunks.parquet \
  --ir-k-values 1 3 5 10 --skip-unanswerable \
  --out-json evaluation/naive_k10.json --out-md evaluation/naive_k10.md

uv run --extra retrieval vacxis-eval --retrieval-only --advanced-rag --top-k 10 \
  --dataset evaluation/eval_dataset_300_final_fixed.json --chunks chunks_storage/chunks.parquet \
  --ir-k-values 1 3 5 10 --skip-unanswerable \
  --out-json evaluation/advanced_k10.json --out-md evaluation/advanced_k10.md

uv run --extra retrieval vacxis-eval --retrieval-only --top-k 10 \
  --dataset evaluation/eval_dataset_300_final_fixed.json --chunks chunks_storage/chunks.parquet \
  --ir-k-values 1 3 5 10 --skip-unanswerable \
  --out-json evaluation/hybrid_k10.json --out-md evaluation/hybrid_k10.md
```

The `--baseline` CLI switch selects BM25 for **Naive**. **Advanced** searches the dense Qdrant index using the original query. **Hybrid (baseline)** is the study reference. It uses dense and learned sparse BGE-M3 retrieval, reciprocal rank fusion and reranking unless `--skip-rerank` is set.

For end-to-end evaluation, remove `--retrieval-only`, supply the generation and judge environment via `uv run --extra retrieval --env-file .env vacxis-eval`, and use separate output paths. Record the judge model, generator, prompts, corpus and index revisions, query exclusions and scoring coverage. IR metrics treat labeled chunk IDs as relevant. A supporting alternative passage can receive no credit if absent from those labels.

4. **Table-aware** — Preliminary M3b retrieval results and summarize_dual indexing code are available. [PLACEHOLDER: final indexing manifest, full implementation specification and answer-quality results.]
5. **Graph RAG** — A preliminary LightRAG-augmented M4 retrieval report is available. [PLACEHOLDER: confirm final configuration mapping and provide complete implementation and answer-quality evaluation.]
6. **Agent RAG** — [PLACEHOLDER: implementation revision, tool configuration, exact invocation, metrics, tool-call counts and latency.]

The current baseline reports are `ir_naive_bm25.json`, `ir_advanced.json` and `ir_hybrid.json`. Their Hit@5 values are 0.5367, 0.5900 and 0.7433 on 300 questions at retrieval depth five. See [the results audit](../results/report_discrepancies.md) for extension scoring and provenance caveats. [PLACEHOLDER: attach answer quality, citation support, abstention and latency results.]

## Advanced retrieval settings


`--advanced-rag` uses the following single-query embedding pipeline:

`user query -> BGE-M3 dense embedding -> Qdrant search -> top-k chunks -> LLM`

The original question is embedded once and sent unchanged
to the standard generator with the top-k chunks from the existing Qdrant dense
index. Search order, similarity scores, document metadata, and citations are
preserved. Default top-k is 5 (`--top-k`).

There is no query planner, query expansion, BM25, rank fusion, or reranker in
this mode. The former `--planner-model`, `--planner-timeout`,
`--advanced-candidate-k`, and `--advanced-rrf-k` options have been removed.
`--skip-rerank` is redundant because this mode always skips reranking.

Configure `--collection-name`, `--qdrant-host`, `--qdrant-port`, and
`--model-path` to select the store and embedding model. The collection must
already contain indexed chunks with dense vectors and text payloads. Retrieval
and generation use Qdrant payloads. A local `--chunks` Parquet file is optional
for `--retrieval-only`; full evaluation requires the matching corpus to hydrate
retrieval context for the quality metrics. Reports identify the retriever as
`advanced_dense` and retain the
original search query, top-k, and retrieved IDs under `metadata.retrieval_trace`.

Run retrieval evaluation with local models and Qdrant, without LLM credentials:

```bash
uv run vacxis-eval --advanced-rag --retrieval-only --top-k 10 \
  --out-json evaluation/advanced_k10.json --out-md evaluation/advanced_k10.md
```

Full RAG evaluation also requires the standard generation/judge configuration:

```bash
uv run vacxis-eval --advanced-rag --top-k 4 --ir-k-values 1 3 4 \
  --skip-unanswerable \
  --out-json evaluation/advanced_rag_k4.json --out-md evaluation/advanced_rag_k4.md
```

Compare variants with the same dataset, corpus, and final top-k. Offline tests
verify the pipeline contract; retrieval quality requires a live evaluation.
The known question/reference inconsistencies in the corrected gold dataset
still need human review before treating every miss as a retrieval failure.

## Resumable generation and evaluation

Generate saved answers using `vacxis-generation --baseline` (naive BM25),
`--advanced-rag` (dense-only), or neither flag (hybrid). All three support
`--input-file`, `--out-file`, `--resume`, and `--top-k 5`. Each completed answer
is saved atomically. Keep the accompanying `.manifest.json` file: resume rejects
changed inputs or experiment settings. Use a new output path when changing the
model, dataset, retrieval settings, or corpus. Corpus contents and live Qdrant
state must remain unchanged between attempts; they are not snapshotted.

Both CLIs accept `--retries 5 --retry-initial 1 --retry-max 60` for bounded
exponential backoff with jitter on transient API errors. Retries are additional
attempts. Authentication and permission failures stop immediately. Generation
accepts `--timeout 480`. Evaluation accepts `--judge-request-timeout 360`,
`--schema-retries 3`, and `--judge-timeout 3600` (DeepEval task budget).

For explicit evaluation recovery, use `vacxis-eval --results OUTPUT.json
--checkpoint REPORT.checkpoint.json --resume --skip-unanswerable`. The checkpoint
contains a fingerprint and a partial `report`, saved after each fully scored
query. Failed or interrupted queries are retried on restart; completed queries
are reused independently of DeepEval's cache. Reports include completion and
coverage metadata. Resume rejects changed answers, contexts, dataset, judge,
threshold, or metric selection. Final `--out-json` and `--out-md` reports are
written after the full run. The checkpoint is single-writer; do not run multiple
processes against the same output path. IR-only scoring needs no API retries:
rerun it against saved answers.

### Environment and response diagnostics

Set `OPENROUTER_API_KEY` and `VACXIS_GENERATION_MODEL` in your shell.
`VACXIS_LLM_BASE_URL` defaults to `https://openrouter.ai/api/v1`.
Set `VACXIS_EVALUATION_MODEL` to choose the judge independently; otherwise it
falls back to the generation model. The CLIs do not automatically load `.env`.
Retry and timeout controls are CLI flags rather than environment variables.

Validation errors include the failure reason, finish reason, content type and
length, reasoning length (when provided), response ID, model, and completion
token count. To include bounded content, refusal, and reasoning excerpts, set
`VACXIS_LLM_ERROR_PREVIEW_CHARS=1000` (default 0; capped at 4000 per field).
The configured API key is redacted before excerpts are truncated. Excerpts can
contain dataset or model text and appear wherever exception messages are saved.

### Unanswerable dataset

Generate actual outputs for `test_dataset.json` using `--dataset-format legacy`
and a separate output file per baseline, then evaluate those files with
`--dataset test_dataset.json --results OUTPUT.json --skip-ir
--skip-unanswerable --checkpoint CHECKPOINT.json --resume`. The final flag
`--skip-unanswerable` disables only the separate auxiliary evaluation pass;
it does not remove unanswerable rows from the chosen dataset. Do not use
`--use-ground-truth-chunks`: retrieval should still run for unanswerable queries.
IR metrics are undefined for queries without relevant-chunk labels; inspect
refusal correctness, answer relevancy, faithfulness, and correctness instead.
Contextual metrics need cautious interpretation when the expected answer is a
refusal and the dataset provides no positive retrieval evidence.

### Spacing LLM calls to reduce rate limits

Add `--llm-call-delay 5` to either `vacxis-generation` or `vacxis-eval` to
require at least five seconds between request starts within that process.
The default is 0 (no added spacing). Generation calls, DeepEval judge calls,
schema-correction calls, and transport retries all use the spacing policy.
Concurrent async judge calls share the same pacing gate; waits do not block
the event loop. Retry backoff still applies, and pacing waits only for any
remaining gap. The first request can start immediately.

You can add or change this flag while resuming an existing checkpoint; it does
not change the experiment fingerprint. For example, rerun your existing
`--resume` command with `--llm-call-delay 10 --retry-initial 5 --retry-max 120`.
This spaces request starts, rather than waiting ten seconds after each response.
Separate processes do not share this limit; run one baseline at a time if they
use the same provider quota. Spacing helps request-rate limits but cannot
resolve exhausted credits or token quotas on its own.
