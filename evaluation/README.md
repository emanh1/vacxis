# VaxiS evaluation

## Working data

- `eval_dataset_300_final_fixed.json` is the default for live evaluation. It contains 300 working goldens, including 210 answerable and 90 partially answerable records. Sixty records are labeled table reasoning.
- `eval_dataset_300_final.json` is the earlier version. Ordered source-chunk lists differ for 279 questions.
- `eval_dataset_300_generation_results.json` contains 300 saved responses matching the corrected goldens. Each response contains four retrieved IDs. Score these rankings at cutoffs up to four.
- `../test_dataset.json` contains 460 records in the query-driven dataset format and is the CLI default for the separate unanswerable pass. Record the actual subset used to assess abstention.

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

4. **Table-aware** — [PLACEHOLDER: implementation revision, exact invocation, table-dependent subset, metrics and error examples.]
5. **Graph RAG** — [PLACEHOLDER: implementation revision, graph schema, exact invocation, multi-hop subset, metrics and provenance examples.]
6. **Agent RAG** — [PLACEHOLDER: implementation revision, tool configuration, exact invocation, metrics, tool-call counts and latency.]

[PLACEHOLDER: attach verified retrieval, answer quality, citation support, abstention and latency results for each completed run.]
