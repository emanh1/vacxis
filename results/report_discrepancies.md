# Results audit and report discrepancies

Reviewed November 20, 2026 against the saved files and source on main at commit 7798195. The report work and referenced result files were subsequently transferred to report-write. Implementation observations below refer to that inspected main revision. The report branch has now merged that main revision, so its runtime source matches the implementation used for this audit. This audit reads existing results and executes only the dependency-light scoring functions. It does not perform retrieval, load embedding models, call an LLM or establish fresh live-run reproducibility.

## Supported baseline comparison

All three reports under evaluation/ contain 300 scored queries with five retrieved IDs per query and cutoffs 1, 3 and 5. Query text, reference answers and ordered gold chunk IDs agree with evaluation/eval_dataset_300_final_fixed.json for every record.

| Configuration | Source | Hits at five | Hit@5 | Recall@5 | MRR@5 | NDCG@5 |
| --- | --- | --- | --- | --- | --- | --- |
| Naive | evaluation/ir_naive_bm25.json | 161/300 | 0.5367 | 0.4733 | 0.3747 | 0.3773 |
| Advanced | evaluation/ir_advanced.json | 177/300 | 0.5900 | 0.5167 | 0.4064 | 0.4093 |
| Hybrid | evaluation/ir_hybrid.json | 223/300 | 0.7433 | 0.6817 | 0.5532 | 0.5607 |

Hybrid exceeds Naive by 20.67 percentage points and Advanced by 15.33 percentage points in Hit@5. These are net differences in aggregate success counts, with no claim that every individual query improves.

## Discrepancies and interpretation limits

1. **The results summary omits newer baseline files.** results/ablation_study.md says separate M0/M1 files are unavailable. evaluation/ir_naive_bm25.json and evaluation/ir_advanced.json now supply these runs. Their Markdown companions are also present. The draft uses the current artifacts and preserves the original results summary for provenance.

2. **A reported scoring gain is distinct from a retrieval gain.** The strict and alias-aware M3b reports have identical ranked IDs for all 300 queries. Their Hit@5 changes from 217/300 to 235/300 solely because aliases credit constituent source IDs within merged table chunks. There are 115 records with alias maps. Eighteen additional queries receive a hit at five under that rule.

3. **Matching definitions differ across configurations.** M2 and the three baseline reports use direct ID matching. M3b alias and M4 use rank-preserving merged-source matching. One merged result occupies one ranking slot and can cover multiple source IDs. The current scorer adjusts MAP/NDCG denominators when multiple relevant sources collapse within a slot. Explain these definitions when comparing merged and unmerged evidence. A controlled comparison should retain an explicit evidence-unit and scoring policy.

4. **Percentage-point and relative-percent changes are mixed in the source wording.** A change from 0.4233 to 0.4933 is approximately seven percentage points. Its relative increase is approximately 16.5%. The draft uses percentage points for changes in success rates.

5. **Some source deltas use rounded inputs.** The table-reasoning Hit@5 gain from 35/60 to 46/60 is 18.33 percentage points. The summary prints 18.34. Overall Recall@3 for M4 minus M2 differs by approximately 0.0516667 using per-query means, which rounds to 0.0517. The summary prints 0.0516. Use unrounded per-query means for derived differences.

6. **Graph generation benefit remains unmeasured.** The summary concludes that LightRAG's main value lies in generation/DeepEval. All seven inspected reports are retrieval-only, with empty actual_output and retrieval_context fields. These fields provide no generated answers or LLM-judged quality observations. That conclusion is a hypothesis requiring generation results.

7. **M4's nominal top-k and stored list length differ.** Its metadata says top_k=5. For 297 queries the saved list contains five document IDs followed by lightrag_kg_context at rank six. For three queries it contains five document IDs with no synthetic KG entry. The graph entry is never scored at cutoffs 1, 3 or 5. Record the additional context budget when comparing generation later.

8. **Equal aggregate metrics conceal ranking differences.** M4 and M3b alias have equal aggregate Hit, Recall and Precision at the reported cutoffs. Their first-five order differs for nine queries, and first-five membership differs for six. M4's MRR@5 and NDCG@5 are slightly lower. The files do not establish that graph retrieval changes no evidence or improves generation.

9. **There are two Hybrid artifacts with equal metrics but different rankings.** eval_ir_m2_hybrid.json and evaluation/ir_hybrid.json have equal aggregate scores and recorded run settings. Their retrieved IDs differ for eval_0201 and eval_0255. All remaining saved fields agree. Missing code/index revisions and run timestamps leave the cause unresolved. The draft uses evaluation/ir_hybrid.json for the primary baseline table and the original M2 file for the extension comparison.

10. **The four-chunk generation file is a separate artifact.** evaluation/eval_dataset_300_generation_results.json records four retrieved IDs per response. It cannot establish which generator or answer quality belongs to the new top-five retrieval runs. Keep their provenance distinct.

11. **Metric success flags do not measure clinical accuracy.** Reports apply a generic threshold of 0.5 to scores. For example, with at most two gold chunks per query, Precision@5 is at most 0.4 under the direct-ID baseline scorer, so its pass rate can be zero despite useful retrieval. Use metric means, denominators and definitions rather than treating the generic pass rate as answer accuracy.

12. **Artifact and code availability have changed since the team's answers.** The team previously reported no concrete design for Table-aware, Graph RAG or Agent RAG. Current main includes Table-aware and LightRAG code, storage artifacts and preliminary retrieval reports. They justify preliminary descriptions, while final implementation records and answer-level results remain pending. M4's mapping to the final fifth Graph RAG configuration still needs confirmation. No Agent RAG result is established by the inspected artifacts.

13. **Run manifests are incomplete.** Report metadata identifies dataset paths, collections, model directories and top-k. It does not freeze code commits, dataset/corpus hashes, Qdrant state, local checkpoint identities, hardware or all candidate/fusion settings. M3b metadata has merge_table_chunks=False even though the named collection and summary describe upstream merged-table indexing. That flag is a retrieval-time setting, so it does not establish the collection's indexing strategy. Preserve the original indexing command and summary-cache configuration.

14. **An earlier draft limitation is now stale.** The draft previously described response validation as disabled. Current main implements checks for empty/non-text completions, provider refusals and unexpected finish reasons. The updated draft reflects the current implementation without inferring which validation revision applied to historical runs.

15. **Document locations differ between branches.** The active workspace was main and lacked the report documents from report-write. The draft, clarification record and its local template/bibliography dependencies were recovered from report-write before editing. The documents were subsequently moved onto report-write. To resolve implementation drift, main revision 7798195 has now been merged into report-write, retaining the report files and their tracked .local dependencies. Source implementation, tests and result artifacts are checked against that main revision. The historical run-provenance questions remain open.

## Checks completed

- Inspected seven JSON reports and their recorded populations, settings and per-query data.
- Recomputed all stored per-query metrics using the current pure IR scoring functions, including merged-source aliases. No score mismatches were found.
- Recomputed overall means from per-query scores. All seven agree with stored four-decimal aggregates.
- Compared questions, references and gold IDs to the corrected 300-question dataset. All 2,100 report records agree.
- Compared strict/alias rankings, graph first-five rankings and the two Hybrid artifacts directly.

## Questions to revisit

- Which baseline and extension runs should be designated final, with code/index revisions and original commands?
- Does M4 represent the intended Graph RAG configuration, and should its extra context be given the same budget as other generation configurations?
- Which table-summary model, prompt and cache revision built the table-dual collection?
- Can the difference between the two Hybrid rankings be explained by index state, reranking ties, code revision or another recorded cause?
- Where are the corresponding generation, citation-support, abstention, latency and expert-review results?

The original deferred questions remain in ../report_clarifications.md.

## Live verification commands for the other machine

Live reruns remain pending. Run from the repository root on the machine with the matching Qdrant collections, models and corpus. The graph command additionally requires the project's LightRAG dependency, graph store and its environment settings. These commands test the current code and store state; they cannot recover the historical index revision. Keep the audit output names separate from the supplied reports.

```bash
uv run --extra retrieval vacxis-eval --retrieval-only --retriever-mode naive \
  --dataset evaluation/eval_dataset_300_final_fixed.json --chunks chunks_storage/chunks.parquet \
  --top-k 5 --ir-k-values 1 3 5 --skip-unanswerable \
  --out-json results/audit_naive.json --out-md results/audit_naive.md

uv run --extra retrieval vacxis-eval --retrieval-only --retriever-mode advanced \
  --dataset evaluation/eval_dataset_300_final_fixed.json --chunks chunks_storage/chunks.parquet \
  --collection-name VacxiS_Capstone_FA26 --top-k 5 --ir-k-values 1 3 5 --skip-unanswerable \
  --out-json results/audit_advanced.json --out-md results/audit_advanced.md

uv run --extra retrieval vacxis-eval --retrieval-only --retriever-mode hybrid \
  --dataset evaluation/eval_dataset_300_final_fixed.json --chunks chunks_storage/chunks.parquet \
  --collection-name VacxiS_Capstone_FA26 --top-k 5 --ir-k-values 1 3 5 --skip-unanswerable \
  --out-json results/audit_hybrid.json --out-md results/audit_hybrid.md

uv run --extra retrieval vacxis-eval --retrieval-only --retriever-mode hybrid \
  --dataset evaluation/eval_dataset_300_final_fixed.json --chunks chunks_storage/chunks.parquet \
  --collection-name VacxiS_Capstone_FA26_table_dual --top-k 5 --ir-k-values 1 3 5 --skip-unanswerable \
  --out-json results/audit_table.json --out-md results/audit_table.md

uv run --extra retrieval --env-file .env vacxis-eval --retrieval-only --retriever-mode hybrid_lightrag \
  --dataset evaluation/eval_dataset_300_final_fixed.json --chunks chunks_storage/chunks.parquet \
  --collection-name VacxiS_Capstone_FA26_table_dual --top-k 5 --ir-k-values 1 3 5 --skip-unanswerable \
  --lightrag-mode hybrid --lightrag-fusion rrf_rerank --lightrag-top-k 20 --lightrag-chunk-top-k 5 \
  --out-json results/audit_graph.json --out-md results/audit_graph.md
```
