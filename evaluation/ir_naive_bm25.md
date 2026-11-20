# VacxiS DeepEval Report: RETRIEVAL
Total evaluated queries: 300

## Overall Metrics
| Metric | Mean | Min | Max | Pass Rate | Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Hit@1 | 0.2767 | 0.0000 | 1.0000 | 27.67% | 300 |
| Recall@1 | 0.2383 | 0.0000 | 1.0000 | 27.67% | 300 |
| Precision@1 | 0.2767 | 0.0000 | 1.0000 | 27.67% | 300 |
| MRR@1 | 0.2767 | 0.0000 | 1.0000 | 27.67% | 300 |
| NDCG@1 | 0.2767 | 0.0000 | 1.0000 | 27.67% | 300 |
| MAP@1 | 0.2383 | 0.0000 | 1.0000 | 27.67% | 300 |
| Hit@3 | 0.4533 | 0.0000 | 1.0000 | 45.33% | 300 |
| Recall@3 | 0.3900 | 0.0000 | 1.0000 | 45.33% | 300 |
| Precision@3 | 0.1567 | 0.0000 | 0.6667 | 1.67% | 300 |
| MRR@3 | 0.3556 | 0.0000 | 1.0000 | 39.67% | 300 |
| NDCG@3 | 0.3409 | 0.0000 | 1.0000 | 39.33% | 300 |
| MAP@3 | 0.3078 | 0.0000 | 1.0000 | 35.67% | 300 |
| Hit@5 | 0.5367 | 0.0000 | 1.0000 | 53.67% | 300 |
| Recall@5 | 0.4733 | 0.0000 | 1.0000 | 53.67% | 300 |
| Precision@5 | 0.1153 | 0.0000 | 0.4000 | 0.00% | 300 |
| MRR@5 | 0.3747 | 0.0000 | 1.0000 | 39.67% | 300 |
| NDCG@5 | 0.3773 | 0.0000 | 1.0000 | 40.00% | 300 |
| MAP@5 | 0.3292 | 0.0000 | 1.0000 | 35.67% | 300 |

IR metrics use binary relevance against labeled chunk IDs; unlabeled alternatives count as nonrelevant.
Pass rate is the fraction of scored queries meeting the configured threshold (0.5); it is not a retrieval accuracy measure.
Scored queries: 300; excluded without relevance labels: 0.

## Run Settings
- mode: live
- dataset: evaluation/eval_dataset_300_final_fixed.json
- results: None
- retriever: bm25
- retriever_mode: naive
- top_k: 5
- chunks: chunks_storage/chunks.parquet

## Breakdown by Answerability
| Answerability | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| ANSWERABLE | 210 | 0.3190 | 0.2738 | 0.3190 | 0.3190 | 0.3190 | 0.2738 | 0.5190 | 0.4452 | 0.1794 | 0.4103 | 0.3916 | 0.3540 | 0.6095 | 0.5357 | 0.1305 | 0.4310 | 0.4307 | 0.3767 |
| PARTIALLY_ANSWERABLE | 90 | 0.1778 | 0.1556 | 0.1778 | 0.1778 | 0.1778 | 0.1556 | 0.3000 | 0.2611 | 0.1037 | 0.2278 | 0.2227 | 0.2000 | 0.3667 | 0.3278 | 0.0800 | 0.2433 | 0.2526 | 0.2183 |

## Breakdown by Complexity
| Complexity | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| MULTI_HOP | 90 | 0.2556 | 0.1278 | 0.2556 | 0.2556 | 0.2556 | 0.1278 | 0.4778 | 0.2667 | 0.1778 | 0.3556 | 0.2543 | 0.1963 | 0.5556 | 0.3444 | 0.1378 | 0.3728 | 0.2924 | 0.2210 |
| SINGLE_HOP | 150 | 0.3533 | 0.3533 | 0.3533 | 0.3533 | 0.3533 | 0.3533 | 0.5333 | 0.5333 | 0.1778 | 0.4344 | 0.4599 | 0.4344 | 0.6200 | 0.6200 | 0.1240 | 0.4548 | 0.4961 | 0.4548 |
| TABLE_REASONING | 60 | 0.1167 | 0.1167 | 0.1167 | 0.1167 | 0.1167 | 0.1167 | 0.2167 | 0.2167 | 0.0722 | 0.1583 | 0.1732 | 0.1583 | 0.3000 | 0.3000 | 0.0600 | 0.1775 | 0.2076 | 0.1775 |

## Breakdown by Query Style
| Query Style | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| CLINICAL_STYLE | 120 | 0.2000 | 0.1667 | 0.2000 | 0.2000 | 0.2000 | 0.1667 | 0.3417 | 0.2875 | 0.1167 | 0.2639 | 0.2498 | 0.2229 | 0.4083 | 0.3583 | 0.0900 | 0.2793 | 0.2816 | 0.2427 |
| PATIENT_STYLE | 180 | 0.3278 | 0.2861 | 0.3278 | 0.3278 | 0.3278 | 0.2861 | 0.5278 | 0.4583 | 0.1833 | 0.4167 | 0.4017 | 0.3644 | 0.6222 | 0.5500 | 0.1322 | 0.4383 | 0.4411 | 0.3869 |

## Breakdown by Difficulty
| Difficulty | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| UNSPECIFIED | 300 | 0.2767 | 0.2383 | 0.2767 | 0.2767 | 0.2767 | 0.2383 | 0.4533 | 0.3900 | 0.1567 | 0.3556 | 0.3409 | 0.3078 | 0.5367 | 0.4733 | 0.1153 | 0.3747 | 0.3773 | 0.3292 |