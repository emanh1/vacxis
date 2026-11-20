# VacxiS DeepEval Report: RETRIEVAL
Total evaluated queries: 300

## Overall Metrics
| Metric | Mean | Min | Max | Pass Rate | Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Hit@1 | 0.4233 | 0.0000 | 1.0000 | 42.33% | 300 |
| Recall@1 | 0.3517 | 0.0000 | 1.0000 | 42.33% | 300 |
| Precision@1 | 0.4233 | 0.0000 | 1.0000 | 42.33% | 300 |
| MRR@1 | 0.4233 | 0.0000 | 1.0000 | 42.33% | 300 |
| NDCG@1 | 0.4233 | 0.0000 | 1.0000 | 42.33% | 300 |
| MAP@1 | 0.3517 | 0.0000 | 1.0000 | 42.33% | 300 |
| Hit@3 | 0.6767 | 0.0000 | 1.0000 | 67.67% | 300 |
| Recall@3 | 0.6167 | 0.0000 | 1.0000 | 67.67% | 300 |
| Precision@3 | 0.2500 | 0.0000 | 0.6667 | 7.33% | 300 |
| MRR@3 | 0.5383 | 0.0000 | 1.0000 | 60.67% | 300 |
| NDCG@3 | 0.5317 | 0.0000 | 1.0000 | 64.67% | 300 |
| MAP@3 | 0.4844 | 0.0000 | 1.0000 | 58.67% | 300 |
| Hit@5 | 0.7433 | 0.0000 | 1.0000 | 74.33% | 300 |
| Recall@5 | 0.6817 | 0.0000 | 1.0000 | 74.33% | 300 |
| Precision@5 | 0.1680 | 0.0000 | 0.4000 | 0.00% | 300 |
| MRR@5 | 0.5532 | 0.0000 | 1.0000 | 60.67% | 300 |
| NDCG@5 | 0.5607 | 0.0000 | 1.0000 | 66.00% | 300 |
| MAP@5 | 0.5019 | 0.0000 | 1.0000 | 59.67% | 300 |

IR metrics use binary relevance against labeled chunk IDs; unlabeled alternatives count as nonrelevant.
Pass rate is the fraction of scored queries meeting the configured threshold (0.5); it is not a retrieval accuracy measure.
Scored queries: 300; excluded without relevance labels: 0.

## Run Settings
- mode: live
- dataset: evaluation/eval_dataset_300_final_fixed.json
- results: None
- retriever: hybrid
- retriever_mode: hybrid
- top_k: 5
- chunks: chunks_storage/chunks.parquet
- collection_name: VacxiS_Capstone_FA26
- model_path: models/bge-m3
- reranker_path: models/bge-reranker
- skip_rerank: False
- merge_table_chunks: False

## Breakdown by Answerability
| Answerability | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| ANSWERABLE | 210 | 0.4714 | 0.3905 | 0.4714 | 0.4714 | 0.4714 | 0.3905 | 0.7190 | 0.6524 | 0.2667 | 0.5849 | 0.5713 | 0.5226 | 0.7762 | 0.7143 | 0.1771 | 0.5978 | 0.5991 | 0.5402 |
| PARTIALLY_ANSWERABLE | 90 | 0.3111 | 0.2611 | 0.3111 | 0.3111 | 0.3111 | 0.2611 | 0.5778 | 0.5333 | 0.2111 | 0.4296 | 0.4394 | 0.3954 | 0.6667 | 0.6056 | 0.1467 | 0.4491 | 0.4711 | 0.4126 |

## Breakdown by Complexity
| Complexity | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| MULTI_HOP | 90 | 0.4778 | 0.2389 | 0.4778 | 0.4778 | 0.4778 | 0.2389 | 0.6444 | 0.4444 | 0.2963 | 0.5556 | 0.4377 | 0.3759 | 0.7333 | 0.5278 | 0.2111 | 0.5756 | 0.4802 | 0.4048 |
| SINGLE_HOP | 150 | 0.5133 | 0.5133 | 0.5133 | 0.5133 | 0.5133 | 0.5133 | 0.7800 | 0.7800 | 0.2600 | 0.6322 | 0.6702 | 0.6322 | 0.8133 | 0.8133 | 0.1627 | 0.6392 | 0.6834 | 0.6392 |
| TABLE_REASONING | 60 | 0.1167 | 0.1167 | 0.1167 | 0.1167 | 0.1167 | 0.1167 | 0.4667 | 0.4667 | 0.1556 | 0.2778 | 0.3266 | 0.2778 | 0.5833 | 0.5833 | 0.1167 | 0.3044 | 0.3746 | 0.3044 |

## Breakdown by Query Style
| Query Style | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| CLINICAL_STYLE | 120 | 0.4083 | 0.3417 | 0.4083 | 0.4083 | 0.4083 | 0.3417 | 0.6500 | 0.5833 | 0.2361 | 0.5208 | 0.5089 | 0.4639 | 0.7417 | 0.6667 | 0.1650 | 0.5417 | 0.5465 | 0.4862 |
| PATIENT_STYLE | 180 | 0.4333 | 0.3583 | 0.4333 | 0.4333 | 0.4333 | 0.3583 | 0.6944 | 0.6389 | 0.2593 | 0.5500 | 0.5470 | 0.4981 | 0.7444 | 0.6917 | 0.1700 | 0.5608 | 0.5702 | 0.5125 |

## Breakdown by Difficulty
| Difficulty | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| UNSPECIFIED | 300 | 0.4233 | 0.3517 | 0.4233 | 0.4233 | 0.4233 | 0.3517 | 0.6767 | 0.6167 | 0.2500 | 0.5383 | 0.5317 | 0.4844 | 0.7433 | 0.6817 | 0.1680 | 0.5532 | 0.5607 | 0.5019 |