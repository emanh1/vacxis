# VacxiS DeepEval Report: RETRIEVAL
Total evaluated queries: 300

## Overall Metrics
| Metric | Mean | Min | Max | Pass Rate | Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Hit@1 | 0.3067 | 0.0000 | 1.0000 | 30.67% | 300 |
| Recall@1 | 0.2567 | 0.0000 | 1.0000 | 30.67% | 300 |
| Precision@1 | 0.3067 | 0.0000 | 1.0000 | 30.67% | 300 |
| MRR@1 | 0.3067 | 0.0000 | 1.0000 | 30.67% | 300 |
| NDCG@1 | 0.3067 | 0.0000 | 1.0000 | 30.67% | 300 |
| MAP@1 | 0.2567 | 0.0000 | 1.0000 | 30.67% | 300 |
| Hit@3 | 0.4767 | 0.0000 | 1.0000 | 47.67% | 300 |
| Recall@3 | 0.4133 | 0.0000 | 1.0000 | 47.67% | 300 |
| Precision@3 | 0.1689 | 0.0000 | 0.6667 | 3.00% | 300 |
| MRR@3 | 0.3806 | 0.0000 | 1.0000 | 41.00% | 300 |
| NDCG@3 | 0.3641 | 0.0000 | 1.0000 | 43.00% | 300 |
| MAP@3 | 0.3306 | 0.0000 | 1.0000 | 38.67% | 300 |
| Hit@5 | 0.5900 | 0.0000 | 1.0000 | 59.00% | 300 |
| Recall@5 | 0.5167 | 0.0000 | 1.0000 | 59.00% | 300 |
| Precision@5 | 0.1280 | 0.0000 | 0.4000 | 0.00% | 300 |
| MRR@5 | 0.4064 | 0.0000 | 1.0000 | 41.00% | 300 |
| NDCG@5 | 0.4093 | 0.0000 | 1.0000 | 44.33% | 300 |
| MAP@5 | 0.3562 | 0.0000 | 1.0000 | 39.00% | 300 |

IR metrics use binary relevance against labeled chunk IDs; unlabeled alternatives count as nonrelevant.
Pass rate is the fraction of scored queries meeting the configured threshold (0.5); it is not a retrieval accuracy measure.
Scored queries: 300; excluded without relevance labels: 0.

## Run Settings
- mode: live
- dataset: evaluation/eval_dataset_300_final_fixed.json
- results: None
- retriever: advanced_dense
- retriever_mode: advanced
- top_k: 5
- chunks: chunks_storage/chunks.parquet
- dense_only: True
- skip_rerank: True
- collection_name: VacxiS_Capstone_FA26
- model_path: models/bge-m3

## Breakdown by Answerability
| Answerability | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| ANSWERABLE | 210 | 0.3381 | 0.2786 | 0.3381 | 0.3381 | 0.3381 | 0.2786 | 0.5381 | 0.4619 | 0.1889 | 0.4230 | 0.4029 | 0.3627 | 0.6476 | 0.5619 | 0.1390 | 0.4475 | 0.4462 | 0.3871 |
| PARTIALLY_ANSWERABLE | 90 | 0.2333 | 0.2056 | 0.2333 | 0.2333 | 0.2333 | 0.2056 | 0.3333 | 0.3000 | 0.1222 | 0.2815 | 0.2737 | 0.2556 | 0.4556 | 0.4111 | 0.1022 | 0.3104 | 0.3233 | 0.2842 |

## Breakdown by Complexity
| Complexity | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| MULTI_HOP | 90 | 0.3333 | 0.1667 | 0.3333 | 0.3333 | 0.3333 | 0.1667 | 0.5222 | 0.3111 | 0.2074 | 0.4148 | 0.3054 | 0.2481 | 0.6556 | 0.4111 | 0.1644 | 0.4448 | 0.3553 | 0.2776 |
| SINGLE_HOP | 150 | 0.3600 | 0.3600 | 0.3600 | 0.3600 | 0.3600 | 0.3600 | 0.5333 | 0.5333 | 0.1778 | 0.4367 | 0.4615 | 0.4367 | 0.6267 | 0.6267 | 0.1253 | 0.4583 | 0.5002 | 0.4583 |
| TABLE_REASONING | 60 | 0.1333 | 0.1333 | 0.1333 | 0.1333 | 0.1333 | 0.1333 | 0.2667 | 0.2667 | 0.0889 | 0.1889 | 0.2087 | 0.1889 | 0.4000 | 0.4000 | 0.0800 | 0.2189 | 0.2632 | 0.2189 |

## Breakdown by Query Style
| Query Style | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| CLINICAL_STYLE | 120 | 0.2667 | 0.2250 | 0.2667 | 0.2667 | 0.2667 | 0.2250 | 0.4417 | 0.3708 | 0.1528 | 0.3403 | 0.3235 | 0.2903 | 0.5333 | 0.4625 | 0.1150 | 0.3611 | 0.3633 | 0.3136 |
| PATIENT_STYLE | 180 | 0.3333 | 0.2778 | 0.3333 | 0.3333 | 0.3333 | 0.2778 | 0.5000 | 0.4417 | 0.1796 | 0.4074 | 0.3912 | 0.3574 | 0.6278 | 0.5528 | 0.1367 | 0.4366 | 0.4400 | 0.3846 |

## Breakdown by Difficulty
| Difficulty | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| UNSPECIFIED | 300 | 0.3067 | 0.2567 | 0.3067 | 0.3067 | 0.3067 | 0.2567 | 0.4767 | 0.4133 | 0.1689 | 0.3806 | 0.3641 | 0.3306 | 0.5900 | 0.5167 | 0.1280 | 0.4064 | 0.4093 | 0.3562 |