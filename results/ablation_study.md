# Ablation Study: IR Evaluation Comparison

Báo cáo tổng hợp kết quả đánh giá Information Retrieval (IR) trên tập `evaluation/eval_dataset_300_final_fixed.json` (`300` câu hỏi, `top_k=5`, `BGE-M3` + `BGE-Reranker`).

## 1. Các cấu hình hiện có

| Ký hiệu | File nguồn | Collection Qdrant | Cơ chế chấm điểm | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| **M2 (Hybrid)** | `eval_ir_m2_hybrid.json` | `VacxiS_Capstone_FA26` | Strict ID | Hybrid (Dense + Sparse + Rerank) trên chunk gốc |
| **M3b (Strict ID)** | `eval_ir_m3b_table_dual_before_preserving.json` | `VacxiS_Capstone_FA26_table_dual` | Strict ID | Hybrid + Table-Aware (`summarize_dual`), trước khi sửa logic khớp `merged_chunk_ids` |
| **M3b (Alias-Aware)** | `eval_ir_m3b_table_dual.json` | `VacxiS_Capstone_FA26_table_dual` | Rank-preserving `merged_chunk_ids` | Hybrid + Table-Aware (`summarize_dual`), tính đúng các sub-chunk đã gộp bảng |
| **M4 (LightRAG RRF)** | `eval_ir_m4_hybrid_lightrag_rrf.json` | `VacxiS_Capstone_FA26_table_dual` + KG | Rank-preserving `merged_chunk_ids` | Hybrid + Table-Aware + LightRAG (`hybrid` + `rrf_rerank`) |

> Ghi chú: Các file hiện có gồm M2, M3b (Strict/Alias) và M4 (`hybrid_lightrag` + `rrf_rerank`). Chưa có file riêng cho M0/M1.

---

## 2. Tổng hợp Overall IR Metrics (300 câu hỏi)

| Metric | M2 (Hybrid) | M3b (Strict ID) | M3b (Alias-Aware) | M4 (LightRAG RRF) | Delta (M4 vs M2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hit@1** | 0.4233 | 0.4567 | **0.4933** | **0.4933** | **+0.0700 (+7.00%)** |
| **Recall@1** | 0.3517 | 0.3850 | **0.4217** | **0.4217** | **+0.0700 (+7.00%)** |
| **Precision@1** | 0.4233 | 0.4567 | **0.4933** | **0.4933** | **+0.0700 (+7.00%)** |
| **MRR@1** | 0.4233 | 0.4567 | **0.4933** | **0.4933** | **+0.0700 (+7.00%)** |
| **NDCG@1** | 0.4233 | 0.4567 | **0.4933** | **0.4933** | **+0.0700 (+7.00%)** |
| **MAP@1** | 0.3517 | 0.3850 | **0.4217** | **0.4217** | **+0.0700 (+7.00%)** |
| **Hit@3** | 0.6767 | 0.6667 | **0.7300** | **0.7300** | **+0.0533 (+5.33%)** |
| **Recall@3** | 0.6167 | 0.6050 | **0.6683** | **0.6683** | **+0.0516 (+5.16%)** |
| **Precision@3** | 0.2500 | 0.2444 | **0.2678** | **0.2678** | **+0.0178 (+1.78%)** |
| **MRR@3** | 0.5383 | 0.5511 | **0.5989** | 0.5983 | **+0.0600 (+6.00%)** |
| **NDCG@3** | 0.5317 | 0.5372 | **0.5897** | 0.5890 | **+0.0573 (+5.73%)** |
| **MAP@3** | 0.4844 | 0.4956 | **0.5444** | 0.5433 | **+0.0589 (+5.89%)** |
| **Hit@5** | 0.7433 | 0.7233 | **0.7833** | **0.7833** | **+0.0400 (+4.00%)** |
| **Recall@5** | 0.6817 | 0.6533 | **0.7167** | **0.7167** | **+0.0350 (+3.50%)** |
| **Precision@5** | 0.1680 | 0.1613 | **0.1753** | **0.1753** | **+0.0073 (+0.73%)** |
| **MRR@5** | 0.5532 | 0.5643 | **0.6114** | 0.6108 | **+0.0576 (+5.76%)** |
| **NDCG@5** | 0.5607 | 0.5597 | **0.6122** | 0.6115 | **+0.0508 (+5.08%)** |
| **MAP@5** | 0.5019 | 0.5089 | **0.5581** | 0.5570 | **+0.0551 (+5.51%)** |

---

## 3. Phân tích theo độ phức tạp (`by_complexity`)

### `TABLE_REASONING` (n = 60)

| Metric | M2 (Hybrid) | M3b (Strict ID) | M3b (Alias-Aware) | Delta (M3b Alias vs M2) |
| :--- | :---: | :---: | :---: | :---: |
| **Hit@1 / Recall@1** | 0.1167 | 0.2833 | **0.4667** | **+0.3500 (+35.00%)** |
| **Hit@3 / Recall@3** | 0.4667 | 0.4833 | **0.7667** | **+0.3000 (+30.00%)** |
| **Hit@5 / Recall@5** | 0.5833 | 0.4833 | **0.7667** | **+0.1834 (+18.34%)** |
| **MRR@5** | 0.3044 | 0.3694 | **0.5972** | **+0.2928 (+29.28%)** |
| **NDCG@5** | 0.3746 | 0.3986 | **0.6407** | **+0.2661 (+26.61%)** |
| **MAP@5** | 0.3044 | 0.3694 | **0.5972** | **+0.2928 (+29.28%)** |

### `SINGLE_HOP` (n = 150)

| Metric | M2 (Hybrid) | M3b (Strict ID) | M3b (Alias-Aware) | Delta (M3b Alias vs M2) |
| :--- | :---: | :---: | :---: | :---: |
| **Hit@1 / Recall@1** | 0.5133 | 0.5133 | 0.5133 | 0.0000 |
| **Hit@3 / Recall@3** | **0.7800** | 0.7600 | 0.7600 | -0.0200 (-2.00%) |
| **Hit@5 / Recall@5** | **0.8133** | 0.8067 | 0.8067 | -0.0066 (-0.66%) |
| **MRR@5** | **0.6392** | 0.6343 | 0.6343 | -0.0049 (-0.49%) |
| **NDCG@5** | **0.6834** | 0.6780 | 0.6780 | -0.0054 (-0.54%) |

### `MULTI_HOP` (n = 90)

| Metric | M2 (Hybrid) | M3b (Strict ID) | M3b (Alias-Aware) | Delta (M3b Alias vs M2) |
| :--- | :---: | :---: | :---: | :---: |
| **Hit@1** | 0.4778 | 0.4778 | 0.4778 | 0.0000 |
| **Recall@1** | 0.2389 | 0.2389 | 0.2389 | 0.0000 |
| **Hit@3** | 0.6444 | 0.6333 | **0.6556** | **+0.0112 (+1.12%)** |
| **Recall@3** | 0.4444 | 0.4278 | **0.4500** | **+0.0056 (+0.56%)** |
| **Hit@5** | 0.7333 | 0.7444 | **0.7556** | **+0.0223 (+2.23%)** |
| **Recall@5** | 0.5278 | 0.5111 | **0.5333** | **+0.0055 (+0.55%)** |
| **MRR@5** | 0.5756 | 0.5774 | **0.5826** | **+0.0070 (+0.70%)** |
| **NDCG@5** | 0.4802 | 0.4700 | **0.4836** | **+0.0034 (+0.34%)** |

---

## 4. Phân tích theo phong cách & khả năng trả lời

### Theo `by_query_style`

| Nhóm | Metric | M2 (Hybrid) | M3b (Strict ID) | M3b (Alias-Aware) | Delta (M3b Alias vs M2) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PATIENT_STYLE** (n=180) | Hit@1 / Hit@5 | 0.4333 / 0.7444 | 0.4556 / 0.7111 | **0.4833 / 0.7667** | +5.00% / +2.23% |
| **PATIENT_STYLE** (n=180) | Recall@5 / NDCG@5 | 0.6917 / 0.5702 | 0.6528 / 0.5595 | **0.7083 / 0.6033** | +1.66% / +3.31% |
| **CLINICAL_STYLE** (n=120) | Hit@1 / Hit@5 | 0.4083 / 0.7417 | 0.4583 / 0.7417 | **0.5083 / 0.8083** | +10.00% / +6.66% |
| **CLINICAL_STYLE** (n=120) | Recall@5 / NDCG@5 | 0.6667 / 0.5465 | 0.6542 / 0.5600 | **0.7292 / 0.6257** | +6.25% / +7.92% |

### Theo `by_answerability`

| Nhóm | Metric | M2 (Hybrid) | M3b (Strict ID) | M3b (Alias-Aware) | Delta (M3b Alias vs M2) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ANSWERABLE** (n=210) | Hit@1 / Hit@5 | 0.4714 / 0.7762 | 0.5000 / 0.7619 | **0.5429 / 0.8238** | +7.15% / +4.76% |
| **ANSWERABLE** (n=210) | Recall@5 / NDCG@5 | 0.7143 / 0.5991 | 0.6857 / 0.5959 | **0.7548 / 0.6545** | +4.05% / +5.54% |
| **PARTIALLY_ANSWERABLE** (n=90) | Hit@1 / Hit@5 | 0.3111 / 0.6667 | 0.3556 / 0.6333 | **0.3778 / 0.6889** | +6.67% / +2.22% |
| **PARTIALLY_ANSWERABLE** (n=90) | Recall@5 / NDCG@5 | 0.6056 / 0.4711 | 0.5778 / 0.4753 | **0.6278 / 0.5135** | +2.22% / +4.24% |

---

## 5. Nhận xét chính

1. **Table-Aware (`summarize_dual`) giải quyết điểm nghẽn truy xuất bảng:** Trên nhóm `TABLE_REASONING` (60 câu), `Hit@1` tăng từ `0.1167` lên `0.4667` (`+35.00%`) và `Hit@3` từ `0.4667` lên `0.7667` (`+30.00%`), kéo toàn bộ `Overall Hit@1` tăng `+7.00%` và `Overall Hit@5` tăng `+4.00%` so với M2.
2. **Tác động của Alias-Aware Matching (`merged_chunk_ids`):** Trước khi sửa logic chấm IR (`M3b Strict ID`), nhiều bảng đã truy xuất đúng vào Top-5 nhưng bị tính miss do ID trả về là primary ID sau gộp bảng còn nhãn chỉ tới sub-chunk ID, làm `Hit@5` bị tụt giả `-2.00%` (`0.7433 -> 0.7233`). Sau khi bật rank-preserving `merged_chunk_ids` alias matching, M3b vượt M2 đồng loạt ở mọi chỉ số tổng thể (`k=1, 3, 5`).
3. **Ổn định trên văn bản thường:** Nhóm `SINGLE_HOP` giữ nguyên `Hit@1 = 0.5133` và chỉ dao động rất nhẹ ở `@3/@5` (`-0.66%` đến `-2.00%`), trong khi `MULTI_HOP` cải thiện nhẹ ở `@3/@5` (`Hit@5 +2.23%`).
4. **Tác động của M4 (`hybrid_lightrag` + `rrf_rerank`) ở retrieval-only:** Giữ nguyên toàn bộ `Hit@k`, `Recall@k`, `Precision@k` so với `M3b (Alias-Aware)` (`Hit@1 = 0.4933`, `Hit@5 = 0.7833`, `Recall@5 = 0.7167`), chỉ chênh lệch rất nhỏ ở thứ tự xếp hạng (`MRR@5`: `0.6108` vs `0.6114`, `NDCG@5`: `0.6115` vs `0.6122`). Điều này cho thấy giá trị chính của LightRAG nằm ở `lightrag_kg_context` trong bước **Generation / DeepEval** hơn là thay đổi Top-5 chunk-level IR.
