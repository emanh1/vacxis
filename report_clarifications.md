# VaxiS report clarification record

Recorded November 19, 2026. Original question numbers are retained so the discussion can resume without repeating confirmed facts. Pending items remain unresolved until the team answers them. Repository defaults describe current code and do not establish historical run settings.

## Confirmed answers

| Question | Team confirmation |
| --- | --- |
| 1 | Official name is VaxiS. |
| 2 | Final cover information in .local/main_final.tex is authoritative. Group 1, AIP491, members Nguyễn Mạnh Hùng HE194092, Trần Trọng Nghĩa HE190336 and Hồ Minh Hiếu HE194091, supervisor Trần Văn Hà. |
| 3 | Submission date and acknowledgments are deferred. |
| 4 | Primary users are ordinary citizens. Secondary users are healthcare workers and healthcare students. Actual user testing remains unanswered under question 34. |
| 5 | The question-category taxonomy belongs to the older testdata dataset and is outdated. The older dataset is retained only to test unanswerable queries. The main benchmark is separate. |
| 6 | All included source PDFs came from dichvucong.dav.gov.vn. |
| 7 | Only usage instructions were selected. Some PDFs contain authorization-decision or regulatory-guidance information within them. |
| 8 | Collected September 18, 2026. A GET request with an increased limit retrieved all 123 vaccine listings. Exclusions were 44 with no documents, 1 malformed download URL, 2 empty folders and 1 with only a label. The remaining corpus has 75 PDFs. |
| 9 | Conversion used the Gemini website and its default model. No exact model version was supplied. |
| 10 | Work was divided into three parts. PDFs were trimmed to necessary information, removing label material and other unnecessary content, and uploaded individually. Gemini converted table images to HTML, which was inserted into the Markdown. |
| 11 | Files were manually reviewed afterward. |
| 12 | Current corpus is intended for the report. Its 75 document IDs correspond to 75 PDF files. Current Parquet inspection found 1,744 chunks, 490 with a table flag. |
| 13 | The current Qdrant store is used. Its source Parquet revision needs clarification below. |
| 16 | Naive is the BM25 pipeline. |
| 17 | Advanced is the current single-query BGE-M3 dense retrieval path. Earlier versions in Git history should be excluded from the report. |
| 18 | Checkpoints are BAAI/bge-m3 and BAAI/bge-reranker-v2-m3. Team asked that settings be checked in code. Current defaults are 50 dense candidates, 50 sparse candidates, fusion constant 60, 30 fused candidates, top-k 5, reranking enabled, and equal dense/sparse fusion weights. Saved responses contain four retrieved IDs, so historical run settings still require a run record. |
| 20 | No concrete design yet for Table-aware, Graph RAG or Agent RAG. Keep clear implementation and evaluation placeholders. |

## Questions deferred because the wording was unclear

### 13. Qdrant and its source file

Original question. Which Qdrant collection or snapshot was used for the experiments? Was it built from the current Parquet file, and was its chunk membership checked?

Plain-language version. The Parquet file is the file containing all document chunks. Qdrant stores searchable copies of those chunks and their vectors. When you last loaded Qdrant, did you use the current chunks_storage/chunks.parquet with 1,744 chunks, or restore an older snapshot? If unknown, this can be checked later on the machine running Qdrant.

### 14. Chunk size and overlap

Original question. Did the experiments use the current chunker defaults—minimum 80, maximum 600, overlap 50, measured using the code's word-based estimate? If settings changed between variants, what were they?

Plain-language version. When the team divided Markdown into smaller pieces, did you run the code as it is now, or change the size settings? The current code aims for up to 600 estimated units per chunk and repeats about 50 units between split prose chunks. A saved command or notebook can answer this.

### 15. Grouping files before chunking

Original question. The current code combines Markdown files within a registration folder. Did the team intend that grouping, and did it match the corpus used in your runs?

Plain-language version. Was each of the 75 PDF files converted into one Markdown file, or could one PDF produce multiple Markdown files in the same folder? The code joins Markdown files from the same folder before splitting the text into chunks. The one-PDF-per-document relationship is already confirmed.

### 19. Website behavior versus evaluation behavior

Original question. Is the website's query planner and conversation memory part of the evaluated system? Were experiments single-turn, conversational, or both?

Plain-language version. In the benchmark, did each question go directly to retrieval independently, or did it pass through the website's step that rewrites questions and uses previous messages? Were follow-up conversations tested, or only standalone questions?

## Remaining unanswered questions

### 21. Answer-generation model

Which model generated answers in each experiment? Was gpt-oss-120b used throughout, and through which provider or endpoint?

### 22. Generation settings

Did all variants use the same prompt, temperature, output limit and context budget? Were any prompts or models fine-tuned?

### 23. Evidence verification

Was the implemented evidence check limited to validating cited chunk IDs, or did the team also perform claim-level verification, external fact-checking, confidence scoring or another step?

### 24. Partially answerable questions

What behavior should count as success on a partially answerable question—answer the supported portion, refuse the entire question, or something else? How was that judged?

### 25. Main benchmark and distributions

Is evaluation/eval_dataset_300_final_fixed.json the main benchmark intended for the report? Were its 210 answerable / 90 partially answerable, 150 single-hop / 90 multi-hop / 60 table-reasoning, and 180 patient-style / 120 clinical-style distributions deliberate?

### 26. DeepEval generation provenance

Was this benchmark generated with the checked-in DeepEval script and gpt-oss-120b? What generation command, endpoint and package version were used, if known? Were questions manually edited or removed afterward?

DeepEval use is already established in the discussion. Exact run provenance and edits remain pending.

### 27. Corrected evidence labels

What caused the source-label corrections affecting 279 questions? Did the correction change only identifiers, or were questions, contexts or reference answers also reviewed?

### 28. Legacy query source and model

For the separate testdata dataset, where did the original queries in scrape_dataset/query.parquet come from? Which Qwen model and generation settings did the team use?

This is historical context for unanswerable testing. The team has confirmed that the old category taxonomy is outdated.

### 29. Unanswerable evaluation artifact

Which artifact is intended for unanswerable evaluation—the 222 unanswerable records in the saved 460-record dataset, a masked output, or another export? Was answerability checked against the whole corpus or only the selected evidence?

The team called the file test_data.json. The checked-in filename is test_dataset.json. Confirm whether both names refer to the same artifact and identify the final subset used.

### 30. Legacy symptom replacement

Was symptom replacement actually used in the dataset reported by the team? What purpose did it serve, and what split/output settings produced the saved files?

Keep this deferred as historical context. Do not describe the current legacy script as the main benchmark workflow.

### 31. Human review

Has expert review happened yet? If so, how many reviewers participated, what were their qualifications, which questions did they review, and how were disagreements resolved? Is there a final reviewed export?

### 32. Completed experimental runs

Which variants have actually been run? Which run produced the saved 300 responses with four retrieved chunks each?

Table-aware, Graph RAG and Agent RAG have no concrete design yet. Retain placeholders for them.

### 33. Metrics and comparison settings

Which metrics did the team actually compute and intend to report? Identify the result files and judge model. Were all variants evaluated on identical questions and corpus versions?

### 34. Additional evaluation

Did the team measure citation support, refusal quality, end-to-end latency, cost, or user experience? The existing saved latency covers generation time only. Have actual intended users tested the system?

### 35. Verified findings

What successful examples, failures and findings has the team verified? What do you consider the project's demonstrated contribution so far?

### 36. Timeline and literature survey

Was the ODT's 14-week schedule followed? What were the actual dates and major deviations? Were VaxBot-HPV, VIRA and K-Bot part of the team's original survey, or should they be presented as background literature added during report preparation?

## Other details retained for later

- Question 3 remains deferred. Do not infer a submission date from the collection date or current date.
- Question 1 confirmed the short name. The exact full report title has not been separately approved.
- No conversion prompt, exact Gemini model version, conversion date or detailed manual-review protocol was supplied. The collection date is confirmed.
- Metadata completeness, source-page links, handling of superseded instructions and the final experiment manifests remain report placeholders.

## Results evidence update on November 20, 2026

The team's earlier answers remain recorded above. New evidence on main now supplies retrieval-only results for questions 18 and 32–33, plus Table-aware and LightRAG implementation artifacts relevant to question 20. This is a repository observation and does not replace the team's confirmation of final designs or completed answer-level evaluation.

- The three primary baseline reports each score 300 questions at top-k five. Naive, Advanced and Hybrid Hit@5 are 0.5367, 0.5900 and 0.7433.
- Preliminary M3b and M4 reports contain Table-aware and graph-augmentation retrieval results. Final design mapping and implementation records remain pending. Agent RAG remains pending.
- The default five-result depth is now recorded for these IR runs. The older four-result generation artifact remains separate.
- Question 23 needs a revision-specific answer because current main has response validation enabled.
- See results/report_discrepancies.md for scoring caveats, run differences and remaining evidence needs. Questions 21–24, 26–31 and 34–36 remain unresolved to the extent not answered by these saved retrieval artifacts.

## Branch reconciliation on November 20, 2026

The report branch integrates main revision 7798195 as its implementation and results base. This resolves the previous branch-level code drift. The implementation audit in the draft refers to that revision. It does not establish which historical revision produced each saved run. Confirmed team answers and deferred questions above remain applicable.
