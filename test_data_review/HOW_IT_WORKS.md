Here is exactly what gets reviewed — 10 checks per record, split into two groups.

A. Deterministic checks (local Python, no LLM, no API cost)

These run in-process on every record regardless of model outcome:

┌─────┬───────────────────────────┬───────────────────────────────────────────┐
│  #  │           Check           │                Passes when                │
├─────┼───────────────────────────┼───────────────────────────────────────────┤
│ 1   │ reference_answer_present  │ REFERENCE_ANSWER is a non-empty string    │
├─────┼───────────────────────────┼───────────────────────────────────────────┤
│ 2   │ answerability_label_valid │ ANSWERABILITY ∈ {ANSWERABLE,              │
│     │                           │ PARTIALLY_ANSWERABLE, UNANSWERABLE}       │
├─────┼───────────────────────────┼───────────────────────────────────────────┤
│ 3   │ query_present             │ QUERY is a non-empty string               │
├─────┼───────────────────────────┼───────────────────────────────────────────┤
│     │                           │ every [CHUNK_ID] token inside             │
│ 4   │ citations_resolve         │ REFERENCE_ANSWER names one of the         │
│     │                           │ record's RELEVANT_CHUNKS                  │
├─────┼───────────────────────────┼───────────────────────────────────────────┤
│ 5   │ claim_evidence_present    │ every CLAIM_EVIDENCE_MAP entry has a      │
│     │                           │ non-empty chunk_id and evidence_quote     │
├─────┼───────────────────────────┼───────────────────────────────────────────┤
│ 6   │ claim_chunk_ids_valid     │ every claim's chunk_id exists in          │
│     │                           │ RELEVANT_CHUNKS                           │
├─────┼───────────────────────────┼───────────────────────────────────────────┤
│     │                           │ every evidence_quote appears literally in │
│ 7   │ claim_quotes_verbatim     │  the chunk it cites                       │
│     │                           │ (whitespace-normalized comparison)        │
└─────┴───────────────────────────┴───────────────────────────────────────────┘

B. Semantic checks (the LLM, one call per record)

The model receives the query, ANSWERABILITY, QUERY_TYPE, all RELEVANT_CHUNKS with their text, REFERENCE_ANSWER, and the claim/evidence map, then answers in this plain-text format:

answerability_match: PASS
answer_grounded: PASS | FAIL - <giải thích>
query_type_match: PASS |
query_wellformed: PASS | FAIL - <giải thích>

ISSUE <critical|major|minor> <check>: <mô tả lỗi>

#: 8
Check: answerability_matc
Question it answers: Does the label match the evidence? ANSWERABLE = chunks fully

answer; PARTIALLY_ANSWERABLE = chunks cover only part; UNANSWERABLE = chunks
don't answer it
────────────────────────────────────────
#: 9
Check: answer_grounded
Question it answers: Is eERENCE_ANSWER supported bythe chunks — no invention, no extrapolation from outside them?
─────────────────────────#: 10
Check: query_type_matchQuestion it answers: Does the query really ask what QUERY_TYPE claims (e.g.
STORAGE only for storage────────────────────────────────────────
#: —Check: query_wellformed
Question it answers: Is tined, unambiguousVietnamese
question naming a concretder, no empty field, no
instruction aimed at the assistant?

Severity guidance given t ANSWERABILITY label,hallucinated answer, or wQUERY_TYPE, ambiguousquery; minor = small form

How a verdict is derivedA record is FAIL if any owise PASS. failed_checkslists exactly which ones.the API call itself failed
(timeout etc.) — those roresume retries them.

Two things worth knowing
Check 7 is strict by df records. It demands thequote be a literal sube reviewer's analysisconfirmed those aren'ts appear in no chunk of the record at all, so theys where the generator
   wrote a summary insteat these as real findings,
   not noise.
2. Two limits of the curray not care to close:
   - The reference answerclinical correctness — only
     that it's grounded ing but chunk-supported
     answer passes.
   - The chunks themselvee source corpus. ARELEVANT_CHUNKS entrfrom the vaccine's document would pass everythines it.