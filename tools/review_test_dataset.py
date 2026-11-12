"""Audit the golden test dataset parts with an LLM reached over HTTP.

The review endpoint is configured through three environment variables:

``TESTDATA_API_KEY``
    Credential sent to the endpoint.
``TESTDATA_BASE_URL``
    Base URL of the OpenAI-compatible API.
``TESTDATA_MODEL``
    Name of the model to audit with.

Every record is audited on its own: the record is rendered into a prompt that
carries the query, the answerability label, the evidence chunks, the
reference answer and the claim/evidence map, and the model answers with a
structured verdict.

The audit combines two kinds of checks:

* **Deterministic checks** run locally: required fields present, answerability
  label known, every ``[CHUNK_ID]`` citation resolving to a relevant chunk,
  every ``evidence_quote`` present verbatim in the chunk it cites.
* **Semantic checks** run in the model: the answerability label matches the
  evidence, the reference answer is grounded only in the given chunks, the
  query really is of the declared query type, and the query is well formed.

Output: one ``review_report_<part>.json`` per part plus a consolidated
``review_summary.json``. Every finished record is appended to a JSONL
checkpoint so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "test_data_review"

ANSWERABILITY_VALUES: frozenset[str] = frozenset(
    {"ANSWERABLE", "PARTIALLY_ANSWERABLE", "UNANSWERABLE"}
)

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_\-]+)\]")
WHITESPACE_PATTERN = re.compile(r"\s+")

SEMANTIC_CHECKS: tuple[tuple[str, str], ...] = (
    (
        "answerability_match",
        "ANSWERABLE when the chunks fully answer the query, PARTIALLY_ANSWERABLE "
        "when they cover only part of it, UNANSWERABLE when they do not answer it.",
    ),
    (
        "answer_grounded",
        "Every factual statement in REFERENCE_ANSWER is supported by the chunks.",
    ),
    (
        "query_type_match",
        "The query really asks what QUERY_TYPE claims it asks.",
    ),
    (
        "query_wellformed",
        "The query is a single, self-contained, unambiguous Vietnamese question.",
    ),
)

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "checks": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                name: {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ok": {"type": "boolean"},
                        "detail": {"type": "string"},
                    },
                    "required": ["ok", "detail"],
                }
                for name, _ in SEMANTIC_CHECKS
            },
            "required": [name for name, _ in SEMANTIC_CHECKS],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                    "check": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["severity", "check", "detail"],
            },
        },
    },
    "required": ["checks", "issues"],
}

SYSTEM_PROMPT = """Bạn là chuyên gia kiểm định chất lượng (QA) cho bộ dữ liệu đánh giá RAG về vắc xin tiếng Việt.

Bạn nhận MỘT bản ghi gồm: câu hỏi, nhãn ANSWERABILITY, QUERY_TYPE, các đoạn văn bằng chứng (chunks), câu trả lời tham chiếu REFERENCE_ANSWER và danh sách claim_evidence_map.

Hãy đánh giá từng mục kiểm tra sau một cách nghiêm ngặt và chỉ dựa trên nội dung chunks được cung cấp:

- answerability_match: nhãn ANSWERABILITY có khớp với bằng chứng không. ANSWERABLE khi chunks trả lời đầy đủ câu hỏi; PARTIALLY_ANSWERABLE khi chunks chỉ trả lời được một phần câu hỏi; UNANSWERABLE khi chunks không trả lời được câu hỏi.
- answer_grounded: mọi khẳng định trong REFERENCE_ANSWER đều được chunks hỗ trợ, không có thông tin bịa đặt hoặc suy diễn ngoài chunks.
- query_type_match: câu hỏi thực sự thuộc loại QUERY_TYPE khai báo (ví dụ STORAGE chỉ dành cho câu hỏi về điều kiện bảo quản).
- query_wellformed: câu hỏi là một câu hỏi tiếng Việt đơn lẻ, tự đủ, không mơ hồ, có nêu rõ loại vắc xin hoặc chủ đề cụ thể; không có placeholder, không có trường rỗng, không có chỉ thị dành cho trợ lý.

Nếu một mục kiểm tra không đạt, ghi thêm một issue với severity phù hợp (critical = sai nhãn ANSWERABILITY, câu trả lời bịa đặt, hoặc trích dẫn sai; major = lệch QUERY_TYPE, câu hỏi mơ hồ; minor = vấn đề hình thức nhỏ). Nếu tất cả đều đạt thì issues là mảng rỗng.

Trả về đúng JSON theo schema được cung cấp. Không dùng Markdown, không thêm chú thích ngoài JSON."""

USER_TEMPLATE = """CÂU HỎI (QUERY): {query}

QUERY_TYPE: {query_type}
ANSWERABILITY: {answerability}

CÁC CHUNKS BẰNG CHỨNG:
{chunks}

REFERENCE_ANSWER:
{answer}

CLAIM_EVIDENCE_MAP:
{claims}

Hãy thực hiện 4 mục kiểm tra đã nêu và trả về JSON."""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def chunk_texts(record: dict[str, Any]) -> dict[str, str]:
    return {
        str(chunk.get("CHUNK_ID", "")): str(chunk.get("TEXT", ""))
        for chunk in record.get("RELEVANT_CHUNKS") or []
        if isinstance(chunk, dict)
    }


def deterministic_checks(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Checks that need no model: field presence, label vocabulary, citations."""
    checks: dict[str, dict[str, Any]] = {}
    texts = chunk_texts(record)

    answer = str(record.get("REFERENCE_ANSWER") or "").strip()
    checks["reference_answer_present"] = {
        "ok": bool(answer),
        "detail": "REFERENCE_ANSWER is non-empty."
        if answer
        else "REFERENCE_ANSWER is missing or empty.",
    }

    answerability = str(record.get("ANSWERABILITY") or "")
    checks["answerability_label_valid"] = {
        "ok": answerability in ANSWERABILITY_VALUES,
        "detail": (
            f"ANSWERABILITY={answerability!r} is a known label."
            if answerability in ANSWERABILITY_VALUES
            else f"ANSWERABILITY={answerability!r} is not one of {sorted(ANSWERABILITY_VALUES)}."
        ),
    }

    query = str(record.get("QUERY") or "").strip()
    checks["query_present"] = {
        "ok": bool(query),
        "detail": "QUERY is non-empty." if query else "QUERY is missing or empty.",
    }

    cited = set(CITATION_PATTERN.findall(answer))
    unknown_citations = sorted(cited - set(texts))
    checks["citations_resolve"] = {
        "ok": not unknown_citations,
        "detail": (
            "Every [CHUNK_ID] in REFERENCE_ANSWER resolves to a relevant chunk."
            if not unknown_citations
            else f"Unknown citation(s) in REFERENCE_ANSWER: {', '.join(unknown_citations)}."
        ),
    }

    bad_chunk_ids: list[str] = []
    empty_evidence: list[str] = []
    missing_quotes: list[str] = []
    for index, claim in enumerate(record.get("CLAIM_EVIDENCE_MAP") or []):
        if not isinstance(claim, dict):
            continue
        chunk_id = str(claim.get("chunk_id", "")).strip()
        quote = normalize_whitespace(str(claim.get("evidence_quote", "")))
        if not chunk_id and not quote:
            empty_evidence.append(f"#{index + 1}")
            continue
        if chunk_id not in texts:
            bad_chunk_ids.append(chunk_id or "<empty>")
            continue
        if quote and quote not in normalize_whitespace(texts[chunk_id]):
            missing_quotes.append(chunk_id)
    checks["claim_evidence_present"] = {
        "ok": not empty_evidence,
        "detail": (
            "Every claim_evidence_map entry carries a chunk id and an evidence quote."
            if not empty_evidence
            else f"claim_evidence_map entr(y/ies) {', '.join(empty_evidence)} have no chunk id and no evidence quote."
        ),
    }
    checks["claim_chunk_ids_valid"] = {
        "ok": not bad_chunk_ids,
        "detail": (
            "Every claim_evidence_map entry cites a relevant chunk."
            if not bad_chunk_ids
            else f"claim_evidence_map cites unknown chunk(s): {', '.join(bad_chunk_ids)}."
        ),
    }
    checks["claim_quotes_verbatim"] = {
        "ok": not missing_quotes,
        "detail": (
            "Every evidence_quote appears verbatim in its cited chunk."
            if not missing_quotes
            else f"evidence_quote not found verbatim in chunk(s): {', '.join(missing_quotes)}."
        ),
    }
    return checks


def record_payload(record: dict[str, Any]) -> str:
    chunks = "\n\n".join(
        f"CHUNK_ID: {chunk.get('CHUNK_ID')}\nNỘI DUNG:\n{chunk.get('TEXT')}"
        for chunk in record.get("RELEVANT_CHUNKS") or []
        if isinstance(chunk, dict)
    )
    claims = json.dumps(record.get("CLAIM_EVIDENCE_MAP") or [], ensure_ascii=False, indent=2)
    return USER_TEMPLATE.format(
        query=record.get("QUERY", ""),
        query_type=record.get("QUERY_TYPE", ""),
        answerability=record.get("ANSWERABILITY", ""),
        chunks=chunks or "(không có chunk nào)",
        answer=record.get("REFERENCE_ANSWER", ""),
        claims=claims,
    )


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```$", "", text).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("model response is not a JSON object")
    return data


class Reviewer:
    """Calls the review model once per record, with a bounded retry."""

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float, retries: int) -> None:
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def review_record(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = record_payload(record)
        last_error = ""
        for _ in range(self.retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": payload},
                    ],
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "review", "strict": True, "schema": REVIEW_SCHEMA},
                    },
                )
                data = parse_json_content(response.choices[0].message.content or "")
                return assemble_result(record, data, None)
            except Exception as exc:  # noqa: BLE001 - reported per record, never fatal
                last_error = f"{type(exc).__name__}: {exc}"
        return assemble_result(record, {}, last_error)


def assemble_result(
    record: dict[str, Any], data: dict[str, Any], error: str | None
) -> dict[str, Any]:
    deterministic = deterministic_checks(record)
    semantic_raw = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    semantic: dict[str, dict[str, Any]] = {}
    for name, description in SEMANTIC_CHECKS:
        entry = semantic_raw.get(name) if isinstance(semantic_raw, dict) else None
        if isinstance(entry, dict):
            semantic[name] = {
                "ok": bool(entry.get("ok")),
                "detail": str(entry.get("detail", "")).strip() or description,
            }
        else:
            semantic[name] = {"ok": False, "detail": "Model did not report this check."}

    issues = [issue for issue in (data.get("issues") or []) if isinstance(issue, dict)]
    failed = [name for name, check in {**deterministic, **semantic}.items() if not check["ok"]]
    return {
        "QUERY_ID": record.get("QUERY_ID"),
        "error": error,
        "deterministic": deterministic,
        "semantic": semantic,
        "issues": issues,
        "failed_checks": failed,
        "verdict": "FAIL" if failed else "PASS",
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[f"{row.get('part')}::{row.get('QUERY_ID')}"] = row
    return done


def review_part(
    part_path: Path,
    records: list[dict[str, Any]],
    reviewer: Reviewer,
    checkpoint: Path,
    checkpoint_lock: threading.Lock,
    done: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    part_name = part_path.name
    rows: list[dict[str, Any]] = []
    for record in records:
        key = f"{part_name}::{record.get('QUERY_ID')}"
        if key in done:
            rows.append(done[key])
            continue
        row = reviewer.review_record(record)
        row["part"] = part_name
        done[key] = row
        with checkpoint_lock:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
    return rows


def cmd_review(args: argparse.Namespace) -> int:
    api_key = os.getenv("TESTDATA_API_KEY", "").strip()
    base_url = os.getenv("TESTDATA_BASE_URL", "").strip()
    model = os.getenv("TESTDATA_MODEL", "").strip()
    missing = [
        name
        for name, value in (
            ("TESTDATA_API_KEY", api_key),
            ("TESTDATA_BASE_URL", base_url),
            ("TESTDATA_MODEL", model),
        )
        if not value
    ]
    if missing:
        print(f"[ERROR] Missing environment variable(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    parts = sorted(out_dir.glob("*.part*.json"))
    if not parts:
        print(f"[ERROR] No part files in {out_dir}; run split_test_dataset.py first.", file=sys.stderr)
        return 1
    if args.limit:
        parts = parts[: args.limit]

    checkpoint = out_dir / "review_checkpoint.jsonl"
    done = load_checkpoint(checkpoint)
    if done:
        print(f"Resuming with {len(done)} record(s) already reviewed.")

    reviewer = Reviewer(model, base_url, api_key, args.timeout, args.retries)
    checkpoint_lock = threading.Lock()
    summary: dict[str, Any] = {"model": model, "base_url": base_url, "parts": []}

    for part_path in parts:
        records = read_json(part_path)
        print(f"Reviewing {part_path.name} ({len(records)} records) ...")
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # One task per record keeps a single slow call from blocking the rest.
            tasks = []
            for record in records:
                tasks.append(
                    pool.submit(
                        _review_one,
                        part_path.name,
                        record,
                        reviewer,
                        checkpoint,
                        checkpoint_lock,
                        done,
                    )
                )
            for task in tasks:
                rows.append(task.result())
        rows.sort(key=lambda row: str(row.get("QUERY_ID")))
        report = {
            "part": part_path.name,
            "records": len(rows),
            "passed": sum(1 for row in rows if row["verdict"] == "PASS"),
            "failed": sum(1 for row in rows if row["verdict"] == "FAIL"),
            "errors": sum(1 for row in rows if row.get("error")),
            "results": rows,
        }
        write_json(out_dir / f"review_report_{part_path.stem}.json", report)
        summary["parts"].append(
            {key: report[key] for key in ("part", "records", "passed", "failed", "errors")}
        )
        print(f"  passed={report['passed']} failed={report['failed']} errors={report['errors']}")

    write_json(out_dir / "review_summary.json", summary)
    print(f"Summary written to {out_dir / 'review_summary.json'}")
    return 0


def _review_one(
    part_name: str,
    record: dict[str, Any],
    reviewer: Reviewer,
    checkpoint: Path,
    checkpoint_lock: threading.Lock,
    done: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = f"{part_name}::{record.get('QUERY_ID')}"
    if key in done:
        return done[key]
    row = reviewer.review_record(record)
    row["part"] = part_name
    done[key] = row
    with checkpoint_lock:
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Only review the first N part files")
    args = parser.parse_args(argv)
    return cmd_review(args)


if __name__ == "__main__":
    raise SystemExit(main())
