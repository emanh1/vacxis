"""Merge split table chunks, classify table kind, and convert tables to entity-rich prose."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal

_PREFIX_LINE_RE = re.compile(r"^\[CHỦ ĐỀ:[^\]]*\]\s*>\s*\[MỤC:[^\]]*\]\s*\n?", re.MULTILINE)
_HTML_TABLE_OPEN_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_HTML_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_HTML_THEAD_RE = re.compile(r"<thead\b[^>]*>.*?</thead\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TFOOT_RE = re.compile(r"<tfoot\b[^>]*>.*?</tfoot\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TD_RE = re.compile(r"<td\b[^>]*>(.*?)</td\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_STRIP_RE = re.compile(r"<[^>]+>")

_STATISTICAL_HEADER_KEYWORDS = (
    "gmt",
    "gmc",
    "opa",
    "rsba",
    "hsba",
    "hai",
    "igg",
    "ci 95%",
    "khoảng tin cậy",
    "hiệu giá",
    "nồng độ kháng thể",
    "tỷ lệ đáp ứng",
    "tỷ suất đáp ứng",
    "hiệu lực",
    "tính sinh miễn dịch",
    "đáp ứng miễn dịch",
    "đáp ứng kháng thể",
    "kháng thể diệt khuẩn",
    "kháng thể kháng khuẩn",
    "huyết thanh bảo vệ",
    "chuyển đổi huyết thanh",
    "placebo",
    "đoàn hệ",
    "mtvc",
    "atp",
    "đề cương",
)

_CLINICAL_HEADER_KEYWORDS = (
    "thành phần",
    "hoạt chất",
    "liều dùng",
    "liều lượng",
    "lịch tiêm",
    "phác đồ",
    "phản ứng có hại",
    "phản ứng bất lợi",
    "phản ứng dự kiến",
    "tác dụng không mong muốn",
    "tác dụng phụ",
    "biến cố",
    "chống chỉ định",
    "chỉ định",
    "bảo quản",
    "hạn dùng",
    "đóng gói",
    "quy cách",
    "vvm",
    "chỉ thị nhiệt",
    "tương tác",
    "thận trọng",
    "cảnh báo",
)

_CLINICAL_CATEGORIES = {
    "thành phần",
    "liều lượng và cách dùng",
    "tác dụng không mong muốn",
    "chỉ định",
    "chống chỉ định",
    "bảo quản",
    "quy cách đóng gói",
    "tương tác thuốc",
    "cảnh báo và thận trọng",
    "thai kỳ và cho con bú",
    "hướng dẫn sử dụng",
    "tương kỵ",
}

_STATISTICAL_CATEGORIES = {
    "dược lực học",
    "bảng số liệu",
    "đặc tính dược lý",
    "dược động học",
}

_NUMERIC_STAT_CELL_RE = re.compile(
    r"^[<>≤≥±~≈*]?\s*\d+(?:[.,]\d+)?(?:\s*%|\s*\([^)]*\)|\s*[-–/]\s*\d+(?:[.,]\d+)?%?)?(?:\s*[a-zA-Z†‡§*]+)?$"
)


def _strip_prefix(text: str) -> tuple[str, str]:
    """Split optional '[CHỦ ĐỀ: ...] > [MỤC: ...]' prefix from body text."""
    stripped = text.strip()
    match = _PREFIX_LINE_RE.match(stripped)
    if not match:
        return "", stripped
    return match.group(0).strip(), stripped[match.end():].strip()


def _dedup_boundary_rows(existing: list[str], incoming: list[str]) -> list[str]:
    """Append incoming rows while removing any suffix-prefix overlap from chunk splitting."""
    if not existing:
        return list(incoming)
    max_k = min(len(existing), len(incoming))
    overlap_k = 0
    for k in range(max_k, 0, -1):
        if existing[-k:] == incoming[:k]:
            overlap_k = k
            break
    return existing + incoming[overlap_k:]


def merge_html_table_fragments(texts: list[str]) -> str:
    """Merge consecutive split HTML table chunks into one deduplicated HTML table."""
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    prefix = ""
    before_parts: list[str] = []
    after_parts: list[str] = []
    open_tag = "<table>"
    thead_html = ""
    tfoot_html = ""
    merged_rows: list[str] = []

    for idx, raw_text in enumerate(texts):
        frag_prefix, body = _strip_prefix(raw_text)
        if idx == 0 and frag_prefix:
            prefix = frag_prefix

        open_m = _HTML_TABLE_OPEN_RE.search(body)
        close_matches = list(_HTML_TABLE_CLOSE_RE.finditer(body))
        if not open_m or not close_matches:
            if body:
                after_parts.append(body)
            continue

        close_m = close_matches[-1]
        if idx == 0:
            open_tag = open_m.group(0)
            before = body[:open_m.start()].strip()
            if before:
                before_parts.append(before)

        after = body[close_m.end():].strip()
        if after and after not in after_parts:
            after_parts.append(after)

        inner = body[open_m.end():close_m.start()]
        thead_m = _HTML_THEAD_RE.search(inner)
        if thead_m:
            if not thead_html:
                thead_html = thead_m.group(0).strip()
            inner = inner[:thead_m.start()] + inner[thead_m.end():]

        tfoot_m = _HTML_TFOOT_RE.search(inner)
        if tfoot_m:
            tfoot_html = tfoot_m.group(0).strip()
            inner = inner[:tfoot_m.start()] + inner[tfoot_m.end():]

        rows = [m.group(0).strip() for m in _HTML_TR_RE.finditer(inner) if m.group(0).strip()]
        merged_rows = _dedup_boundary_rows(merged_rows, rows)

    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    if before_parts:
        parts.append("\n".join(before_parts))

    table_parts = [open_tag]
    if thead_html:
        table_parts.append(thead_html)
    if merged_rows:
        table_parts.append("<tbody>")
        table_parts.extend(merged_rows)
        table_parts.append("</tbody>")
    if tfoot_html:
        table_parts.append(tfoot_html)
    table_parts.append("</table>")
    parts.append("\n".join(table_parts))

    if after_parts:
        parts.append("\n".join(after_parts))
    return "\n".join(parts)


def merge_markdown_table_fragments(texts: list[str]) -> str:
    """Merge consecutive split Markdown pipe table chunks into one deduplicated table."""
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    prefix = ""
    header_lines: list[str] = []
    merged_rows: list[str] = []

    for idx, raw_text in enumerate(texts):
        frag_prefix, body = _strip_prefix(raw_text)
        if idx == 0 and frag_prefix:
            prefix = frag_prefix

        lines = [line for line in body.splitlines() if line.strip()]
        if len(lines) >= 2 and lines[0].lstrip().startswith("|") and set(lines[1].strip()) <= set("|-: "):
            if not header_lines:
                header_lines = lines[:2]
            data_lines = lines[2:]
        else:
            data_lines = lines
        merged_rows = _dedup_boundary_rows(merged_rows, data_lines)

    body_out = "\n".join([*header_lines, *merged_rows])
    return f"{prefix}\n{body_out}" if prefix else body_out


def merge_table_fragments(texts: list[str]) -> str:
    """Merge consecutive split table chunks (HTML or Markdown) into a single table."""
    if any(_HTML_TABLE_OPEN_RE.search(t) for t in texts):
        return merge_html_table_fragments(texts)
    if any("|" in t for t in texts):
        return merge_markdown_table_fragments(texts)
    return "\n\n".join(texts)


def group_and_merge_document_chunks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group consecutive split table chunks sharing (doc_id, header) into one merged record."""
    if not records:
        return []

    def _has_table(rec: dict[str, Any]) -> bool:
        meta = rec.get("metadata")
        if isinstance(meta, dict) and "has_table" in meta:
            return bool(meta["has_table"])
        return bool(rec.get("has_table", False))

    merged_records: list[dict[str, Any]] = []
    idx = 0
    while idx < len(records):
        current = dict(records[idx])
        if not _has_table(current):
            merged_records.append(current)
            idx += 1
            continue

        group = [current]
        j = idx + 1
        while j < len(records):
            nxt = records[j]
            if (
                _has_table(nxt)
                and nxt.get("doc_id") == current.get("doc_id")
                and nxt.get("header") == current.get("header")
            ):
                group.append(dict(nxt))
                j += 1
            else:
                break

        if len(group) > 1:
            current["text"] = merge_table_fragments([g["text"] for g in group])
            current["merged_chunk_ids"] = [g.get("chunk_id") for g in group]
        merged_records.append(current)
        idx = j

    return merged_records


def _compute_numeric_cell_ratio(content: str) -> float:
    """Compute ratio of purely numeric/statistical cells among non-empty table cells."""
    raw_cells = _HTML_TD_RE.findall(content)
    if not raw_cells:
        # Markdown pipe cells fallback
        pipe_cells: list[str] = []
        for line in content.splitlines():
            s = line.strip()
            if s.startswith("|") and not set(s) <= set("|-: "):
                pipe_cells.extend(c.strip() for c in s.strip("|").split("|"))
        raw_cells = pipe_cells

    non_empty = 0
    numeric_stat = 0
    for cell in raw_cells:
        plain = " ".join(_HTML_TAG_STRIP_RE.sub(" ", cell).split()).strip()
        if not plain or set(plain) <= set("-–— ") or plain in ("N/A", "n.a."):
            continue
        non_empty += 1
        if _NUMERIC_STAT_CELL_RE.match(plain):
            numeric_stat += 1

    return numeric_stat / non_empty if non_empty else 0.0


def classify_table_kind(
    header: str,
    content: str,
    category: str = "",
) -> Literal["clinical", "statistical"]:
    """Classify a table into 'clinical' (row-by-row prose) or 'statistical' (overview summary)."""
    header_lower = header.lower()
    cat_lower = category.lower().strip()

    has_clinical_kw = any(kw in header_lower for kw in _CLINICAL_HEADER_KEYWORDS)
    has_stat_kw = any(kw in header_lower for kw in _STATISTICAL_HEADER_KEYWORDS)

    # Strong clinical headers (e.g. adverse reactions, dosage, composition, storage) take priority
    if has_clinical_kw and not has_stat_kw:
        return "clinical"

    if has_stat_kw and not has_clinical_kw:
        return "statistical"

    # Check thead or leading table body snippet when header itself is generic
    thead_m = _HTML_THEAD_RE.search(content)
    if thead_m:
        thead_plain = " ".join(_HTML_TAG_STRIP_RE.sub(" ", thead_m.group(0)).split()).lower()
        thead_clinical = any(kw in thead_plain for kw in _CLINICAL_HEADER_KEYWORDS)
        thead_stat = any(kw in thead_plain for kw in _STATISTICAL_HEADER_KEYWORDS)
        if thead_clinical and not thead_stat:
            return "clinical"
        if thead_stat and not thead_clinical:
            return "statistical"
    else:
        lead_plain = " ".join(_HTML_TAG_STRIP_RE.sub(" ", content[:500]).split()).lower()
        if any(kw in lead_plain for kw in ("thành phần hoạt chất", "thành phần tá dược")):
            return "clinical"

    numeric_ratio = _compute_numeric_cell_ratio(content)
    if numeric_ratio >= 0.45:
        return "statistical"

    if cat_lower in _CLINICAL_CATEGORIES:
        return "clinical"
    if cat_lower in _STATISTICAL_CATEGORIES:
        return "statistical"

    return "statistical" if numeric_ratio >= 0.35 else "clinical"


def build_table_summary_prompt(
    doc_title: str,
    header: str,
    table_body: str,
    table_kind: Literal["clinical", "statistical"],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for converting a table into entity-rich prose."""
    if table_kind == "clinical":
        system_prompt = (
            "Bạn là chuyên gia y khoa vắc xin. Nhiệm vụ của bạn là chuyển đổi bảng dữ liệu "
            "hướng dẫn lâm sàng (thành phần, liều dùng, lịch tiêm, tác dụng không mong muốn, "
            "chống chỉ định, bảo quản) sang các câu văn xuôi tiếng Việt hoàn chỉnh, rõ ràng "
            "Chủ ngữ - Vị ngữ - Tân ngữ để phục vụ trích xuất thực thể vào Knowledge Graph.\n"
            "Quy tắc bắt buộc:\n"
            "1. Bảo toàn ĐẦY ĐỦ tất cả các hàng, cột, đối tượng, liều lượng, tần suất và chú thích chân bảng.\n"
            "2. Luôn nêu rõ tên vắc xin trong câu để không mất ngữ cảnh.\n"
            "3. Không dùng thẻ HTML hay bảng Markdown trong đầu ra; chỉ viết văn xuôi mạch lạc."
        )
    else:
        system_prompt = (
            "Bạn là chuyên gia y khoa vắc xin. Nhiệm vụ của bạn là tóm tắt TỔNG QUAN bảng số liệu "
            "thử nghiệm lâm sàng / tính sinh miễn dịch sang 1 đoạn văn xuôi tiếng Việt súc tích "
            "để phục vụ trích xuất thực thể vào Knowledge Graph.\n"
            "Quy tắc bắt buộc:\n"
            "1. Nêu rõ tên vắc xin nghiên cứu, vắc xin đối chứng (nếu có), nhóm đối tượng/độ tuổi, "
            "các mầm bệnh/týp huyết thanh được đánh giá, chỉ số đo lường (GMT, GMC, rSBA, hSBA, hiệu lực bảo vệ) "
            "và kết luận lâm sàng chính.\n"
            "2. KHÔNG liệt kê từng con số thống kê rời rạc của từng hàng/cột (như từng giá trị n, CI 95% của mọi týp).\n"
            "3. Không dùng thẻ HTML hay bảng Markdown trong đầu ra; chỉ viết văn xuôi mạch lạc."
        )

    user_prompt = (
        f"Vắc xin / Tài liệu: {doc_title}\n"
        f"Tiêu đề mục/bảng: {header}\n\n"
        f"Nội dung bảng:\n{table_body}"
    )
    return system_prompt, user_prompt


def load_table_summary_cache(cache_path: Path) -> dict[str, str]:
    """Load persisted SHA-256 -> prose summary cache from disk."""
    if not cache_path.is_file():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_table_summary_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """Persist SHA-256 -> prose summary cache atomically to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(cache_path)


def make_table_cache_key(
    doc_id: str,
    header: str,
    table_kind: str,
    merged_text: str,
    model_name: str,
) -> str:
    """Compute deterministic SHA-256 cache key for a merged table."""
    payload = f"{doc_id}\n{header}\n{table_kind}\n{model_name}\n{merged_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def summarize_table_to_entity_prose(
    record: dict[str, Any],
    llm_func: Callable[..., Awaitable[str]],
    cache: dict[str, str],
    model_name: str,
) -> str:
    """Convert a merged table record into entity-rich prose prefixed with topic/section header."""
    raw_text = record["text"]
    prefix, table_body = _strip_prefix(raw_text)
    doc_title = record.get("doc_title") or record.get("doc_id") or "Vắc xin"
    header = record.get("header") or "Bảng dữ liệu"
    category = record.get("category") or ""

    table_kind = classify_table_kind(header=header, content=table_body, category=category)
    cache_key = make_table_cache_key(
        doc_id=str(record.get("doc_id", "")),
        header=header,
        table_kind=table_kind,
        merged_text=raw_text,
        model_name=model_name,
    )

    if cache_key in cache and cache[cache_key].strip():
        prose = cache[cache_key].strip()
    else:
        system_prompt, user_prompt = build_table_summary_prompt(
            doc_title=doc_title,
            header=header,
            table_body=table_body,
            table_kind=table_kind,
        )
        prose = (await llm_func(prompt=user_prompt, system_prompt=system_prompt)).strip()
        if prose:
            cache[cache_key] = prose

    effective_prefix = prefix or f"[CHỦ ĐỀ: {doc_title}] > [MỤC: {header}]"
    return f"{effective_prefix}\n{prose}"
