"""Semantic chunking: merge short sections, split long ones, preserve tables."""
from bisect import bisect_right
from collections.abc import Callable, Iterator
import re

from .schemas import classify_category, detect_content_type

DEFAULT_CHUNK_OVERLAP_TOKENS = 50

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?/?>")
_HTML_TABLE_OPEN_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_HTML_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_HTML_THEAD_RE = re.compile(r"<thead\b[^>]*>.*?</thead\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TFOOT_RE = re.compile(r"<tfoot\b[^>]*>.*?</tfoot\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TH_RE = re.compile(r"<th\b", re.IGNORECASE)
_HTML_CELL_RE = re.compile(r"<(td|th)\b([^>]*)>(.*?)</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_FOOTNOTE_P_RE = re.compile(
    r"^<p\b[^>]*>\s*(?:<(?:sup|em|i)\b|\*|[a-z]\)|Ghi\s*chú|Chú\s*thích)",
    re.IGNORECASE,
)
_HTML_DIV_ONLY_RE = re.compile(r"^\s*</?div\b[^>]*>\s*$", re.IGNORECASE)


def _count_tokens(text: str) -> int:
    """Word-based token count with HTML tag overhead for tokenizer alignment."""
    if "<" not in text:
        return len(text.split())
    tags = _HTML_TAG_RE.findall(text)
    if not tags:
        return len(text.split())
    visible_words = len(_HTML_TAG_RE.sub(" ", text).split())
    return visible_words + len(tags) * 3


def _contains_table_markup(content: str) -> bool:
    """Check if content contains Markdown pipe or HTML table markup."""
    return "|" in content or bool(_HTML_TABLE_OPEN_RE.search(content))


def create_medical_chunks(
    sections: list[dict],
    min_tokens: int = 80,
    max_tokens: int = 600,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    *,
    embedding_prefix: Callable[[str], str] | None = None,
) -> list[dict]:
    """
    Merge short sections (< min_tokens) into the next section,
    split sections exceeding max_tokens, and preserve standalone tables.

    Limits and overlap use the same word-based estimate. When provided,
    embedding_prefix returns the prefix for the emitted chunk's header.

    Each output chunk contains:
        header, parent_header, content, category, content_type, has_table
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must be nonnegative")

    chunks: list[dict] = []
    buffer_sections: list[dict] = []
    buffer_tokens = 0

    def content_budget(header: str) -> int:
        prefix_tokens = _count_tokens(embedding_prefix(header)) if embedding_prefix else 0
        return max_tokens - prefix_tokens

    def merged_header(sections: list[dict]) -> str:
        return " | ".join(dict.fromkeys(s["header"] for s in sections))

    def flush_buffer():
        nonlocal buffer_sections, buffer_tokens
        if not buffer_sections:
            return

        # Pick dominant section by token count
        dominant = max(buffer_sections, key=lambda s: _count_tokens(s["content"]))

        # Check for specific medical category among all buffered sections
        chosen_cat = "thông tin chung"
        for s in sorted(buffer_sections, key=lambda x: _count_tokens(x["content"]), reverse=True):
            cat = classify_category(s["header"], s.get("parent_header", ""))
            if cat != "thông tin chung":
                chosen_cat = cat
                break

        combined_content = "\n\n".join(s["content"] for s in buffer_sections)
        combined_header = merged_header(buffer_sections)
        has_table = any(s.get("has_table", False) for s in buffer_sections)

        chunks.append({
            "header": dominant["header"] if len(buffer_sections) == 1 else combined_header,
            "parent_header": dominant.get("parent_header", ""),
            "content": combined_content,
            "category": chosen_cat,
            "content_type": detect_content_type(combined_content, has_table),
            "has_table": has_table,
        })

        buffer_sections = []
        buffer_tokens = 0

    for section in sections:
        sec_tokens = _count_tokens(section["content"])
        if not sec_tokens:
            continue
        budget = content_budget(section["header"])
        if budget < 1:
            raise ValueError(f"Embedding prefix leaves no content budget for {section['header']!r}")
        blocks = (
            _split_section_into_blocks(section["content"])
            if _contains_table_markup(section["content"])
            else []
        )
        has_table = section.get("has_table", False) or any(kind == "table" for kind, _ in blocks)

        # Tables: if section exceeds max_tokens, split it while preserving table blocks;
        # otherwise keep intact as a standalone chunk.
        if has_table:
            flush_buffer()
            if sec_tokens > budget:
                _split_long_section(section, blocks, budget, chunk_overlap_tokens, chunks)
            else:
                chunks.append({
                    "header": section["header"],
                    "parent_header": section.get("parent_header", ""),
                    "content": section["content"],
                    "category": classify_category(section["header"], section.get("parent_header", "")),
                    "content_type": (
                        "table" if len(blocks) == 1 and blocks[0][0] == "table"
                        else detect_content_type(section["content"], True)
                    ),
                    "has_table": True,
                })
            continue

        if sec_tokens > budget:
            flush_buffer()
            _split_long_section(section, blocks, budget, chunk_overlap_tokens, chunks)
            continue

        if buffer_sections and buffer_tokens + sec_tokens > content_budget(
            merged_header([*buffer_sections, section])
        ):
            flush_buffer()
        buffer_sections.append(section)
        buffer_tokens += sec_tokens
        if buffer_tokens >= min_tokens:
            flush_buffer()

    flush_buffer()
    return chunks


def _table_cells(line: str) -> list[str]:
    """Split on unescaped pipes and ignore optional outer pipes."""
    line = line.strip()
    if "|" not in line:
        return []
    cells: list[str] = []
    start = 0
    escaped = False
    for index, char in enumerate(line):
        if char == "|" and not escaped:
            if index:
                cells.append(line[start:index])
            start = index + 1
        escaped = char == "\\" and not escaped
    if start and start < len(line):
        cells.append(line[start:])
    return cells


def _has_table_header(header: str, separator: str) -> bool:
    rules = _table_cells(separator)
    if not rules or len(rules) != len(_table_cells(header)):
        return False
    for rule in rules:
        dashes = rule.strip().removeprefix(":").removesuffix(":")
        if not dashes or dashes.strip("-"):
            return False
    return True


def _is_html_footnote_line(line: str) -> bool:
    """Check if a line after </table> is a short HTML footnote paragraph."""
    stripped = line.strip()
    if not stripped or _HTML_DIV_ONLY_RE.match(stripped):
        return False
    return bool(_HTML_FOOTNOTE_P_RE.match(stripped)) and _count_tokens(stripped) <= 80


def _split_section_into_blocks(content: str) -> list[tuple[str, str]]:
    """Keep HTML tables, pipe tables, and surrounding text in their original order."""
    lines = content.splitlines(keepends=True)
    blocks: list[tuple[str, str]] = []
    start = 0
    kind = "text"
    in_html_table = False
    index = 0

    def append_block(end: int, block_kind: str) -> None:
        block = "".join(lines[start:end])
        if block_kind == "text":
            block = block.strip()
        else:
            block = block.rstrip("\r\n")
        if block:
            normalized = "table" if block_kind in ("table", "html_table") else "text"
            blocks.append((normalized, block))

    while index < len(lines):
        line = lines[index]
        line_clean = line.strip()

        if in_html_table:
            if _HTML_TABLE_CLOSE_RE.search(line_clean) and not (
                _HTML_TABLE_OPEN_RE.search(line_clean)
                and list(_HTML_TABLE_OPEN_RE.finditer(line_clean))[-1].start()
                > list(_HTML_TABLE_CLOSE_RE.finditer(line_clean))[-1].start()
            ):
                in_html_table = False
                # Attach adjacent short footnote <p> lines immediately after </table>
                lookahead = index + 1
                attached_end = index + 1
                while lookahead < len(lines):
                    la_clean = lines[lookahead].strip()
                    if not la_clean or _HTML_DIV_ONLY_RE.match(la_clean):
                        lookahead += 1
                        continue
                    if _is_html_footnote_line(la_clean):
                        attached_end = lookahead + 1
                        lookahead += 1
                        continue
                    break
                append_block(attached_end, "html_table")
                index = attached_end
                start = index
                kind = "text"
                continue
            index += 1
            continue

        if _HTML_TABLE_OPEN_RE.search(line_clean):
            if start < index:
                append_block(index, kind)
            start = index
            kind = "html_table"
            last_open = list(_HTML_TABLE_OPEN_RE.finditer(line_clean))[-1].start()
            closes = list(_HTML_TABLE_CLOSE_RE.finditer(line_clean))
            last_close = closes[-1].start() if closes else -1
            if last_open > last_close:
                in_html_table = True
                index += 1
                continue
            # Single-line <table>...</table>
            lookahead = index + 1
            attached_end = index + 1
            while lookahead < len(lines):
                la_clean = lines[lookahead].strip()
                if not la_clean or _HTML_DIV_ONLY_RE.match(la_clean):
                    lookahead += 1
                    continue
                if _is_html_footnote_line(la_clean):
                    attached_end = lookahead + 1
                    lookahead += 1
                    continue
                break
            append_block(attached_end, "html_table")
            index = attached_end
            start = index
            kind = "text"
            continue

        is_md_table = line.lstrip().startswith("|") or (
            bool(_table_cells(line)) and (
                kind == "table" or (
                    index + 1 < len(lines) and _has_table_header(line, lines[index + 1])
                )
            )
        )
        line_kind = "table" if is_md_table else "text"
        if line_kind != kind:
            append_block(index, kind)
            start = index
            kind = line_kind
        index += 1

    append_block(len(lines), kind)
    return blocks


def _split_text_paragraphs(text: str, max_tokens: int, overlap: int) -> Iterator[str]:
    """Prefer paragraph or line endings, with word boundaries as the fallback."""
    if _count_tokens(text) <= max_tokens:
        yield text
        return

    starts: list[int] = []
    line_ends: list[int] = []
    paragraph_ends: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        words = line.split()
        cursor = offset
        for word in words:
            start = text.index(word, cursor)
            starts.append(start)
            cursor = start + len(word)
        if words:
            line_ends.append(len(starts))
        elif starts:
            paragraph_ends.append(len(starts))
        offset += len(line)
    starts.append(len(text))

    end = 0
    overlap = min(overlap, max_tokens - 1)
    while end < len(starts) - 1:
        start = max(0, end - overlap)
        limit = min(start + max_tokens, len(starts) - 1)
        if limit < len(starts) - 1:
            for boundaries in (paragraph_ends, line_ends):
                boundary = bisect_right(boundaries, limit) - 1
                if boundary >= 0 and boundaries[boundary] > end:
                    limit = boundaries[boundary]
                    break
        yield text[starts[start]:starts[limit]].rstrip()
        end = limit


def _split_table_row(row: str, column_count: int, max_tokens: int) -> Iterator[str]:
    """Continue an oversized row in the original column slots, in cell order."""
    cells = _table_cells(row)
    column_count = max(column_count, len(cells))
    if cells and not any(cell.strip() for cell in cells):
        yield "|" * (column_count + 1)
        return
    if not cells:
        raise ValueError("Cannot split an oversized table row without cell delimiters")
    # Under a tight budget, the header supplies the implied empty trailing cells.
    compact = max_tokens < 3
    cell_budget = max_tokens if compact else max_tokens - 2
    for column, cell in enumerate(cells):
        if not cell.strip():
            continue
        left = "|" * (column + 1)
        for part in _split_text_paragraphs(cell.strip(), cell_budget, 0):
            if compact:
                yield f"{left}{part}"
            else:
                yield f"{left} {part} {'|' * (column_count - column)}"


def _split_table(content: str, max_tokens: int, overlap: int) -> Iterator[str]:
    """Pack complete rows with a repeated header and a suffix of complete rows."""
    if _count_tokens(content) <= max_tokens:
        yield content
        return

    lines = content.splitlines()
    has_header = len(lines) > 1 and _has_table_header(lines[0], lines[1])
    header = "\n".join(lines[:2]) + "\n" if has_header else ""
    column_count = len(_table_cells(lines[0])) if has_header else 0
    row_budget = max_tokens - _count_tokens(header)
    if row_budget < 1:
        raise ValueError("max_tokens cannot fit the Markdown table header and row content")

    rows: list[tuple[str, int]] = []
    row_tokens = 0
    for row in lines[2 if has_header else 0:]:
        size = _count_tokens(row)
        if row_tokens + size > row_budget:
            if rows:
                yield header + "\n".join(line for line, _ in rows)
            keep = len(rows)
            row_tokens = 0
            # Choose the suffix closest to the target that leaves space for new content.
            while keep:
                candidate = row_tokens + rows[keep - 1][1]
                if candidate + size > row_budget or abs(overlap - candidate) >= abs(overlap - row_tokens):
                    break
                keep -= 1
                row_tokens = candidate
            rows = rows[keep:]
        if size > row_budget:
            for part in _split_table_row(row, column_count, row_budget):
                yield header + part
        else:
            rows.append((row, size))
            row_tokens += size
    if rows:
        yield header + "\n".join(line for line, _ in rows)


def _split_html_table_row(
    row: str,
    build_subtable: Callable[[list[str]], str],
    row_budget: int,
) -> Iterator[str]:
    """Split an oversized HTML <tr> across its cells while preserving valid <tr><td> markup."""
    cells = list(_HTML_CELL_RE.finditer(row))
    if not cells:
        for part in _split_text_paragraphs(row, max(1, row_budget), 0):
            yield build_subtable([f"<tr><td>{part}</td></tr>"])
        return

    for match in cells:
        tag, attrs, inner = match.group(1), match.group(2), match.group(3).strip()
        if not inner:
            continue
        cell_wrapper_tokens = _count_tokens(f"<tr><{tag}{attrs}></{tag}></tr>")
        cell_budget = max(1, row_budget - cell_wrapper_tokens)
        if _count_tokens(inner) <= cell_budget:
            yield build_subtable([f"<tr><{tag}{attrs}>{inner}</{tag}></tr>"])
        else:
            for part in _split_text_paragraphs(inner, cell_budget, 0):
                yield build_subtable([f"<tr><{tag}{attrs}>{part}</{tag}></tr>"])


def _split_html_table(content: str, max_tokens: int, overlap: int) -> Iterator[str]:
    """Pack complete HTML <tr> rows with repeated <thead> and row-level overlap."""
    if _count_tokens(content) <= max_tokens:
        yield content
        return

    open_match = _HTML_TABLE_OPEN_RE.search(content)
    close_match = None
    for m in _HTML_TABLE_CLOSE_RE.finditer(content):
        close_match = m
    if not open_match or not close_match or close_match.start() < open_match.end():
        yield from _split_text_paragraphs(content, max_tokens, overlap)
        return

    open_tag = open_match.group(0)
    inner = content[open_match.end():close_match.start()]
    trailing_notes = content[close_match.end():].strip()

    thead_match = _HTML_THEAD_RE.search(inner)
    table_header = thead_match.group(0).strip() if thead_match else ""
    if thead_match:
        inner = inner[:thead_match.start()] + inner[thead_match.end():]

    tfoot_match = _HTML_TFOOT_RE.search(inner)
    tfoot_html = tfoot_match.group(0).strip() if tfoot_match else ""
    if tfoot_match:
        inner = inner[:tfoot_match.start()] + inner[tfoot_match.end():]

    rows = [m.group(0).strip() for m in _HTML_TR_RE.finditer(inner) if m.group(0).strip()]
    if not table_header and rows and _HTML_TH_RE.search(rows[0]):
        table_header = f"<thead>\n{rows.pop(0)}\n</thead>"

    if not rows:
        yield from _split_text_paragraphs(content, max_tokens, overlap)
        return

    def build_subtable(sub_rows: list[str], include_tfoot: bool = False) -> str:
        parts = [open_tag]
        if table_header:
            parts.append(table_header)
        if sub_rows:
            parts.append("<tbody>")
            parts.extend(sub_rows)
            parts.append("</tbody>")
        if include_tfoot and tfoot_html:
            parts.append(tfoot_html)
        parts.append("</table>")
        return "\n".join(parts)

    wrapper_tokens = _count_tokens(build_subtable([""]))
    row_budget = max(1, max_tokens - wrapper_tokens)

    buf_rows: list[tuple[str, int]] = []
    row_tokens = 0
    emitted: list[str] = []

    for row in rows:
        size = _count_tokens(row)
        if buf_rows and row_tokens + size > row_budget:
            emitted.append(build_subtable([r for r, _ in buf_rows]))
            keep = len(buf_rows)
            row_tokens = 0
            while keep:
                candidate = row_tokens + buf_rows[keep - 1][1]
                if candidate + size > row_budget or abs(overlap - candidate) >= abs(overlap - row_tokens):
                    break
                keep -= 1
                row_tokens = candidate
            buf_rows = buf_rows[keep:]

        if size > row_budget:
            emitted.extend(_split_html_table_row(row, build_subtable, row_budget))
        else:
            buf_rows.append((row, size))
            row_tokens += size

    if buf_rows:
        last_with_tfoot = build_subtable([r for r, _ in buf_rows], include_tfoot=True)
        if tfoot_html and _count_tokens(last_with_tfoot) <= max_tokens:
            emitted.append(last_with_tfoot)
        else:
            emitted.append(build_subtable([r for r, _ in buf_rows]))
            if tfoot_html:
                emitted.append(build_subtable([], include_tfoot=True))
    elif tfoot_html:
        emitted.append(build_subtable([], include_tfoot=True))

    if trailing_notes:
        if emitted and _count_tokens(f"{emitted[-1]}\n{trailing_notes}") <= max_tokens:
            emitted[-1] = f"{emitted[-1]}\n{trailing_notes}"
        else:
            emitted.extend(_split_text_paragraphs(trailing_notes, max_tokens, overlap))

    yield from emitted


def _split_long_section(
    section: dict,
    blocks: list[tuple[str, str]],
    max_tokens: int,
    overlap: int,
    output: list[dict],
):
    """Split each block in order, keeping overlap inside that block."""
    sub_idx = 0
    for kind, content in blocks or [("text", section["content"])]:
        has_table = kind == "table"
        if has_table and _HTML_TABLE_OPEN_RE.search(content):
            split = _split_html_table
        elif has_table:
            split = _split_table
        else:
            split = _split_text_paragraphs
        for part in split(content, max_tokens, overlap):
            part_has_table = has_table and _contains_table_markup(part)
            _emit_sub_chunk(section, part, sub_idx, part_has_table, output)
            sub_idx += 1


def _emit_sub_chunk(section: dict, content: str, sub_idx: int, has_table: bool, output: list[dict]):
    """Emit a sub-chunk produced from splitting a long section."""
    output.append({
        "header": section["header"],
        "parent_header": section.get("parent_header", "") if sub_idx == 0 else section["header"],
        "content": content,
        "category": classify_category(section["header"], section.get("parent_header", "")),
        "content_type": "table" if has_table else "text",
        "has_table": has_table,
    })

