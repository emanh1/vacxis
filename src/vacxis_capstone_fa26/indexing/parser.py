"""Parse Markdown vaccine documents into structured sections."""
import re

# Standard medical keywords used to identify plain-text headings
MEDICAL_KEYWORDS = [
    "thành phần", "dạng bào chế", "chỉ định", "liều lượng", "liều dùng",
    "cách dùng", "chống chỉ định", "thận trọng", "cảnh báo", "tương tác",
    "tác dụng không mong muốn", "tác dụng phụ", "quá liều", "bảo quản",
    "hạn dùng", "đóng gói", "dược lực học", "dược động học", "phi lâm sàng",
    "đặc tính dược lý", "đặc tính lâm sàng", "đặc tính dược học", "nhóm trị liệu",
    "ảnh hưởng", "tương kỵ", "phụ nữ có thai", "cho con bú", "thai kỳ", "lái xe",
    "hướng dẫn sử dụng", "nhà sản xuất", "mô tả", "qui cách", "nghiên cứu lâm sàng",
]

# Common advisory phrases that should not be treated as standalone section headers
BOILERPLATE_HEADERS = [
    "đọc kỹ hướng dẫn sử dụng",
    "để xa tầm tay trẻ em",
    "thuốc bán theo đơn",
    "dùng theo đơn",
    "thông báo cho bác sỹ",
    "xin hỏi ý kiến bác sỹ",
    "lắc kỹ trước khi dùng",
]

# Regex patterns for headers and page markers
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
TABLE_HEADER_PATTERN = re.compile(r"^(\*\*)?(Bảng\s+\d+[:\.]\s*.+?)(\*\*)?$", re.IGNORECASE)
BOLD_HEADING_PATTERN = re.compile(
    r"^\*\*(?:([0-9]{1,2}(?:\.[0-9]{1,2})*\.?)\s*)?([^\*]+)\*\*:?\s*$"
)
PAGE_MARKER_PATTERN = re.compile(
    r"^(?:#{1,6}\s*(?:\[?\s*(?:PAGE|Trang)\s*\d+[^\]]*\]?)|Page\s+\d+\s+of\s+\d+|TRANG\s+\d+.*|---|\d+/\d+|\\?\[Page\s*\d+\\?\])$",
    re.IGNORECASE,
)
TABLE_LINE_PATTERN = re.compile(r"^\|.*\|")
NUMBERED_HEADER_PATTERN = re.compile(
    r"^(?:([0-9]{1,2}(?:\.[0-9]{1,2})*\.?)\s+)[A-ZÀ-Ỹ]"
)

# HTML patterns for table state, headings, captions, and wrapper divs
HTML_TABLE_OPEN_PATTERN = re.compile(r"<table\b", re.IGNORECASE)
HTML_TABLE_CLOSE_PATTERN = re.compile(r"</table\s*>", re.IGNORECASE)
HTML_HEADING_PATTERN = re.compile(r"^<h([1-6])\b[^>]*>(.*?)</h\1\s*>$", re.IGNORECASE)
HTML_P_PATTERN = re.compile(r"^<p\b[^>]*>(.*?)</p\s*>$", re.IGNORECASE)
HTML_EMPHASIS_PATTERN = re.compile(
    r"^(?:<(?:strong|b|u)\b[^>]*>\s*)+(.+?)(?:\s*</(?:strong|b|u)\s*>)+$",
    re.IGNORECASE,
)
HTML_DIV_ONLY_PATTERN = re.compile(r"^</?div\b[^>]*>$", re.IGNORECASE)
HTML_DIV_WRAPPER_STRIP_PATTERN = re.compile(r"^(?:<div\b[^>]*>\s*)+|(?:\s*</div\s*>)+$", re.IGNORECASE)
HTML_TAG_STRIP_PATTERN = re.compile(r"<[^>]+>")


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags and normalize surrounding whitespace."""
    return " ".join(HTML_TAG_STRIP_PATTERN.sub(" ", text).split())


def _is_boilerplate(text: str) -> bool:
    """Check if header text is common advisory boilerplate."""
    t_lower = text.lower().strip()
    return any(bp in t_lower for bp in BOILERPLATE_HEADERS)


def _is_all_caps_heading(line: str) -> bool:
    """Check if a line is an uppercase medical heading."""
    if len(line) > 100 or len(line) < 3:
        return False
    words = [w for w in line.split() if w.isalpha()]
    if not words or len(words) > 10:
        return False
    if not all(w.isupper() for w in words):
        return False
    line_lower = line.lower()
    return any(kw in line_lower for kw in MEDICAL_KEYWORDS) and not _is_boilerplate(line)


def _is_page_marker(line: str) -> bool:
    """Check if line is a page break or separator."""
    line_clean = line.strip()
    if line_clean.startswith("## PAGE") or line_clean.startswith("## Trang"):
        return True
    return bool(PAGE_MARKER_PATTERN.match(line_clean))


def _update_html_table_state(line_clean: str, in_html_table: bool) -> bool:
    """Update whether parser is currently inside an HTML <table> block."""
    last_open = -1
    for m in HTML_TABLE_OPEN_PATTERN.finditer(line_clean):
        last_open = m.start()
    last_close = -1
    for m in HTML_TABLE_CLOSE_PATTERN.finditer(line_clean):
        last_close = m.start()

    if last_open == -1 and last_close == -1:
        return in_html_table
    if last_open > last_close:
        return True
    return False


def _match_html_heading_or_caption(line_clean: str) -> tuple[bool, int, str]:
    """Detect HTML headings (<h1..h6>) and <p>-wrapped headings or table captions."""
    matched_hh = HTML_HEADING_PATTERN.match(line_clean)
    if matched_hh:
        raw_text = _strip_html_tags(matched_hh.group(2)).strip()
        if raw_text and not _is_boilerplate(raw_text) and not _is_page_marker(raw_text):
            matched_tb = TABLE_HEADER_PATTERN.match(raw_text)
            if matched_tb or re.match(r"^Bảng(?:\s+\d+)?[:\.]", raw_text, re.IGNORECASE):
                return True, 4, matched_tb.group(2).strip() if matched_tb else raw_text
            return True, int(matched_hh.group(1)), raw_text
        return False, 2, ""

    matched_p = HTML_P_PATTERN.match(line_clean)
    if matched_p:
        inner = matched_p.group(1).strip()
        plain_text = _strip_html_tags(inner).strip()
        if not plain_text:
            return False, 2, ""

        matched_tb = TABLE_HEADER_PATTERN.match(plain_text)
        if matched_tb:
            return True, 4, matched_tb.group(2).strip()

        matched_em = HTML_EMPHASIS_PATTERN.match(inner)
        if matched_em and len(plain_text.split()) <= 20 and not _is_boilerplate(plain_text):
            matched_num = re.match(r"^([0-9]{1,2}(?:\.[0-9]{1,2})*\.?)\s+(.+)$", plain_text)
            if matched_num:
                parts = [p for p in matched_num.group(1).split(".") if p]
                return True, min(1 + len(parts), 5), plain_text
            words = plain_text.split()
            level = 2 if words and all(w.isupper() for w in words) else 3
            return True, level, plain_text

    return False, 2, ""


def _is_noise_line(line: str) -> bool:
    """Check if line is OCR artifact or watermark."""
    line_clean = line.strip()
    return line_clean.startswith("[Hình dấu") or line_clean.startswith("(Đã bỏ")


def parse_markdown_to_sections(md_text: str) -> list[dict]:
    """
    Parse markdown text into structured sections with hierarchical parent headers.

    Returns:
        List of dicts: {
            "header": str,
            "parent_header": str,
            "level": int,
            "content": str,
            "has_table": bool
        }
    """
    lines = md_text.splitlines()
    sections: list[dict] = []

    current_header = "Thông tin chung"
    current_parent_header = ""
    current_level = 1
    current_body: list[str] = []
    current_has_table = False
    in_html_table = False

    # Heading hierarchy stack: list of (level, header_text)
    heading_stack: list[tuple[int, str]] = []

    def _flush_section():
        nonlocal current_body, current_has_table
        content_str = "\n".join(current_body).strip()
        if content_str:
            sections.append({
                "header": current_header,
                "parent_header": current_parent_header,
                "level": current_level,
                "content": content_str,
                "has_table": current_has_table,
            })
        current_body = []
        current_has_table = False

    for line in lines:
        line_clean = line.strip()

        if in_html_table or HTML_TABLE_OPEN_PATTERN.search(line_clean):
            current_has_table = True
            current_body.append(line)
            in_html_table = _update_html_table_state(line_clean, in_html_table)
            continue

        if not line_clean:
            if current_body:
                current_body.append("")
            continue

        if _is_page_marker(line_clean) or _is_noise_line(line_clean):
            continue

        if HTML_DIV_ONLY_PATTERN.match(line_clean):
            continue

        if TABLE_LINE_PATTERN.match(line_clean):
            current_has_table = True
            current_body.append(line)
            continue

        heading_candidate = HTML_DIV_WRAPPER_STRIP_PATTERN.sub("", line_clean).strip()

        is_new_section = False
        new_header = ""
        new_level = 2

        # 1. Table caption pattern: Bảng 1: ..., **Bảng 3: ...**
        matched_tb = TABLE_HEADER_PATTERN.match(heading_candidate)
        if matched_tb:
            is_new_section = True
            new_level = 4
            new_header = matched_tb.group(2).strip()

        # 2. Standard markdown heading: # / ## / ### / ####
        if not is_new_section:
            matched_h = HEADING_PATTERN.match(heading_candidate)
            if matched_h:
                is_new_section = True
                new_level = len(matched_h.group(1))
                new_header = matched_h.group(2).strip()

        # 2b. HTML heading (<h1..h6>) or <p>-wrapped heading / table caption
        if not is_new_section:
            matched_html, html_level, html_header = _match_html_heading_or_caption(heading_candidate)
            if matched_html:
                is_new_section = True
                new_level = html_level
                new_header = html_header

        # 3. Bold heading: **3.1. Chỉ định** or **THÀNH PHẦN**
        if not is_new_section:
            matched_b = BOLD_HEADING_PATTERN.match(heading_candidate)
            if matched_b and len(heading_candidate.split()) <= 20:
                raw_text = matched_b.group(2).strip()
                if not _is_boilerplate(raw_text):
                    is_new_section = True
                    num_part = matched_b.group(1)
                    if num_part:
                        parts = [p for p in num_part.split(".") if p]
                        new_level = min(1 + len(parts), 5)
                        new_header = f"{num_part} {raw_text}"
                    else:
                        words = raw_text.split()
                        new_level = 2 if words and all(w.isupper() for w in words) else 3
                        new_header = raw_text

        # 4. Uppercase text heading without markdown tag (e.g. CHỈ ĐỊNH ĐIỀU TRỊ)
        if not is_new_section and _is_all_caps_heading(heading_candidate):
            is_new_section = True
            new_level = 2
            new_header = heading_candidate

        # 5. Numbered heading without markdown tag (e.g. 1. Mô tả, 6. Liều dùng)
        if not is_new_section:
            matched_num = NUMBERED_HEADER_PATTERN.match(heading_candidate)
            if matched_num and len(heading_candidate.split()) <= 12:
                line_lower = heading_candidate.lower()
                if any(kw in line_lower for kw in MEDICAL_KEYWORDS) and not _is_boilerplate(heading_candidate):
                    is_new_section = True
                    num_str = matched_num.group(1)
                    parts = [p for p in num_str.split(".") if p]
                    new_level = min(1 + len(parts), 5)
                    new_header = heading_candidate

        if is_new_section:
            _flush_section()

            # Update heading stack to determine parent_header
            while heading_stack and heading_stack[-1][0] >= new_level:
                heading_stack.pop()
            parent_hdr = heading_stack[-1][1] if heading_stack else ""
            heading_stack.append((new_level, new_header))

            current_header = new_header
            current_parent_header = parent_hdr
            current_level = new_level
            current_body = []
            current_has_table = False
        else:
            current_body.append(line)

    _flush_section()
    return sections
