"""Schema definitions, category mapping, and ID generation."""
import re
import unicodedata

# Keyword patterns to map headers to standard medical categories
# Note: More specific keywords must precede general ones (e.g., 'chống chỉ định' before 'chỉ định')
CATEGORY_RULES: list[tuple[list[str], str]] = [
    (["chống chỉ định"], "chống chỉ định"),
    (["chỉ định"], "chỉ định"),
    (["thành phần", "định tính", "định lượng", "hoạt chất", "tá dược"], "thành phần"),
    (["dạng bào chế"], "dạng bào chế"),
    (["liều lượng", "liều dùng", "cách dùng", "đường dùng"], "liều lượng và cách dùng"),
    (["cảnh báo", "thận trọng"], "cảnh báo và thận trọng"),
    (["tương tác thuốc", "tương tác"], "tương tác thuốc"),
    (["tác dụng không mong muốn", "tác dụng phụ", "phản ứng"], "tác dụng không mong muốn"),
    (["quá liều"], "quá liều"),
    (["dược lực", "hiệu quả", "tính sinh miễn dịch"], "dược lực học"),
    (["dược động"], "dược động học"),
    (["phi lâm sàng", "tiền lâm sàng"], "dữ liệu phi lâm sàng"),
    (["đặc tính dược lý", "đặc tính lâm sàng", "nhóm trị liệu"], "đặc tính dược lý"),
    (["bảo quản", "hạn dùng"], "bảo quản"),
    (["đóng gói", "quy cách"], "quy cách đóng gói"),
    (["phụ nữ có thai", "cho con bú", "thai kỳ"], "thai kỳ và cho con bú"),
    (["lái xe", "vận hành máy móc"], "ảnh hưởng lái xe"),
    (["hướng dẫn sử dụng", "chuẩn bị"], "hướng dẫn sử dụng"),
    (["nhà sản xuất", "đăng ký"], "thông tin nhà sản xuất"),
    (["tương kỵ"], "tương kỵ"),
    (["bảng"], "bảng số liệu"),
]


def classify_category(header: str, parent_header: str = "") -> str:
    """Map section header and optional parent header to a standardized category."""
    header_lower = header.lower()
    for keywords, category in CATEGORY_RULES:
        if any(kw in header_lower for kw in keywords):
            # If header is generic 'bảng số liệu', inherit parent's category if available
            if category == "bảng số liệu" and parent_header:
                parent_category = classify_category(parent_header)
                if parent_category != "thông tin chung":
                    return parent_category
            return category

    if parent_header:
        parent_lower = parent_header.lower()
        for keywords, category in CATEGORY_RULES:
            if any(kw in parent_lower for kw in keywords):
                return category

    return "thông tin chung"


_HTML_TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TABLE_OPEN_RE = re.compile(r"<table\b", re.IGNORECASE)
_HTML_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)


def detect_content_type(content: str, has_table: bool) -> str:
    """Detect whether chunk content is text, table, or mixed."""
    if not has_table:
        return "text"

    stripped = content.strip()
    if not stripped:
        return "mixed"

    # Compute token (word) ratio for HTML tables + Markdown pipe lines
    total_words = len(stripped.split())
    html_table_spans = _HTML_TABLE_BLOCK_RE.findall(stripped)
    html_table_words = sum(len(span.split()) for span in html_table_spans)
    outside_html = _HTML_TABLE_BLOCK_RE.sub(" ", stripped)
    md_table_words = sum(
        len(line.split())
        for line in outside_html.splitlines()
        if line.strip().startswith("|")
    )
    word_ratio = (html_table_words + md_table_words) / total_words if total_words else 0.0

    # Also compute non-empty line ratio (handles multi-line HTML tables and Markdown tables)
    lines = [line for line in stripped.splitlines() if line.strip()]
    table_lines = 0
    in_html_table = False
    for line in lines:
        line_clean = line.strip()
        if in_html_table or _HTML_TABLE_OPEN_RE.search(line_clean):
            table_lines += 1
            if _HTML_TABLE_OPEN_RE.search(line_clean):
                in_html_table = True
            if _HTML_TABLE_CLOSE_RE.search(line_clean):
                in_html_table = False
        elif line_clean.startswith("|"):
            table_lines += 1

    line_ratio = table_lines / len(lines) if lines else 0.0
    if max(word_ratio, line_ratio) > 0.6:
        return "table"
    return "mixed"


def slugify(text: str) -> str:
    """Convert input string to a clean alphanumeric slug with underscores."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def make_chunk_id(doc_id: str, chunk_idx: int) -> str:
    """Generate chunk_id matching convention: {doc_id}_c_{idx:04d}."""
    return f"{doc_id}_c_{chunk_idx:04d}"
