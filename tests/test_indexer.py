"""Unit tests for vacxis_capstone_fa26.indexing.indexer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pytest
from qdrant_client import models

from vacxis_capstone_fa26.indexing.chunker import _count_tokens, create_medical_chunks
from vacxis_capstone_fa26.indexing.indexer import (
    convert_sparse_to_qdrant,
    init_qdrant_collection,
    load_chunks_from_parquet,
)
from vacxis_capstone_fa26.indexing.main import run_indexing_pipeline
from vacxis_capstone_fa26.indexing.parser import parse_markdown_to_sections
from vacxis_capstone_fa26.indexing.schemas import detect_content_type
from vacxis_capstone_fa26.indexing.chunker import _split_sentences


def test_convert_sparse_to_qdrant_empty():
    """Test convert_sparse_to_qdrant with None or empty dict."""
    res_none = convert_sparse_to_qdrant(None)
    assert isinstance(res_none, models.SparseVector)
    assert res_none.indices == []
    assert res_none.values == []

    res_empty = convert_sparse_to_qdrant({})
    assert isinstance(res_empty, models.SparseVector)
    assert res_empty.indices == []
    assert res_empty.values == []


def test_convert_sparse_to_qdrant_values():
    """Test convert_sparse_to_qdrant with string token ids and weights."""
    sample_weights = {
        "101": 0.45,
        "2045": 1.25,
        "309": 0.08,
    }
    sparse_vec = convert_sparse_to_qdrant(sample_weights)
    assert isinstance(sparse_vec, models.SparseVector)
    assert sparse_vec.indices == [101, 2045, 309]
    assert [round(v, 2) for v in sparse_vec.values] == [0.45, 1.25, 0.08]


def test_init_qdrant_collection_creation():
    """Test init_qdrant_collection creates collection if not existing."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.name = "Other_Collection"
    mock_client.get_collections.return_value.collections = [mock_collection]

    init_qdrant_collection(mock_client, collection_name="VacxiS_Capstone_FA26", recreate=True)

    mock_client.delete_collection.assert_not_called()
    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == "VacxiS_Capstone_FA26"
    assert "dense" in call_kwargs["vectors_config"]
    assert "sparse" in call_kwargs["sparse_vectors_config"]


def test_init_qdrant_collection_recreate():
    """Test init_qdrant_collection deletes existing collection when recreate=True."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.name = "VacxiS_Capstone_FA26"
    mock_client.get_collections.return_value.collections = [mock_collection]

    init_qdrant_collection(mock_client, collection_name="VacxiS_Capstone_FA26", recreate=True)

    mock_client.delete_collection.assert_called_once_with("VacxiS_Capstone_FA26")
    mock_client.create_collection.assert_called_once()


def test_load_chunks_from_parquet(tmp_path: Path):
    """Test loading chunks from parquet using PyArrow."""
    test_parquet = tmp_path / "test_chunks.parquet"

    schema = pa.schema([
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("doc_title", pa.string()),
        ("category", pa.string()),
        ("parent_header", pa.string()),
        ("header", pa.string()),
        ("content_type", pa.string()),
        ("text", pa.string()),
        ("metadata", pa.struct([
            ("has_table", pa.bool_()),
            ("source_file", pa.string()),
        ])),
    ])

    data = {
        "chunk_id": ["chk_1"],
        "doc_id": ["doc_1"],
        "doc_title": ["Doc 1"],
        "category": ["chỉ định"],
        "parent_header": [""],
        "header": ["Section 1"],
        "content_type": ["text"],
        "text": ["Content text"],
        "metadata": [{"has_table": False, "source_file": "file.md"}],
    }

    table = pa.Table.from_pydict(data, schema=schema)
    pq.write_table(table, str(test_parquet))

    records = load_chunks_from_parquet(test_parquet)
    assert len(records) == 1
    assert records[0]["chunk_id"] == "chk_1"
    assert records[0]["category"] == "chỉ định"
    assert records[0]["metadata"]["source_file"] == "file.md"


def _section(content, *, header="Thông tin", has_table=False):
    return {"header": header, "parent_header": "Tổng quan", "content": content, "has_table": has_table}


def test_chunker_keeps_small_table_unchanged():
    table = "| Tên | Giá trị |\n| :--- | ---: |\n| Mẫu |  Đã ghi  |  "
    section = _section(table, has_table=True)

    chunk, = create_medical_chunks([section], max_tokens=_count_tokens(table))

    assert chunk["content"] == table
    assert chunk["parent_header"] == section["parent_header"]
    assert chunk["has_table"] and chunk["content_type"] == "table"


@pytest.mark.parametrize(("columns", "row_count"), [(1, 18), (2, 400), (5, 18)])
def test_chunker_splits_large_tables_without_losing_rows(columns, row_count):
    def row(cells):
        return "| " + " | ".join(cells) + " |"

    heading = [row([f"Cột {i}" for i in range(columns)]), row([":---:"] * columns)]
    rows = [row([f"giá trị {r}.{c}" for c in range(columns)]) for r in range(row_count)]
    section = _section("\n".join(heading + rows), has_table=True)

    chunks = create_medical_chunks([section], max_tokens=75, chunk_overlap_tokens=0)

    assert 1 < len(chunks) <= len(rows)
    split_rows = []
    for chunk in chunks:
        lines = chunk["content"].splitlines()
        assert lines[:2] == heading and len(lines) > 2
        assert _count_tokens(chunk["content"]) <= 75
        assert chunk["has_table"] and chunk["content_type"] == "table"
        split_rows.extend(lines[2:])
    assert split_rows == rows


def test_chunker_continues_oversized_row_in_its_original_columns():
    heading = [r"| Nhãn \| phụ | Đường dẫn \\ | Ghi chú |", "| :--- | ---: | :---: |"]
    cells = [
        r"Một \| hai " + " ".join(f"giá-trị-{i}" for i in range(58)),
        r"Đường dẫn \\ giữ nguyên",
        "Tiếng Việt có dấu",
    ]
    row = "| " + " | ".join(cells) + " |"
    following_row = r"| a\|b | c | d |"
    section = _section("\n".join([*heading, row, following_row]), has_table=True)

    chunks = create_medical_chunks([section], max_tokens=35)

    for chunk in chunks:
        assert chunk["content"].splitlines()[:2] == heading
        assert _count_tokens(chunk["content"]) <= 35
    fragments = [chunk["content"].splitlines()[2:] for chunk in chunks]
    assert fragments[-1] == [following_row]
    assert all(len(lines) == 1 for lines in fragments)
    recovered = [[], [], []]
    for (line,) in fragments[:-1]:
        left, _, value_and_right = line.partition(" ")
        value, _, right = value_and_right.rpartition(" ")
        column = len(left) - 1
        assert left == "|" * (column + 1) and right == "|" * (3 - column)
        recovered[column].extend(value.split())
    assert recovered == [cell.split() for cell in cells]
    assert create_medical_chunks([section], max_tokens=35) == chunks


def test_chunker_preserves_oversized_empty_row():
    heading = ["|A|B|C|D|", "|-|-|-|-|"]
    section = _section("\n".join([*heading, "| | | | |", "|a|b|c|d|"]), has_table=True)

    chunks = create_medical_chunks([section], max_tokens=4)

    assert [chunk["content"].splitlines() for chunk in chunks] == [
        [*heading, "|||||"], [*heading, "|a|b|c|d|"],
    ]


@pytest.mark.parametrize(("max_tokens", "fragments"), [
    (3, ["|alpha\\", "|beta\\", "|gamma", "||delta"]),
    (4, [r"|alpha\ beta" + "\\", "|gamma", "||delta"]),
])
def test_chunker_continues_oversized_row_when_only_compact_rows_fit(max_tokens, fragments):
    heading = ["|A|B|", "|---|---|"]
    section = _section("\n".join([*heading, r"| alpha\ beta\ gamma | delta |"]), has_table=True)

    chunks = create_medical_chunks([section], max_tokens=max_tokens)

    assert [chunk["content"].splitlines() for chunk in chunks] == [
        [*heading, fragment] for fragment in fragments
    ]
    assert all(_count_tokens(chunk["content"]) <= max_tokens for chunk in chunks)


@pytest.mark.parametrize("overlap", [None, 0, 12])
def test_chunker_text_overlap_uses_word_boundaries(overlap):
    words = [f"từ-{i}" for i in range(240)]
    section = _section(" ".join(words))
    kwargs = {} if overlap is None else {"chunk_overlap_tokens": overlap}
    expected_overlap = 50 if overlap is None else overlap

    chunks = create_medical_chunks([section], max_tokens=90, **kwargs)

    parts = [chunk["content"].split() for chunk in chunks]
    assert len(parts) > 1
    assert all(expected_overlap < len(part) <= 90 for part in parts)
    recovered = parts[0].copy()
    for previous, current in zip(parts, parts[1:]):
        assert current[:expected_overlap] == previous[len(previous) - expected_overlap:]
        recovered.extend(current[expected_overlap:])
    assert recovered == words


def test_chunker_table_overlap_repeats_complete_rows_and_fits_budget():
    heading = ["| Mẫu | Giá-trị |", "| --- | --- |"]
    rows = [f"| r{i} | một hai ba bốn năm sáu |" for i in range(20)]
    section = _section("\n".join(heading + rows), has_table=True)

    chunks = create_medical_chunks([section], max_tokens=90)

    parts = [chunk["content"].splitlines() for chunk in chunks]
    assert all(lines[:2] == heading and len(lines) > 2 for lines in parts)
    assert all(_count_tokens(chunk["content"]) <= 90 for chunk in chunks)
    recovered = parts[0][2:]
    for previous, current in zip(parts, parts[1:]):
        assert current[2:7] == previous[-5:]
        assert _count_tokens("\n".join(current[2:7])) == 50
        assert current[7:]
        recovered.extend(current[7:])
    assert recovered == rows

    tight_chunks = create_medical_chunks([section], max_tokens=20)
    assert [chunk["content"].splitlines()[2:] for chunk in tight_chunks] == [[row] for row in rows]


def test_chunker_mixed_blocks_and_section_boundaries():
    before = " ".join(f"trước-{i}" for i in range(100))
    after = " ".join(f"sau-{i}" for i in range(100))
    heading = ["Cột | Ghi chú", ":--- | ---:"]
    rows = [r"mẫu \| gốc | kết quả", "mẫu khác | ghi chú |"]
    table = "\n".join(heading + rows)
    section = _section(f"{before}\n\n{table}\n\n{after}")
    neighbor = _section("Tách riêng", header="Mục khác")

    chunks = create_medical_chunks([section, neighbor], max_tokens=80)

    assert len(chunks) == 6
    assert [chunk["content_type"] for chunk in chunks] == [
        "text", "text", "table", "text", "text", "text",
    ]
    assert chunks[2]["content"] == table
    assert chunks[1]["content"].split()[:50] == chunks[0]["content"].split()[-50:]
    assert chunks[4]["content"].split()[:50] == chunks[3]["content"].split()[-50:]
    assert chunks[3]["content"].startswith("sau-0 ")
    assert chunks[-1]["content"] == neighbor["content"]


def test_chunker_rejects_negative_overlap_and_progresses_on_tight_budget():
    with pytest.raises(ValueError, match="chunk_overlap_tokens"):
        create_medical_chunks([], chunk_overlap_tokens=-1)

    chunks = create_medical_chunks([_section("một hai ba"), _section(" ")], max_tokens=1)

    assert [chunk["content"] for chunk in chunks] == ["một", "hai", "ba"]


def test_indexing_budgets_final_embedding_text(tmp_path):
    folder = tmp_path / "CLEANED" / "HUNG" / "001"
    folder.mkdir(parents=True)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text('{"001": "Tài liệu thử nghiệm"}', encoding="utf-8")
    first = " ".join(f"trước-{i}" for i in range(38))
    second = " ".join(f"sau-{i}" for i in range(38))
    heading = ["| Mẫu | Giá-trị |", "| --- | --- |"]
    rows = [f"| r{i} | một hai ba bốn năm sáu |" for i in range(18)]
    table = "\n".join(heading + rows)
    (folder / "fixture.md").write_text(
        f"# Mục đầu\n{first}\n\n# Mục tiếp\n{second}\n\n# Bảng\n{table}", encoding="utf-8",
    )

    output = run_indexing_pipeline(tmp_path, mapping_path=mapping_path, min_tokens=100, max_tokens=85)
    records = pq.read_table(output).to_pylist()

    assert [record["text"].split("\n", 1)[1] for record in records[:2]] == [first, second]
    assert len(records) > 3
    for index, record in enumerate(records):
        assert _count_tokens(record["text"]) <= 85
        assert record["text"].startswith(f"[CHỦ ĐỀ: Tài liệu thử nghiệm] > [MỤC: {record['header']}]\n")
        assert record["chunk_id"] == f"{record['doc_id']}_c_{index:04d}"
    parts = [record["text"].splitlines()[1:] for record in records[2:]]
    assert all(lines[:2] == heading for lines in parts)
    assert parts[1][2:7] == parts[0][-5:]


def test_parser_detects_html_headings_tables_divs_and_page_markers():
    md = "\n".join([
        "<h2>THÀNH PHẦN</h2>",
        "<p>Mỗi liều 0.5 ml chứa kháng nguyên.</p>",
        "<div>",
        "<p><strong>Bảng 1: Thành phần vắc xin</strong></p>",
        "<table>",
        "<thead><tr><th>Hoạt chất</th><th>Hàm lượng</th></tr></thead>",
        "<tbody>",
        "<tr><td># Không tách mục trong bảng</td><td>10 mcg</td></tr>",
        "</tbody>",
        "</table>",
        "# PAGE 12",
        r"\[Page 13\]",
        "<p><sup>a</sup> Chú thích quan trọng sau bảng.</p>",
        "</div>",
        "<p><u><strong>2. Liều dùng</strong></u></p>",
        "<p>Tiêm bắp.</p>",
    ])

    sections = parse_markdown_to_sections(md)

    assert [s["header"] for s in sections] == [
        "THÀNH PHẦN",
        "Bảng 1: Thành phần vắc xin",
        "2. Liều dùng",
    ]
    assert sections[0]["has_table"] is False
    assert sections[1]["has_table"] is True
    assert sections[1]["parent_header"] == "THÀNH PHẦN"
    assert sections[1]["level"] == 4
    assert "<sup>a</sup>" in sections[1]["content"]
    assert "# PAGE 12" not in sections[1]["content"]
    assert r"\[Page 13\]" not in sections[1]["content"]
    assert sections[2]["level"] == 2


def test_count_tokens_accounts_for_html_tags_without_changing_plain_text():
    assert _count_tokens("một hai ba bốn") == 4
    assert _count_tokens("giá trị a < b là đúng") == 7
    html_row = "<tr><td>A</td><td>B</td></tr>"
    assert _count_tokens(html_row) > len(html_row.split())


def test_chunker_keeps_small_html_table_with_footnote():
    html_table = (
        "<table>\n"
        "<thead><tr><th>Nhóm tuổi</th><th>Liều</th></tr></thead>\n"
        "<tbody><tr><td>Trẻ em</td><td>0.5 ml</td></tr></tbody>\n"
        "</table>\n"
        "<p><sup>a</sup> Tiêm nhắc lại sau 6 tháng.</p>"
    )
    section = _section(html_table, header="Bảng 1: Lịch tiêm", has_table=True)

    chunk, = create_medical_chunks([section], max_tokens=_count_tokens(html_table) + 10)

    assert chunk["has_table"] is True
    assert chunk["content_type"] == "table"
    assert "<sup>a</sup>" in chunk["content"]
    assert "<table>" in chunk["content"]


def test_chunker_splits_large_html_table_repeating_thead_and_overlap():
    thead = "<thead><tr><th>Nhóm</th><th>Giá trị</th></tr></thead>"
    rows = [
        f"<tr><td>nhóm-{i}</td><td>một hai ba bốn năm</td></tr>"
        for i in range(12)
    ]
    html = f"<table>\n{thead}\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"
    section = _section(html, header="Bảng 2: Dữ liệu", has_table=True)

    chunks = create_medical_chunks([section], max_tokens=120, chunk_overlap_tokens=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["has_table"] is True
        assert chunk["content_type"] == "table"
        assert thead in chunk["content"]
        assert "<table>" in chunk["content"]
        assert "</table>" in chunk["content"]
        assert _count_tokens(chunk["content"]) <= 120


def test_chunker_promotes_first_th_row_when_thead_missing_and_splits_oversized_row():
    header_row = "<tr><th>Cột A</th><th>Cột B</th></tr>"
    big_row = (
        "<tr><td>"
        + " ".join(f"giá-trị-{i}" for i in range(45))
        + "</td><td>Ngắn</td></tr>"
    )
    normal_row = "<tr><td>Nhỏ</td><td>Bình thường</td></tr>"
    html = f"<table>\n<tbody>\n{header_row}\n{big_row}\n{normal_row}\n</tbody>\n</table>"
    section = _section(html, header="Bảng 3: Không thead", has_table=True)

    chunks = create_medical_chunks([section], max_tokens=80, chunk_overlap_tokens=0)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert "<thead>" in chunk["content"]
        assert header_row in chunk["content"]
        assert _count_tokens(chunk["content"]) <= 80


def test_detect_content_type_for_html_and_mixed():
    html_only = "<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>"
    assert detect_content_type(html_only, True) == "table"
    mixed = (
        "<p>Đoạn văn bản giới thiệu rất dài về chỉ định tiêm chủng cho nhiều đối tượng khác nhau.</p>\n"
        "<p>Thêm một đoạn giải thích chi tiết về cách sử dụng và bảo quản vắc xin.</p>\n"
        "<p>Thêm đoạn thứ ba mô tả các lưu ý quan trọng trước khi tiêm.</p>\n"
        "<table><tr><td>A</td><td>1</td></tr></table>"
    )
    assert detect_content_type(mixed, True) == "mixed"


def test_spans_split_sentences_skips_html_table_markup():
    chunk_text = (
        "[CHỦ ĐỀ: Vắc xin A] > [MỤC: Liều dùng]\n"
        "<table><thead><tr><th>Độ tuổi</th></tr></thead><tbody><tr><td>Người lớn cần tiêm một liều duy nhất 0.5 ml vào cơ delta.</td></tr></tbody></table>\n\n"
        "Liều tiêm bắp tiêu chuẩn cho người lớn là 0.5 ml mỗi lần tiêm tại cơ delta."
    )
    spans = _split_sentences(chunk_text)
    extracted = [chunk_text[s:e] for s, e in spans]
    assert len(extracted) == 1
    assert "<table>" not in extracted[0]
    assert "Liều tiêm bắp tiêu chuẩn" in extracted[0]
