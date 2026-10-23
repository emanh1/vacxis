"""Prompt templates and formatting for Medical Vaccine RAG."""
from __future__ import annotations

from typing import Any, Sequence


DEFAULT_MEDICAL_SYSTEM_PROMPT = """Bạn là trợ lý thông tin y tế chuyên sâu về vắc-xin tại Việt Nam.
Nhiệm vụ của bạn là giải đáp thắc mắc của người dùng dựa DUY NHẤT trên các đoạn tài liệu Hướng dẫn sử dụng (Product Inserts) được cung cấp.

CÁC NGUYÊN TẮC BẮT BUỘC:
1. TRUNG THỰC VÀ BÁM SÁT TÀI LIỆU (GROUNDEDNESS):
   - Chỉ sử dụng các thông tin có trong các đoạn văn bản được cung cấp.
   - Tuyệt đối không tự suy diễn, phỏng đoán, suy diễn liều lượng hoặc thêm thông tin ngoài tài liệu.
2. TRÍCH DẪN NGUỒN TƯỜNG MINH (CITATIONS):
   - Sau mỗi khẳng định y khoa, sự kiện, liều lượng hoặc khuyến cáo, BẮT BUỘC chèn mã định danh chunk dạng `[CHUNK_ID]` ngay cuối câu hoặc mệnh đề đó.
   - Ví dụ: "Vaccine Gardasil 9 được chỉ định cho người từ 9 đến 45 tuổi [gardasil_9_001310178700_c_0001]."
3. NGUYÊN TẮC AN TOÀN Y TẾ VÀ TỪ CHỐI (SAFETY & GUARDRAILS):
   - Nếu các đoạn văn bản KHÔNG chứa đủ thông tin để trả lời câu hỏi, hãy trả lời rõ ràng:
     "Tài liệu hướng dẫn sử dụng hiện tại không cung cấp thông tin cho câu hỏi này. Bạn nên tham khảo ý kiến bác sĩ hoặc chuyên viên y tế tại cơ sở tiêm chủng để được tư vấn chính xác."
   - Không tự ý chẩn đoán bệnh hoặc kê đơn thuốc thay bác sĩ chuyên khoa.
4. VĂN PHONG:
   - Rõ ràng, trung thực, chuẩn mực y khoa, súc tích và dễ hiểu đối với người đọc."""


REFUSAL_TRIGGER_PHRASES: tuple[str, ...] = (
    "tài liệu hướng dẫn sử dụng hiện tại không cung cấp",
    "không có thông tin trong tài liệu",
    "không tìm thấy thông tin",
    "không được đề cập trong tài liệu",
    "tài liệu không nhắc đến",
    "tài liệu không cung cấp",
)


def get_chunk_field(c: Any, field_name: str, default: Any = "") -> Any:
    """Safely extract a field from either an object or a dictionary."""
    if isinstance(c, dict):
        if field_name in c and c[field_name] is not None:
            return c[field_name]
        upper_field = field_name.upper()
        if upper_field in c and c[upper_field] is not None:
            return c[upper_field]
        return default
    return getattr(c, field_name, default)


def format_context(chunks: Sequence[Any], include_metadata: bool = True) -> str:
    """Format retrieved chunks into a clear, referenced text block."""
    if not chunks:
        return "Không có đoạn tài liệu tham khảo nào được tìm thấy."

    formatted_blocks: list[str] = []
    for idx, c in enumerate(chunks, start=1):
        cid = get_chunk_field(c, "chunk_id", str(idx))
        text = str(get_chunk_field(c, "text", ""))

        if include_metadata:
            doc_title = get_chunk_field(c, "doc_title", "")
            category = get_chunk_field(c, "category", "")
            header = get_chunk_field(c, "header", "")

            header_parts: list[str] = []
            if doc_title:
                header_parts.append(f"Vắc-xin: {doc_title}")
            if category:
                header_parts.append(f"Chuyên mục: {category}")
            if header:
                header_parts.append(f"Mục: {header}")

            meta_line = " | ".join(header_parts) if header_parts else "Thông tin chung"
            block = f"--- [TÀI LIỆU {idx} | CHUNK_ID: {cid} | {meta_line}] ---\n{text.strip()}"
        else:
            block = f"--- [TÀI LIỆU {idx} | CHUNK_ID: {cid}] ---\n{text.strip()}"

        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


def build_messages(
    query: str,
    chunks: Sequence[Any],
    system_prompt: str | None = None,
    include_metadata: bool = True,
    enforce_citations: bool = True,
) -> list[dict[str, str]]:
    """Construct messages payload compatible with LiteLLM/OpenAI format."""
    sys_prompt = system_prompt or DEFAULT_MEDICAL_SYSTEM_PROMPT
    context_str = format_context(chunks, include_metadata=include_metadata)

    citation_instruction = (
        "Hãy trả lời câu hỏi dựa trên các tài liệu trên và trích dẫn mã [CHUNK_ID] đầy đủ sau mỗi câu khẳng định."
        if enforce_citations
        else "Hãy trả lời câu hỏi dựa trên các tài liệu trên."
    )

    user_content = (
        f"CÂU HỎI CỦA NGƯỜI DÙNG: {query.strip()}\n\n"
        f"CÁC ĐOẠN TÀI LIỆU HƯỚNG DẪN SỬ DỤNG:\n{context_str}\n\n"
        f"{citation_instruction}"
    )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
