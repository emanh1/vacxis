# VaxiS

VaxiS hỗ trợ tra cứu tài liệu vắc xin bằng câu hỏi tiếng Việt. Pipeline phân đoạn Markdown đã làm sạch, lập chỉ mục, truy xuất bằng chứng và sinh câu trả lời kèm mã đoạn nguồn. Hệ thống phục vụ tra cứu thông tin đã công bố.

- `CLEANED/`, `scrape_dataset/` và `75_vaccine_names_mapping.json` chứa dữ liệu nguồn, tài liệu đã làm sạch và ánh xạ tên vắc xin.
- `chunks_storage/` chứa dữ liệu phân đoạn Parquet và snapshot Qdrant.
- `src/vacxis_capstone_fa26/` chứa pipeline lập chỉ mục, truy xuất, sinh câu trả lời, đánh giá và ứng dụng chat.
- [Ứng dụng rà soát](review-web/README.md) hỗ trợ đối chiếu câu hỏi, câu trả lời tham chiếu và bằng chứng.

Pipeline kiểm tra mã trích dẫn có nằm trong các đoạn đã cung cấp cho mô hình. Việc xác nhận đoạn nguồn có thực sự hỗ trợ nội dung câu trả lời là bước đánh giá riêng.

## Thu thập và chuyển đổi tài liệu

Tài liệu PDF được thu thập từ cổng dịch vụ công của Cục Quản lý Dược tại [dichvucong.dav.gov.vn](https://dichvucong.dav.gov.vn/). Mã trong `scrape_dataset/scrape_data.py` đọc danh sách đăng ký đã lưu và lấy lịch sử cùng thông tin nguyên liệu. `extract_documents.py` trích xuất đường dẫn tài liệu từ danh sách, gồm hướng dẫn sử dụng, nhãn, nhãn kết hợp hướng dẫn sử dụng và tài liệu tiêu chuẩn chất lượng. `download_documents.py` tải tài liệu vào thư mục theo số đăng ký và lưu tiến độ cùng lỗi tải.

Nhóm thu thập ngày 18/09/2026 từ 123 listing, loại 44 listing không có tài liệu, 1 lỗi URL, 2 thư mục rỗng và 1 chỉ có nhãn, còn 75 PDF hướng dẫn sử dụng. PDF được cắt bỏ phần không cần thiết, tải từng tệp lên website **Google Gemini** với model mặc định để chuyển sang Markdown. Ảnh bảng được chuyển thành HTML rồi chèn vào Markdown. Các tệp được rà soát thủ công sau chuyển đổi. [PLACEHOLDER: bổ sung prompt chuyển đổi nếu còn lưu.]

## Chạy cục bộ

Dùng Python 3.12 trở lên và uv. Chạy từ thư mục gốc.

```bash
uv sync --group dev
cp .env.example .env
```

Điền `OPENROUTER_API_KEY` và `VACXIS_GENERATION_MODEL` để sinh câu trả lời trực tuyến. `VACXIS_LLM_BASE_URL` mặc định là `https://openrouter.ai/api/v1`; `VACXIS_EVALUATION_MODEL` chọn model chấm độc lập. Đặt `VACXIS_QUERY_PLANNER_MODEL` nếu dùng giao diện hội thoại. Truy xuất vector mặc định dùng Qdrant tại `localhost:6333`, collection `VacxiS_Capstone_FA26`, BGE-M3 tại `models/bge-m3` và reranker tại `models/bge-reranker`.

Chạy các lệnh truy xuất trên máy có đủ tài nguyên:

```bash
uv run --extra retrieval vacxis-pipeline prepare
uv run --extra retrieval --env-file .env vacxis-pipeline ask --query 'Tờ hướng dẫn Verorab đề cập các phản ứng sau tiêm nào?'
uv run --extra retrieval --extra web --env-file .env vacxis-web
```

`prepare` tạo dữ liệu phân đoạn và lập chỉ mục. Để dùng snapshot sẵn có, xem [hướng dẫn nhập Qdrant](import_qdrant_vecstore.md). Tùy chọn `ask --dry-run` dùng câu trả lời trích xuất offline và vẫn chạy truy xuất.

## Đánh giá và báo cáo

Thứ tự khảo sát là **Naive → Advanced → Hybrid (baseline) → Table-aware → Graph RAG → Agent RAG**. Naive dùng BM25. Advanced dùng truy xuất dense BGE-M3. Hybrid hợp nhất thứ hạng dense và sparse của BGE-M3 và có thể chạy reranker. Xem [hướng dẫn đánh giá](evaluation/README.md) để chạy với cùng dữ liệu và độ sâu truy xuất.

- **Table-aware** — Có mã lập chỉ mục `summarize_dual` và kết quả truy xuất sơ bộ M3b. [PLACEHOLDER: bổ sung cấu hình lập chỉ mục cuối cùng, đặc tả triển khai và kết quả chất lượng câu trả lời.]
- **Graph RAG** — Có tích hợp LightRAG và kết quả truy xuất sơ bộ M4. [PLACEHOLDER: xác nhận cấu hình cuối cùng, đặc tả đồ thị và kết quả chất lượng câu trả lời.]
- **Agent RAG** — [PLACEHOLDER: bổ sung phiên bản triển khai, công cụ và điều kiện dừng, giao thức đánh giá và chỉ số.]

Các tùy chọn `--retriever-mode`, lập chỉ mục bảng, retry và resume được mô tả trong [hướng dẫn đánh giá](evaluation/README.md) và [hướng dẫn phục hồi kho dữ liệu](import_qdrant_vecstore.md). Kết quả và các điểm cần xác minh được lưu trong [ghi chú kiểm tra kết quả](results/report_discrepancies.md).

Bản thảo tiếng Anh nằm trong [report_draft.tex](report_draft.tex). Trên Overleaf, tải bản thảo và các tệp `.local/sn-jnl.cls`, `.local/references.bib` và `.local/logo.png` vào cùng thư mục. Chọn `report_draft.tex` làm tài liệu chính với pdfLaTeX. Tìm `PLACEHOLDER` để xác định thông tin còn thiếu.

## Đồng bộ nhánh báo cáo

`main` là nguồn tham chiếu cho mã triển khai và kết quả. Nhánh `report-write` tích hợp các thay đổi từ `main` rồi cập nhật báo cáo, tài liệu và các câu hỏi còn thiếu. Các xác nhận của nhóm được lưu trong [report_clarifications.md](report_clarifications.md).
