# VaxiS Review

Ứng dụng FastAPI dùng để rà soát câu hỏi, câu trả lời tham chiếu và đoạn bằng chứng. Quản trị viên cấp lời mời, phân công và xem lịch sử. Người duyệt có thể đối chiếu tài liệu, sửa nội dung, chọn bằng chứng và gửi kết quả. Mã nguồn hỗ trợ luồng đồng thuận và phân xử.

## Chuẩn bị và chạy cục bộ

Từ thư mục gốc, tạo dữ liệu nhập từ bộ 300 câu đã sửa nhãn nguồn. Lệnh chuẩn bị đọc Parquet.

```bash
uv run scripts/prepare_review_dataset.py \
  evaluation/eval_dataset_300_final_fixed.json \
  chunks_storage/chunks.parquet \
  review-web/eval_dataset_300_native.json
cd review-web
uv sync
cp .env.example .env
uv run python -c 'import secrets; print(secrets.token_urlsafe(48)); print(secrets.token_urlsafe(32))'
```

Điền hai giá trị khác nhau vào `REVIEW_SECRET_KEY` và `REVIEW_ADMIN_PASSWORD`. Giữ `DATABASE_URL` trống để dùng SQLite cục bộ. Có thể đặt `REVIEW_DB_PATH` để dùng một tệp SQLite riêng cho đợt duyệt.

```bash
uv run --env-file .env store.py eval_dataset_300_native.json
uv run --env-file .env uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở [trang quản trị cục bộ](http://127.0.0.1:8000/admin/login). Dùng lời mời thử nghiệm nội bộ để kiểm tra luồng trước khi cấp lời mời duyệt thật. Người duyệt mở liên kết hoặc nhập mã và dùng trình duyệt đã nhận lời mời để quay lại phiên.

## Tài liệu, duyệt và xuất dữ liệu

- Màn hình duyệt cho phép đối chiếu câu hỏi, câu trả lời và đoạn nguồn. Sau khi chỉnh sửa, dùng nút lưu nháp hoặc gửi kết quả để lưu lên máy chủ.
- Toàn bộ tài liệu có thể được cung cấp từ `data/documents.json` hoặc từ cơ sở dữ liệu. Để mở PDF theo trang, cần PDF nguồn và ánh xạ trang.
- Bản xuất theo cấu trúc DeepEval dùng `metadata` và `goldens`. Giữ sự tương ứng giữa `context` và `source_chunk_ids` khi dùng bản xuất. Kiểm tra trạng thái và tiêu chí chọn mẫu trước khi đưa câu hỏi vào benchmark.
- [PLACEHOLDER: bổ sung phiên bản dữ liệu đang duyệt, ngày chốt, số câu theo trạng thái, quy tắc đồng thuận đã áp dụng và tệp xuất sau phân xử.]

## Cập nhật dữ liệu và triển khai

Từ thư mục gốc, đóng gói lại tài liệu khi nguồn thay đổi:

```bash
uv run scripts/prepare_review_documents.py \
  evaluation/eval_dataset_300_final_fixed.json \
  chunks_storage/chunks.parquet \
  review-web/data/documents.json
```

Từ `review-web`, dùng `uv run --env-file .env store.py --migrate` để cập nhật lược đồ và `uv run --env-file .env store.py --documents data/documents.json` nếu cần nhập nội dung vào cơ sở dữ liệu. Khi cấu hình triển khai, chọn thư mục gốc `review-web` và cung cấp `DATABASE_URL` của PostgreSQL, thông tin xác thực quản trị và URL công khai nếu cần chỉ định miền cho lời mời. Giữ secret ổn định để các lời mời hiện hành tiếp tục hoạt động.

[PLACEHOLDER: xác nhận môi trường triển khai đang hoạt động, phiên bản ứng dụng và đầu mối liên hệ trên trang quyền riêng tư.]

## Kiểm thử cục bộ

Từ thư mục `review-web`:

```bash
uv run python -m unittest discover -s tests -v
```
