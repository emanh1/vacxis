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

## Nâng cấp thảo luận Calibration và hàng đợi Audit

Bản nâng cấp này dùng các bảng đang có, không thay đổi phân công hay cấp lại lời mời. Khi redeploy cùng project, giữ nguyên `DATABASE_URL`, `REVIEW_SECRET_KEY` và tên miền đang gửi cho bác sĩ. Tiến độ lưu trên Neon và phiên đăng nhập tiếp tục được sử dụng. Lời mời cũ vẫn tuân theo hạn dùng 7 ngày và quy tắc dùng một lần; nếu đã dùng, bác sĩ tiếp tục bằng phiên đăng nhập hiện có.

Không nạp lại dataset, khởi tạo lại 3 gói chuyên gia hay thu hồi phân công để cập nhật phiên bản này. Nếu database đã chạy phiên bản bốn bước đánh giá có `review_data`, không cần migration mới. Với database cũ hơn, dùng lệnh `store.py --migrate` ở phần trên; lệnh chỉ bổ sung cấu trúc còn thiếu. Có thể tải JSON kết quả hiện tại trước khi cập nhật để lưu một bản đối chiếu; JSON này không thay thế bản sao lưu toàn bộ database.

Sau khi cả ba chuyên gia đã nộp đủ 15 câu Calibration, trang quản lý hiện **Thống nhất Calibration** và một liên kết chung cho toàn bộ các câu **Cần đối chiếu**. Gửi liên kết cho nhóm chuyên gia; một người đại diện ghi kết luận sau thảo luận, không cần đăng nhập qua lời mời cá nhân. Liên kết cho phép xem ý kiến và chốt kết quả nên chỉ chia sẻ trong nhóm tham gia. Liên kết gắn với bộ dữ liệu, lượt duyệt thật/thử nghiệm và ba hồ sơ chuyên gia hiện tại.

Trong liên kết chung:

Nội dung xin phép mới (phiên bản 3) nêu rõ việc chia sẻ tên và vai trò với chuyên gia kiểm định và người có liên kết Calibration. Hồ sơ đã đồng ý phiên bản 2 vẫn giữ quyền xuất tên trong JSON; trên trang thảo luận dùng tên Expert cho đến khi người duyệt tự cập nhật tại **Bảo vệ thông tin cá nhân**. Không yêu cầu khai lại để tiếp tục chấm. Nơi công tác không xuất hiện trên trang thảo luận.

- **Đối chiếu các bản đánh giá** hiện đầy đủ nội dung và lí do sửa. Các bản giống nhau được gộp theo thứ tự Expert 1, 2, 3. Người không chia sẻ tên được hiển thị bằng số Expert.
- **Đoạn trích căn cứ** nằm ngay dưới mục đối chiếu, mặc định thu gọn. Nút **Toàn bộ tài liệu** mở khung rộng và hỗ trợ tìm kiếm, chuyển kết quả trước/tiếp như màn hình duyệt thường.
- Chọn **Đồng ý với ý kiến của** để lấy nội dung và lí do của chuyên gia; chọn **Ý kiến thống nhất khác** để tự chỉnh sửa phần đang có khác biệt. Phần cả ba đã đồng ý được giữ sẵn. **Ghi chú thống nhất** dành cho kết luận chung bổ sung.
- **Hoàn tất và Sang câu tiếp theo** lưu kết quả, chuyển đến câu chưa chốt tiếp theo trong cùng liên kết. Mở lại liên kết tiếp tục phần còn lại. Hai người chốt đồng thời cùng một câu sẽ được yêu cầu tải lại thay vì ghi đè kết quả của nhau. Các bản chấm độc lập được giữ nguyên.

Hàng đợi Audit được tính lại sau mỗi lần nộp, theo thứ tự:

1. Câu Calibration/Overlap chưa nộp, theo thứ tự các giai đoạn hiện có.
2. Câu chấm chính chưa nộp cần gửi người khác kiểm định.
3. Câu chấm chính bị kiểm định trả lại để sửa.
4. Câu Audit đã có bản nộp của người chấm chính.
5. Câu chấm chính thường chưa nộp.
6. Nếu không còn câu nào làm được, mở câu Audit đang **Chờ người chấm chính**.

Các câu Audit chưa sẵn sàng được bỏ qua cùng lúc, kể cả nhiều câu liên tiếp. Câu trước đó vừa sẵn sàng vẫn được chọn lại theo mức ưu tiên, không phụ thuộc vị trí câu vừa làm. Màn hình chờ tự kiểm tra mỗi 15 giây khi đang hiển thị và không mở khung toàn bộ tài liệu; nếu mất mạng, có thể tải lại trang khi kết nối phục hồi. Bản chưa nộp hoặc bị trả lại không thể được xác nhận đạt chuẩn bằng cách gửi yêu cầu trực tiếp.

Sau khi đưa code lên Vercel, kiểm tra lượt **Thử nghiệm nội bộ**: mở lại lời mời/phiên đang dùng, kiểm tra tiến độ và xem **Thống nhất Calibration** trên trang quản lý. Không dùng nút khởi tạo phân công để kiểm tra một lượt đang có người chấm.

## Kiểm thử cục bộ

Từ thư mục `review-web`:

```bash
uv run python -m unittest discover -s tests -v
```
