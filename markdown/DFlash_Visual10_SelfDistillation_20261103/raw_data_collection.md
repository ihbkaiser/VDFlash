# Thu thập dữ liệu thô cho post-training DFlash 10% visual

Ngày bắt đầu: 2026-11-04. Phạm vi: tải video archives và original annotations; kiểm tra nguồn và lập catalog. Target responses, hidden states, logits và feature caches chưa được tạo.

## Nguồn và vị trí

- Dataset nguồn: [lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K).
- Revision ghim: 6d8c562dc26d70042a0d9704d1cae58c94b89098.
- Seed chọn archive: 20261104.
- Thư mục raw: data/dflash_posttraining/raw/llava_video_178k, tính từ repository root.
- Pool dự kiến: 8 archive video gốc, 12 JSON caption/open-ended QA và dataset card, tổng khoảng 44,04 GB.

| Nhóm thời lượng theo nguồn | YouTube archives | Academic archives |
|---|---:|---:|
| 0–30 giây | 1 | 0 |
| 30–60 giây | 1 | 0 |
| 1–2 phút | 2 | 1 |
| 2–3 phút | 2 | 1 |

Các shard được chọn ngẫu nhiên với seed cố định trong từng nhóm; đây là một raw pool, chưa phải mẫu ngẫu nhiên đều trên toàn bộ video của dataset. Số shard ưu tiên long video không tự bảo đảm tỷ lệ video hoặc records dài/ngắn sau đóng gói.

JSON gốc bao phủ toàn bộ duration/source group, còn video chỉ thuộc các archive đã chọn. Do đó không coi toàn bộ annotation rows tải về là records có video local. Catalog sẽ join theo video key và lưu vị trí row trong file gốc.

Thời lượng nguồn 1–3 phút chưa xác nhận visual budget 13K–25K. Cần calibration bằng đúng processor, sampling frames và resolution khi chuẩn bị training. Prompt marker nguồn có thể là <image> dù record trỏ tới video; bước chuyển đổi sau phải dùng đúng video message contract của Qwen2.5-VL.

## Công cụ và resume

Job nền đang chạy tải và tự catalog khi hoàn tất. Theo dõi bằng:

~~~bash
tail -f data/dflash_posttraining/raw/llava_video_178k/collection.log
cat data/dflash_posttraining/raw/llava_video_178k/pipeline_status.json
~~~

pipeline_status.json có status=complete và stage=raw_collection_verified khi tải, checksum và catalog đều đã xong. download_status.json ghi chi tiết file. collection_summary.json và SHA256SUMS chỉ được tạo ở các bước sau khi tải đủ. Trạng thái running không có nghĩa raw pool đã sẵn sàng.

Để resume toàn bộ pipeline từ repository root, khi job trước đã dừng:

~~~bash
python -u scripts/collect_posttraining_raw.py --workers 4 --range-workers 16
~~~

Lệnh chỉ tải raw, không tự catalog:

~~~bash
python -u scripts/download_posttraining_raw.py --workers 4 --range-workers 16
~~~

Script giữ download_plan.json và source_inventory.json với revision, kích thước và SHA256 từ Hugging Face LFS. File hoàn tất chỉ được công bố sau khi khớp kích thước và checksum. File .part và .segments là các phần truyền dữ liệu để resume, không phải feature cache; chỉ dọn transfer chunks của chính file đó sau khi đã xác minh toàn bộ file.

download_status.json ghi các file đã xác minh và lỗi nếu có. Chạy lại cùng lệnh sẽ kiểm tra lại các file hoàn tất và tiếp tục các phần tải dở. Chỉ chạy một downloader trên cùng thư mục tại một thời điểm.

Sau khi tất cả file được tải và xác minh:

~~~bash
python -u scripts/catalog_posttraining_raw.py \
  --exclude-vdc-metadata dataset/VideoDetailCaption/source_metadata.parquet
~~~

Catalog tạo raw_video_inventory.jsonl và collection_summary.json, quét gzip/tar và liên kết original annotation refs. Catalog không giải nén hoặc chuyển mã video. Nó kiểm tra cấu trúc archive, chưa kiểm tra decode toàn bộ video.

Các video trùng ID VDC được đánh dấu excluded_vdc; raw source được giữ nguyên. Đối chiếu ID chỉ phát hiện trùng ID sau chuẩn hóa ytb_/v_, chưa phát hiện mọi clip gần trùng hoặc cùng nội dung ở URL khác. Metadata VDC hiện có dùng làm exclusion source, không đưa vào training.

## Các bước sau thu thập

1. Xem collection_summary.json để biết số unique videos và số annotation refs có video local, gồm coverage caption/QA và overlap VDC.
2. Chọn khoảng 3.000 records training với trọng tâm context dài, cộng validation/test trên video gốc riêng biệt. Split theo video/source trước khi nhân prompts hoặc budget variants.
3. Xây manifest chuẩn của Qwen2.5-VL, ghi rõ source revision, video key, annotation row index, task và vị trí media. Câu trả lời nguồn tiếp tục được giữ cho provenance; target responses sẽ được sinh ở bước riêng.
4. Probe/decode các video được chọn và calibrate visual token budgets; lọc hoặc ghi nhận video lỗi theo lý do cụ thể.
5. Sau khi dataset đã được chọn và kiểm tra, mới chuẩn bị target responses/features và huấn luyện theo [phương pháp hai nguồn supervision](research_proposal.md).

Raw data nằm trong thư mục data/ bị Git ignore. Các file checkpoint, cache cũ và artifact thực nghiệm trước không bị thay đổi.
