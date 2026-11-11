# DFlash on Sparrow Experiments: Comprehensive Review & Roadmap

Thư mục này chứa tài liệu tổng hợp và đánh giá toàn diện các thực nghiệm đã thực hiện cho **DFlash** đối chiếu với các phát hiện và thiết kế thực nghiệm trong bài báo **Sparrow** (*ACL 2026*).

## Tài liệu chính

* [**`BAO_CAO_TONG_HOP_DFLASH_SPARROW_EXP.md`**](BAO_CAO_TONG_HOP_DFLASH_SPARROW_EXP.md): Báo cáo chi tiết gồm:
  1. **Ma trận đối chiếu tổng thể** giữa các thực nghiệm/hình vẽ của Sparrow và artifact tương ứng trong `results/`.
  2. **Chi tiết những gì ĐÃ LÀM ĐƯỢC** kèm bảng số liệu chính thức (H3.2 18.900 rows, H2.1 5.400 rows, Attention density, Layer-wise Cosine/Attention, H1.1 Representation Shift).
  3. **So sánh bản chất DFlash vs MSD** dưới góc nhìn các insight của Sparrow.
  4. **Kế hoạch hành động cụ thể cho những gì CẦN THỰC HIỆN THÊM** (bổ sung LongVideoBench, VideoMME, benchmark ảnh, Task accuracy).
  5. **Kết quả thực nghiệm mới đối đầu trực diện 1-1 (13/10/2026)**: Chạy trên 2 GPU (RTX 3090 + RTX A4000 = 40GB VRAM) cho DFlash vs MSD trên **VidHalluc** và **VideoChair** qua 4 mức retention (100%, 50%, 25%, 0%), kèm các biểu đồ trực quan trong `figures/`.
  6. **Chỉ mục truy vết artifact** trong workspace.
