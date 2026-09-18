# Canonical MSD report

Đây là report canonical của Sparrow insight local. Các tài liệu mở trực tiếp
ở thư mục này:

- `BAO_CAO_VIET.md` và `BAO_CAO_VIET.html`: báo cáo tiếng Việt.
- `REPORT.md`: báo cáo kỹ thuật.

Artifact được nhóm theo thí nghiệm để hình, statistics và audit liên quan ở
cùng một nơi:

```text
report_MSD/
├── figure1/       Figure 1(a/b): hình paper và hai statistics CSV
├── figure2/       Figure 2: hình, statistics, homogeneous cohort và audit riêng
├── figure3/       Figure 3(a/b): composite Qwen2.5-VL-3B, panels và statistics
├── figure6/       Figure 6: hình paper và statistics
├── audit/         Audit tổng thể của report
└── metadata/      summary.json và paper_statistics.json tổng hợp
```

Các run archive khác trong `results/` được giữ nguyên làm provenance. Figure
3(a) Qwen2.5-VL-3B đầy đủ có source bundle riêng tại
`figure3a_qwen25vl3b_20260821/full/`.
Figure 3(b) Qwen2.5-VL-3B all-layer có source bundle riêng tại
`../figure3b_qwen25vl3b_visual_attention_20260822/`; composite canonical và
presentation copies nằm trong `figure3/`.
`metadata/paper_statistics.json` hiện dùng Figure 3(b) 36-layer; bản metadata
legacy được giữ tại `metadata/paper_statistics_legacy_qwen2vl_20260819.json`.
