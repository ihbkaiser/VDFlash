# Qwen2.5-VL-3B Figure 3(a) experiment — 2026-08-21

Các artifact của cùng một thí nghiệm MVBench được nhóm theo mức chạy:

```text
figure3a_qwen25vl3b_20260821/
├── full/           Full run: 1,000 records, 10,000 scored rows
└── pilot/
    ├── initial/    Pilot ban đầu, giữ lại để provenance lỗi/runtime
    └── v2/         Pilot thành công: 25 records, 250 scored rows
```

Mỗi run giữ nguyên tên file gốc (`.jsonl`, `.summary.json`, `.png`, `.log`)
để đối chiếu lệnh chạy và provenance. Kết quả Figure 3(a) được dùng trong
report canonical là presentation copy tại `../report_MSD/figure3/`.
