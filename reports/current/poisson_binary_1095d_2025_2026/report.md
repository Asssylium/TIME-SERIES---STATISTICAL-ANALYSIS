# Báo cáo đã hiệu chỉnh: Top-3 Poisson và Binary, cửa sổ 1095 ngày

> Báo cáo này thay thế bản trước. Đặc trưng nhân quả được dựng từ cùng một mốc lịch sử nguồn cố định, nên kết quả của một quý không đổi khi replay riêng quý đó.

Phạm vi: chọn 3 số trong `00–99`, chấm trên đủ 27 kết quả mỗi ngày, giữ nháy lặp; tái huấn luyện trước mỗi quý. Không có mô hình 23 giải và không có Frequency 90d trong báo cáo này.

| Giai đoạn | Mô hình | Ngày hit | Tỷ lệ hit (block 95%) | Lượt trúng | Lãi/lỗ | ROI |
|---|---|---:|---:|---:|---:|---:|
| 2025 | Poisson | 201/360 | 55,83% (50,56%–61,11%) | 290 | -450.000đ | -1,54% |
| 2025 | Binary | 201/360 | 55,83% (51,11%–60,56%) | 293 | -153.000đ | -0,52% |
| 2026 | Poisson | 167/297 | 56,23% (51,18%–61,28%) | 260 | +1.683.000đ | 7,00% |
| 2026 | Binary | 164/297 | 55,22% (49,16%–61,28%) | 264 | +2.079.000đ | 8,64% |
| all | Poisson | 368/657 | 56,01% (52,21%–59,67%) | 550 | +1.233.000đ | 2,32% |
| all | Binary | 365/657 | 55,56% (51,90%–59,21%) | 557 | +1.926.000đ | 3,62% |

## So sánh ghép cặp toàn kỳ

Binary − Poisson về tỷ lệ ngày hit: **-0,46%** (block 95% -4,87% đến 3,81%; McNemar p=0.897).

[Dữ liệu ngày](./daily.csv) · [Tổng hợp](./summary.csv) · [So sánh ghép cặp](./paired_comparison.csv) · [Giao thức](./protocol.json)
