# Sổ dự báo giả lập XSMB

Thời gian: **2026-10-30 đến 2027-01-27** (90 ngày lịch).
Phương án chính: **Poisson Top 5**. Binary và Regularized chỉ là đối chứng. Không dùng tiền thật.

Đã ghi 1 ngày; chờ kết quả 1; bỏ lỡ dự báo 0 ngày.

| Mô hình | Ngày đã chấm | Vốn giả lập | Thu về | Lãi/lỗ | Sụt giảm tối đa |
|---|---:|---:|---:|---:|---:|
| prize_poisson | 0 | 0 | 0 | +0 | 0 |
| prize_binary | 0 | 0 | 0 | +0 | 0 |
| regularized | 0 | 0 | 0 | +0 | 0 |

Mỗi mô hình là một tài khoản giả lập riêng. Không cộng vốn/lợi nhuận ba mô hình thành một chiến lược.
Ngày chưa có kết quả và ngày bỏ lỡ dự báo không được tính thành thua. Báo cáo tỷ lệ bao phủ riêng.
Sổ dùng đồng hồ máy và chuỗi băm nội bộ; không phải chứng nhận thời gian độc lập. Báo cáo CSV/JSON có thể tái tạo, không phải sổ gốc.

[Kết quả từng ngày](./daily.csv) · [Bao phủ 90 ngày](./coverage.csv) · [Chi tiết](./summary.json)

## Dự báo đã khóa gần nhất: 2026-10-30

- prize_poisson: **45 · 86 · 96 · 75 · 91**
- prize_binary: **16 · 96 · 90 · 86 · 91**
- regularized: **91 · 14 · 52 · 34 · 69**
