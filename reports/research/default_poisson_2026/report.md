# Poisson toàn lịch sử cố định — holdout năm 2026

**Kỳ đánh giá:** 01/01/2026–2026-11-04; 303 ngày hợp lệ, 5 ngày bị loại.

**Giao thức:** `CountLGBMPredictor` mặc định của `main.py --model poisson --top-k 3`, fit đúng một lần bằng toàn bộ nhãn lịch sử có trước 01/01/2026 (nhãn huấn luyện đến 31/12/2025). Trong năm 2026, feature được cập nhật theo kết quả ngày liền trước, nhưng không huấn luyện lại model. Đây là holdout một năm để kiểm định ngoài mẫu; **không tương đương báo cáo Poisson cửa sổ 90 ngày refit hằng ngày, cũng không phải replay lệnh production chạy lại mỗi ngày.**

**Tính tiền:** chọn 3 số/ngày; cược 27.000đ/số (81.000đ/ngày); trả gộp 99.000đ cho mỗi lần xuất hiện, kể cả lặp.

## Tổng kết

| Ngày | Ngày hit | Tỷ lệ hit | Lượt trúng | Tổng chi | Tổng thu | Lãi/lỗ | ROI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 303 | 173 | 57.10% | 239 | 24,543,000đ | 23,661,000đ | -882,000đ | -3.59% |

## Chi tiết từng tháng

| Tháng | Ngày cược / ngày trong kỳ | Bị loại | Ngày hit | Tỷ lệ hit | Lượt trúng | Chi | Thu | Lãi/lỗ | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01 (31) | 31/31 | 0 | 16/31 | 51.61% | 24 | 2,511,000đ | 2,376,000đ | -135,000đ | -5.38% |
| 2026-02 (28) | 23/28 | 5 | 11/23 | 47.83% | 17 | 1,863,000đ | 1,683,000đ | -180,000đ | -9.66% |
| 2026-03 (31) | 31/31 | 0 | 16/31 | 51.61% | 19 | 2,511,000đ | 1,881,000đ | -630,000đ | -25.09% |
| 2026-04 (30) | 30/30 | 0 | 24/30 | 80.00% | 34 | 2,430,000đ | 3,366,000đ | +936,000đ | 38.52% |
| 2026-05 (31) | 31/31 | 0 | 15/31 | 48.39% | 20 | 2,511,000đ | 1,980,000đ | -531,000đ | -21.15% |
| 2026-06 (30) | 30/30 | 0 | 15/30 | 50.00% | 21 | 2,430,000đ | 2,079,000đ | -351,000đ | -14.44% |
| 2026-07 (31) | 31/31 | 0 | 18/31 | 58.06% | 28 | 2,511,000đ | 2,772,000đ | +261,000đ | 10.39% |
| 2026-08 (31) | 31/31 | 0 | 19/31 | 61.29% | 25 | 2,511,000đ | 2,475,000đ | -36,000đ | -1.43% |
| 2026-09 (30) | 30/30 | 0 | 17/30 | 56.67% | 24 | 2,430,000đ | 2,376,000đ | -54,000đ | -2.22% |
| 2026-10 (31) | 31/31 | 0 | 20/31 | 64.52% | 25 | 2,511,000đ | 2,475,000đ | -36,000đ | -1.43% |
| 2026-11 (30) | 4/4 | 0 | 2/4 | 50.00% | 2 | 324,000đ | 198,000đ | -126,000đ | -38.89% |

Ngày thiếu kết quả thực tế hoặc thiếu kết quả ngày liền trước bị loại, không tính là ngày cược/thua. Đây là kết quả hồi cứu, không đảm bảo hiệu quả tương lai.

SHA-256 nguồn: `4819b779c23d7969ba6c0e1a17b7a1249615dc35ced1f8ae45ea114ba5791922`. [Dữ liệu ngày](./daily.csv) · [Theo tháng](./monthly.csv) · [Ngày bị loại](./excluded_dates.csv) · [Giao thức](./protocol.json).
