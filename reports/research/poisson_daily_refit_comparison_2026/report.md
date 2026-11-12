> **Đã được thay thế:** rà soát protocol cho thấy dòng `poisson_90d_daily` trong báo cáo này đã lấy dự đoán từ Poisson Prize, không phải Poisson Count. Vì vậy không dùng báo cáo này để kết luận Count 90 ngày so với Count toàn lịch sử. Xem [báo cáo so sánh các biến thể Poisson 2026](../poisson_variants_2026_to_2026-11-05/report.md), được chạy lại cùng snapshot và giao thức.

# So sánh Poisson 90 ngày và Poisson mặc định — refit hằng ngày

**Kỳ đánh giá:** 2026-01-01–2026-10-29; 297 ngày ghép cặp.

**So sánh cùng giao thức:** hai biến thể dùng cùng `CountLGBMPredictor`, feature, Top-3, dữ liệu kết quả và cùng refit trước từng ngày dự báo. Poisson 90d chỉ dùng nhãn trong 90 ngày lịch trước ngày dự báo; Poisson mặc định dùng toàn bộ nhãn có trước ngày dự báo như khi chạy `main.py --model poisson --top-k 3` mỗi ngày. Không mô hình nào dùng nhãn của ngày dự báo hoặc tương lai.

**Tính tiền:** 27.000đ/số/ngày; tổng cược 81.000đ/ngày; trả gộp 99.000đ cho mỗi lượt xuất hiện, kể cả lặp.

## Tổng hợp

| Model | Ngày | Ngày hit | Tỷ lệ hit | Lượt | Chi | Thu | Lãi/lỗ | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| poisson_90d_daily | 297 | 179 | 60.27% | 279 | 24,057,000đ | 27,621,000đ | +3,564,000đ | 14.81% |
| poisson_all_history_daily | 297 | 169 | 56.90% | 260 | 24,057,000đ | 25,740,000đ | +1,683,000đ | 7.00% |

## Chi tiết từng tháng

| Tháng | Ngày cược / ngày trong kỳ | Bị loại | Poisson 90d: hit (tỷ lệ); lượt; lãi/lỗ (ROI) | Poisson mặc định: hit (tỷ lệ); lượt; lãi/lỗ (ROI) | Chênh lãi 90d − mặc định |
|---|---:|---:|---|---|---:|
| 2026-01 (31) | 31/31 | 0 | 15/31 (48.39%); 31; +558,000đ (22.22%) | 15/31 (48.39%); 19; -630,000đ (-25.09%) | +1,188,000đ |
| 2026-02 (28) | 23/28 | 5 | 9/23 (39.13%); 12; -675,000đ (-36.23%) | 12/23 (52.17%); 18; -81,000đ (-4.35%) | -594,000đ |
| 2026-03 (31) | 31/31 | 0 | 22/31 (70.97%); 31; +558,000đ (22.22%) | 12/31 (38.71%); 19; -630,000đ (-25.09%) | +1,188,000đ |
| 2026-04 (30) | 30/30 | 0 | 22/30 (73.33%); 39; +1,431,000đ (58.89%) | 20/30 (66.67%); 32; +738,000đ (30.37%) | +693,000đ |
| 2026-05 (31) | 31/31 | 0 | 19/31 (61.29%); 28; +261,000đ (10.39%) | 22/31 (70.97%); 40; +1,449,000đ (57.71%) | -1,188,000đ |
| 2026-06 (30) | 30/30 | 0 | 16/30 (53.33%); 23; -153,000đ (-6.30%) | 16/30 (53.33%); 29; +441,000đ (18.15%) | -594,000đ |
| 2026-07 (31) | 31/31 | 0 | 19/31 (61.29%); 31; +558,000đ (22.22%) | 15/31 (48.39%); 25; -36,000đ (-1.43%) | +594,000đ |
| 2026-08 (31) | 31/31 | 0 | 18/31 (58.06%); 24; -135,000đ (-5.38%) | 22/31 (70.97%); 30; +459,000đ (18.28%) | -594,000đ |
| 2026-09 (30) | 30/30 | 0 | 16/30 (53.33%); 29; +441,000đ (18.15%) | 16/30 (53.33%); 22; -252,000đ (-10.37%) | +693,000đ |
| 2026-10 (31) | 29/29 | 0 | 23/29 (79.31%); 31; +720,000đ (30.65%) | 19/29 (65.52%); 26; +225,000đ (9.58%) | +495,000đ |

**Chênh lãi cộng dồn 90d − mặc định: +1,881,000đ.** Đây là kết quả hồi cứu, không phải bảo đảm lợi thế tương lai.

SHA-256 nguồn: `a032215a5a38838b19fadad80da0539320d300dfaf75b9e545f9893801e2eaf1`. [Dữ liệu ngày](./daily.csv) · [Theo tháng](./monthly.csv) · [Tổng hợp](./summary.csv) · [Ngày bị loại](./excluded_dates.csv) · [Giao thức](./protocol.json).
