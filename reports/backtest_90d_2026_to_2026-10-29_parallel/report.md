# Lãi/lỗ Top 1–6 theo mốc 90 ngày từ đầu năm 2026

Giai đoạn 2026-01-01 đến 2026-10-29. Giá mua 23.000đ/số/ngày; tổng nhận 80.000đ mỗi lần xuất hiện, tính cả lặp. Cùng thứ hạng LightGBM, huấn luyện lại trước từng ngày bằng toàn bộ lịch sử đã có. Không thay tham số hay chọn K theo kết quả.

90 ngày là độ dài khoảng báo cáo, không phải giới hạn lịch sử huấn luyện. Khoảng cuối có thể chưa đủ 90 ngày. Các ngày thiếu kết quả hoặc thiếu ngày liền trước bị loại khỏi phép đánh giá, không được coi là ngày trượt hay tự động bỏ mua.

297 ngày đủ dữ liệu / 302 ngày lịch. Ngày bị loại: 2026-02-16 (actual outcome missing); 2026-02-17 (actual outcome missing); 2026-02-18 (actual outcome missing); 2026-02-19 (actual outcome missing); 2026-02-20 (preceding calendar day missing).

## Lãi/lỗ (đồng)

| Khoảng | Ngày đánh giá/ngày lịch | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 | Top 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01/01–31/03 | 85/90 | -515,000 | -310,000 | -25,000 | -380,000 | -895,000 | -850,000 |
| 01/04–29/06 | 90/90 | +650,000 | -220,000 | -690,000 | -920,000 | -670,000 | -740,000 |
| 30/06–27/09 | 90/90 | -70,000 | -300,000 | -690,000 | -1,080,000 | -1,390,000 | -1,140,000 |
| 28/09–29/10 | 32/32 | -96,000 | +128,000 | +112,000 | -304,000 | -400,000 | -896,000 |
| Tổng | 297/302 | -31,000 | -702,000 | -1,293,000 | -2,684,000 | -3,355,000 | -3,626,000 |

## Tổng tiền và tỷ lệ

| Top K | Chi | Nhận | Lãi/lỗ | ROI | Lần xuất hiện | Ngày có lãi | Sụt giảm vốn lớn nhất |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,831,000 | 6,800,000 | -31,000 | -0.45% | 85 | 77/297 | 621,000 |
| 2 | 13,662,000 | 12,960,000 | -702,000 | -5.14% | 162 | 130/297 | 968,000 |
| 3 | 20,493,000 | 19,200,000 | -1,293,000 | -6.31% | 240 | 174/297 | 1,577,000 |
| 4 | 27,324,000 | 24,640,000 | -2,684,000 | -9.82% | 308 | 82/297 | 3,000,000 |
| 5 | 34,155,000 | 30,800,000 | -3,355,000 | -9.82% | 385 | 101/297 | 3,570,000 |
| 6 | 40,986,000 | 37,360,000 | -3,626,000 | -8.85% | 467 | 146/297 | 3,772,000 |

## Diễn giải và kiểm tra

- Đây là backtest hồi cứu. Các khoảng đã được dùng trong nghiên cứu trước; không phải tập kiểm tra hoàn toàn mới. Không suy ra K tối ưu cho tương lai từ K có lãi cao nhất trong bảng.
- Không có ngày cần dự báo trong nhãn huấn luyện của chính ngày đó. Đặc trưng tại t chỉ dùng kết quả đến t, dự báo t+1.
- Khi chọn K số đều ngẫu nhiên, kỳ vọng số lần xuất hiện là 0,27 mỗi lượt mua; kỳ vọng lỗ 1.400đ/lượt với mức trả trên. Đây là kỳ vọng, không phải kết quả của một danh sách ngẫu nhiên cụ thể.
- Tổng tiền trong bảng chỉ cho các ngày có thể đánh giá. Không thể suy ra dòng tiền thực tế của những ngày bị thiếu kết quả.

[Tổng hợp đầy đủ](./summary.csv) · [Từng ngày](./daily.csv) · [Giao thức](./protocol.json) · [Ngày bị loại](./excluded_dates.json) · [Mốc nhãn huấn luyện](./training_cutoffs.csv)
