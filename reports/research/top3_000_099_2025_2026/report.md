# Kiểm định Top-3 chỉ trong 000–099

## Câu hỏi và protocol

- Mỗi ngày vẫn chọn đúng 3 số ba chữ số, nhưng chiến lược hạn chế chỉ được xếp hạng `000`–`099`.
- Đối chứng chọn Top-3 từ đủ `000`–`999` bằng chính cùng điểm số mô hình trên cùng ngày. Vì vậy phép so sánh ghép cặp đo riêng tác động của giới hạn không gian chọn số.
- Target gồm 23 đuôi ba chữ số mỗi ngày, loại bốn Giải bảy hai chữ số, giữ số 0 đầu và các lần lặp.
- Walk-forward theo quý, cửa sổ lịch sử 1095 ngày; nhãn huấn luyện cuối luôn trước ngày kiểm thử đầu của quý.
- Kỳ kiểm thử: **2025-01-01 đến 2026-11-04** (663 ngày có đủ ngày liền trước).
- Nhóm `000–099` chiếm **1,516/15,249 = 9.94%** lượt kết quả thực tế; tỷ lệ đồng đều kỳ vọng là 10%.
- Xác suất hit trung bình nếu chọn ngẫu nhiên 3 số, có điều kiện trên kết quả thực tế: toàn bộ `000–999` **6.68%**; chỉ `000–099` **6.65%**.
- P&L dùng cùng giả định: 23 đơn vị/số/ngày, trả gộp 900 đơn vị cho mỗi lượt xuất hiện.

## Kết quả toàn kỳ

| Mô hình | Không gian chọn | Ngày hit (block 95%) | Lượt trúng | Miss tối đa | Lãi ròng | ROI |
|---|---|---:|---:|---:|---:|---:|
| poisson | all_000_999 | 45/663 (6.79%; 4.83%–8.90%) | 47 | 58 | -3,447 | -7.53% |
| poisson | restricted_000_099 | 54/663 (8.14%; 5.88%–10.56%) | 57 | 52 | +5,553 | +12.14% |
| binary | all_000_999 | 43/663 (6.49%; 4.83%–8.30%) | 47 | 47 | -3,447 | -7.53% |
| binary | restricted_000_099 | 44/663 (6.64%; 4.83%–8.60%) | 46 | 47 | -4,347 | -9.50% |
| frequency_90d | all_000_999 | 44/663 (6.64%; 4.83%–8.60%) | 46 | 49 | -4,347 | -9.50% |
| frequency_90d | restricted_000_099 | 44/663 (6.64%; 4.68%–8.60%) | 45 | 62 | -5,247 | -11.47% |

## So sánh ghép cặp: giới hạn trừ không giới hạn

| Mô hình | 000–099 | 000–999 | Chênh hit-rate (block 95%) | Chênh lãi ròng |
|---|---:|---:|---:|---:|
| poisson | 54/663 (8.14%) | 45/663 (6.79%) | +1.36% (-1.06%–+3.77%) | +9,000 |
| binary | 44/663 (6.64%) | 43/663 (6.49%) | +0.15% (-2.26%–+2.56%) | -900 |
| frequency_90d | 44/663 (6.64%) | 44/663 (6.64%) | +0.00% (-2.26%–+2.26%) | -900 |

Khoảng tin cậy dùng circular block bootstrap 7 ngày, 5.000 lần lấy mẫu. Đây là bất định hồi cứu, không hiệu chỉnh cho việc thử nhiều ý tưởng.

## Kiểm định bổ sung

- Tỷ trọng thực tế của nhóm `0xx` so với mốc 10%: binomial hai phía **p=0.819**.
- Poisson giới hạn so với chọn ngẫu nhiên 3 số trong `000–099`, có điều kiện trên từng ngày thực tế: Poisson-binomial một phía **p=0.072**.
- So sánh hit ghép cặp Poisson: giới hạn thắng riêng 39 ngày, không giới hạn thắng riêng 30 ngày; McNemar exact hai phía **p=0.336**.

## Theo năm

| Năm | Mô hình | Không gian chọn | Ngày hit | Lượt trúng | Lãi ròng |
|---:|---|---|---:|---:|---:|
| 2025 | poisson | all_000_999 | 25/360 (6.94%) | 26 | -1,440 |
| 2026 | poisson | all_000_999 | 20/303 (6.60%) | 21 | -2,007 |
| 2025 | poisson | restricted_000_099 | 28/360 (7.78%) | 30 | +2,160 |
| 2026 | poisson | restricted_000_099 | 26/303 (8.58%) | 27 | +3,393 |
| 2025 | binary | all_000_999 | 23/360 (6.39%) | 26 | -1,440 |
| 2026 | binary | all_000_999 | 20/303 (6.60%) | 21 | -2,007 |
| 2025 | binary | restricted_000_099 | 21/360 (5.83%) | 22 | -5,040 |
| 2026 | binary | restricted_000_099 | 23/303 (7.59%) | 24 | +693 |
| 2025 | frequency_90d | all_000_999 | 27/360 (7.50%) | 29 | +1,260 |
| 2026 | frequency_90d | all_000_999 | 17/303 (5.61%) | 17 | -5,607 |
| 2025 | frequency_90d | restricted_000_099 | 23/360 (6.39%) | 23 | -4,140 |
| 2026 | frequency_90d | restricted_000_099 | 21/303 (6.93%) | 22 | -1,107 |

## Top-3 của ngày kiểm thử cuối

| Mô hình | Không gian chọn | Top-3 |
|---|---|---|
| poisson | all_000_999 | 294 124 194 |
| poisson | restricted_000_099 | 014 074 024 |
| binary | all_000_999 | 114 294 774 |
| binary | restricted_000_099 | 054 084 094 |
| frequency_90d | all_000_999 | 077 188 200 |
| frequency_90d | restricted_000_099 | 077 040 053 |

Đây là dự báo hồi cứu trước kết quả ngày 2026-11-04, không phải dự báo hiện tại.

## Kết luận

Chưa có bằng chứng giới hạn 000–099 làm tăng hit-rate Poisson; khoảng chênh lệch bootstrap vẫn chứa 0. Kết luận chính phải dựa vào hàng Poisson ghép cặp và độ ổn định theo năm, không dựa vào một tháng hoặc một lần trúng riêng lẻ.
