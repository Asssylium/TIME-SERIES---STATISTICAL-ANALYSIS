# Kiểm định dài hạn Top-3 số hai chữ số: tín hiệu tần suất 90 ngày

## Phát hiện sơ bộ

- Walk-forward có **5,774 ngày dự báo**, từ **2010-09-30** đến **2026-11-04**; mỗi dự báo chỉ dùng kết quả trước ngày mục tiêu.
- Top-3 tần suất 90 ngày đạt **0.8424 lượt/ngày** khi hòa điểm được chia ngẫu nhiên công bằng; mốc chọn đều là **0.8100** (27 kết quả × 3/100). Chênh lệch: **+0.0324 lượt/ngày**.
- Khoảng block-bootstrap 95% cho chênh lệch với mốc đều: **+0.0097 đến +0.0544** (khối 7 ngày) và **+0.0064 đến +0.0590** (khối 90 ngày).
- Tỷ lệ ngày hit là **57.22%** so với **56.18%** của chọn ngẫu nhiên; chênh lệch block-bootstrap 95% theo khối 7 ngày là **-0.20% đến +2.20%**, còn chứa 0.
- So với tần suất 5 năm, chênh lệch ghép cặp là **+0.0297 lượt/ngày**, block-bootstrap 95% (khối 90 ngày) **+0.0026 đến +0.0587**; khoảng này không chứa 0 ở mức bootstrap chưa hiệu chỉnh.

Đây là **ứng viên cần xác nhận**, chưa phải đột phá đã được chứng minh. Mức tăng lợi nhuận suy ra từ giả định trả 99.000đ mỗi lần xuất hiện và chi 27.000đ mỗi số là xấp xỉ **2,395đ/ngày** trong hồi cứu; không được xem là kỳ vọng tương lai đã xác nhận.

## So sánh các chiến lược

| strategy | days | expected occurrences per day | delta occurrences vs uniform | occurrence delta block95 low 7d | occurrence delta block95 high 7d | expected hit rate | random hit rate conditional | stable tie occurrences per day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frequency_90d | 5774 | 0.8424 | +0.0324 | +0.0110 | +0.0558 | 57.2247% | 56.1827% | 0.8467 |
| frequency_365d | 5774 | 0.8121 | +0.0021 | -0.0194 | +0.0243 | 56.6237% | 56.1827% | 0.8109 |
| frequency_5y | 5774 | 0.8126 | +0.0026 | -0.0205 | +0.0250 | 56.5004% | 56.1827% | 0.8116 |
| expanding | 5774 | 0.8207 | +0.0107 | -0.0115 | +0.0331 | 56.4138% | 56.1827% | 0.8194 |
| weekday_5y | 5774 | 0.8198 | +0.0098 | -0.0134 | +0.0323 | 57.0136% | 56.1827% | 0.8218 |
| weekday_shrink_3y | 5774 | 0.8129 | +0.0029 | -0.0196 | +0.0257 | 56.5812% | 56.1827% | 0.8133 |

Tần suất 90 ngày tốt nhất trong nhóm so sánh, nhưng chiến lược và ngưỡng 90 ngày không được đăng ký trước lần xem dữ liệu này. Các khoảng tin cậy ở bảng là mô tả, chưa hiệu chỉnh cho thử nhiều chiến lược hoặc việc tái sử dụng dữ liệu.

## Kiểm tra ổn định theo thời kỳ

| era | days | expected occurrences per day |
| --- | --- | --- |
| 2010-2019 | 3332 | 0.8374 |
| 2020-2022 | 1058 | 0.8441 |
| 2023-2026 | 1384 | 0.8529 |

Mức trung bình dương qua ba thời kỳ làm giả thuyết tần suất ngắn hạn đáng kiểm tra tiếp, nhưng không tạo ra ba tập xác nhận độc lập: cửa sổ huấn luyện chồng lấn và toàn bộ lịch sử đã từng được xem.

## Thiết kế và giới hạn

- Nguồn: `D:\TIME-SERIES & STATISTICAL ANALYSIS\data\raw\xsmb_2026-11-05.csv`; SHA-256 `4819b779c23d7969ba6c0e1a17b7a1249615dc35ced1f8ae45ea114ba5791922`; 5,774 ngày dự báo.
- Dữ liệu là 27 kết quả hai chữ số mỗi ngày, giữ nguyên các lần lặp. Huấn luyện có tối thiểu 1.825 ngày lịch; chỉ dự báo ngày liền sau một ngày có kết quả.
- Đối chứng occurrences ngẫu nhiên chính xác là 0,81/ngày, không cần giả định 27 kết quả độc lập: mỗi ngày chọn 3 số đều nhau từ 100 số.
- Mốc hit ngẫu nhiên được tính có điều kiện theo số lượng số khác nhau thực tế trong ngày, không dùng xấp xỉ nhị thức.
- Chỉ số `expected_*_random_ties` tính kỳ vọng chính xác nếu chọn ngẫu nhiên trong nhóm hòa điểm ở ngưỡng Top-3. Cột `stable_tie_*` ghi quy tắc vận hành xác định: ưu tiên số nhỏ hơn.
- Các khoảng bootstrap dùng khối tròn theo ngày dự báo; khoảng không xử lý việc chọn mô hình sau khi xem kết quả, cũng không chứng minh tính ngẫu nhiên hoặc lợi thế tương lai.
- Kiểm toán trước đây đã thử nhiều giả thuyết tần suất/thứ trong tuần/tự tương quan trên cùng nguồn. Đây không phải holdout mới. Không thay production; cần khóa dự báo tiền cứu, paper-trial ít nhất 6–12 tháng và không tinh chỉnh trong thời gian đó.

## Tệp tái lập

- [Giao thức](./protocol.json) · [Dữ liệu theo ngày](./daily.csv) · [Tổng hợp](./summary.csv) · [Theo năm](./yearly.csv) · [Theo thời kỳ](./eras.csv).
