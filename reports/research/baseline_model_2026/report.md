# Kiểm tra cách tính và nghiên cứu mô hình Top 5

## Kết luận theo cấu hình đã chọn trước khi xem tập kiểm tra

| Trả thưởng | Cấu hình chọn trên phát triển | Lãi/lỗ kiểm tra | ROI | Khoảng 95% lãi/ngày | Có bằng chứng kỳ vọng dương? |
|---|---|---:|---:|---|---|
| once_per_day | window_14 | -4,890,000đ | -20.64% | [-31,893; -15,194]đ | Chưa có |
| per_occurrence | window_30 | -1,850,000đ | -7.81% | [-20,243; 2,282]đ | Chưa có |

## Cách tính

- Mỗi ngày chọn 5 số, chi 5 × 23.000 = 115.000đ. 80.000đ là tổng nhận, không cộng hoàn vốn.
- once_per_day: số trúng khác nhau × 80.000 − tiền mua.
- per_occurrence: tổng số lần xuất hiện của các số đã chọn × 80.000 − tiền mua.
- Hòa vốn: 0,2875 đơn vị trả thưởng/lượt chọn. Với per_occurrence đây là kỳ vọng số lần, không phải xác suất xuất hiện ít nhất một lần.
- Chọn ngẫu nhiên 5 số có kỳ vọng 1,35 lần xuất hiện/ngày (27 × 5/100), thu 108.000đ, lỗ kỳ vọng 7.000đ/ngày nếu trả mỗi lần. Không cần giả định 27 kết quả độc lập để tính kỳ vọng này khi chọn 5 số đều nhau.

## Thiết kế nghiên cứu

- 14 cấu hình cố định: 6 cửa sổ tần suất, lịch sử tích lũy, 2 EWMA, Markov làm trơn, Logistic Regression, LightGBM cũ, LightGBM điều chuẩn, Poisson.
- Phát triển/chọn: 2024-01-01 đến 2025-06-30, chia 3 khối nửa năm. Chọn theo tổng đơn vị trả thưởng/lượt chọn trên phát triển; không dùng tập kiểm tra để chọn.
- Hiệu chỉnh xác suất: khớp sigmoid ở quý 3/2025; chỉ giữ nếu Brier tốt hơn và độ dốc dương trên quý 4/2025.
- Kiểm tra khóa: 2026-01-01 đến 2026-07-30. Giai đoạn 2026-07-31 trở đi là kiểm tra hồi cứu đã từng xem trước đó, không phải tập độc lập mới.
- Mỗi mô hình ML huấn luyện trước từng khối, không huấn luyện lại bằng nhãn trong khối. Đặc trưng và baseline cập nhật sau mỗi kết quả đã biết. Đây là cách đánh giá theo khối, khác lượt chạy cũ huấn luyện ML lại hàng ngày.
- Đây là train/development/calibration/test theo thời gian, không tuyên bố nested CV đầy đủ. Dữ liệu lịch sử từng được dùng huấn luyện trong các lượt trước nên vẫn cần kiểm chứng dự báo ghi trước kết quả thực tế.
- Khoảng bất định dùng bootstrap theo khối 7 ngày, 2.000 mẫu, giữ nguyên cụm 100 số/ngày. Khoảng chỉ có tính xấp xỉ dưới giả định ổn định cục bộ; không bảo đảm cho tương lai.

## Tập kiểm tra khóa: mọi cấu hình (chỉ mang tính chẩn đoán, không chọn lại theo bảng này)

| Mô hình | Trả thưởng | Số ngày | Precision@5 | Lần xuất hiện/lượt chọn | Lãi/lỗ | ROI | Brier thô | Brier sau hiệu chỉnh |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| window_7 | once_per_day | 206 | 23.79% | 0.2709 | -4,090,000đ | -17.26% | 0.182338 | 0.182338 |
| window_7 | per_occurrence | 206 | 22.72% | 0.2660 | -1,770,000đ | -7.47% | 0.182338 | 0.182338 |
| window_14 | once_per_day | 206 | 22.82% | 0.2650 | -4,890,000đ | -20.64% | 0.182581 | 0.181187 |
| window_14 | per_occurrence | 206 | 22.72% | 0.2612 | -2,170,000đ | -9.16% | 0.182581 | 0.181187 |
| window_30 | once_per_day | 206 | 24.95% | 0.2854 | -3,130,000đ | -13.21% | 0.182560 | 0.181190 |
| window_30 | per_occurrence | 206 | 23.59% | 0.2650 | -1,850,000đ | -7.81% | 0.182560 | 0.181190 |
| window_60 | once_per_day | 206 | 26.70% | 0.3146 | -1,690,000đ | -7.13% | 0.182471 | 0.182471 |
| window_60 | per_occurrence | 206 | 26.31% | 0.3068 | 1,590,000đ | 6.71% | 0.182471 | 0.182471 |
| window_90 | once_per_day | 206 | 24.85% | 0.2951 | -3,210,000đ | -13.55% | 0.182170 | 0.182170 |
| window_90 | per_occurrence | 206 | 25.44% | 0.2981 | 870,000đ | 3.67% | 0.182170 | 0.182170 |
| window_365 | once_per_day | 206 | 26.60% | 0.2990 | -1,770,000đ | -7.47% | 0.181486 | 0.181486 |
| window_365 | per_occurrence | 206 | 25.92% | 0.2971 | 790,000đ | 3.33% | 0.181486 | 0.181486 |
| historical | once_per_day | 206 | 23.11% | 0.2680 | -4,650,000đ | -19.63% | 0.181278 | 0.181278 |
| historical | per_occurrence | 206 | 23.98% | 0.2796 | -650,000đ | -2.74% | 0.181278 | 0.181278 |
| ewma_30 | once_per_day | 206 | 22.52% | 0.2573 | -5,130,000đ | -21.65% | 0.187414 | 0.181184 |
| ewma_30 | per_occurrence | 206 | 23.11% | 0.2612 | -2,170,000đ | -9.16% | 0.187414 | 0.181184 |
| ewma_90 | once_per_day | 206 | 24.76% | 0.2922 | -3,290,000đ | -13.89% | 0.183188 | 0.183188 |
| ewma_90 | per_occurrence | 206 | 26.21% | 0.3068 | 1,590,000đ | 6.71% | 0.183188 | 0.183188 |
| markov_smoothed | once_per_day | 206 | 23.30% | 0.2660 | -4,490,000đ | -18.95% | 0.181301 | 0.181301 |
| markov_smoothed | per_occurrence | 206 | 23.88% | 0.2718 | -1,290,000đ | -5.45% | 0.181301 | 0.181301 |
| logistic_3y | once_per_day | 206 | 23.30% | 0.2583 | -4,490,000đ | -18.95% | 0.181563 | 0.181193 |
| logistic_3y | per_occurrence | 206 | 23.50% | 0.2612 | -2,170,000đ | -9.16% | 0.181563 | 0.181193 |
| lgbm_original | once_per_day | 206 | 26.02% | 0.2981 | -2,250,000đ | -9.50% | 0.182020 | 0.182020 |
| lgbm_original | per_occurrence | 206 | 26.31% | 0.3029 | 1,270,000đ | 5.36% | 0.182020 | 0.182020 |
| lgbm_regularized_3y | once_per_day | 206 | 24.08% | 0.2767 | -3,850,000đ | -16.25% | 0.181264 | 0.181264 |
| lgbm_regularized_3y | per_occurrence | 206 | 24.47% | 0.2796 | -650,000đ | -2.74% | 0.181264 | 0.181264 |
| poisson_3y | once_per_day | 206 | 22.82% | 0.2621 | -4,890,000đ | -20.64% | 0.181234 | 0.181234 |
| poisson_3y | per_occurrence | 206 | 22.82% | 0.2621 | -2,090,000đ | -8.82% | 0.181234 | 0.181234 |

Mốc xác suất đều nhau 23.7657%: Brier=0.181183, log loss=0.548379. Xác suất không phải điểm xếp hạng; sigmoid đơn điệu chủ yếu sửa độ tin cậy, không tạo thêm thông tin để chọn số.

## Kiểm tra hồi cứu giai đoạn gần đây: cấu hình đã chọn

| Trả thưởng | Mô hình | Số ngày | Lãi/lỗ | ROI |
|---|---|---:|---:|---:|
| once_per_day | window_14 | 90 | -1,390,000đ | -13.43% |
| per_occurrence | window_30 | 90 | -750,000đ | -7.25% |

## Tệp để kiểm tra và tái lập

- [Giao thức khóa trước khi chạy](./protocol.json), [cấu hình chọn](./selection.json), [quyết định](./decision.json).
- [Kết quả phát triển](./development.csv), [toàn bộ chỉ số](./metrics.csv), [từng ngày và dòng tiền](./daily_results.csv).
- [Bảng hiệu chỉnh xác suất](./calibration_bins.csv), [ngày cuối nhãn huấn luyện](./training_cutoffs.csv), [quy tắc sigmoid](./calibration.json).
- [Ảnh chụp dữ liệu nguồn](./source_snapshot.csv).

## Tài liệu phương pháp

- [scikit-learn: lựa chọn mô hình và sai lệch khi tái dùng dữ liệu kiểm tra](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- [scikit-learn: hiệu chỉnh xác suất và reliability diagrams](https://scikit-learn.org/stable/modules/calibration.html).
- [LightGBM: mục tiêu binary và Poisson](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective).

Không tự động thay mô hình sản xuất chỉ vì một cấu hình có ROI tốt nhất sau khi xem tập kiểm tra. Khi chưa có bằng chứng ổn định, kết luận đúng là chưa chứng minh được lợi thế, không phải tiếp tục tìm tham số cho đến khi báo cáo có lãi.
