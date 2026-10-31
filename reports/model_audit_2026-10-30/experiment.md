# Kiểm tra mô hình và thí nghiệm cải tiến — 30/10/2026

Hiệu chỉnh trên dự báo ngoài mẫu theo thời gian của năm 2025; đánh giá hồi cứu trên 2026 đến 29/10. Tất cả ML được huấn luyện lại trước mỗi quý, cùng lịch. Không so trực tiếp tiền lời với báo cáo cũ huấn luyện lại hàng ngày.

Không tìm thêm tham số sau khi xem kết quả. Không tuyên bố năm 2026 là tập kiểm tra chưa từng được xem. Các so sánh Top 1–6 đều là chẩn đoán, không tự chọn cấu hình có lãi cao nhất để triển khai.

## Độ chính xác xác suất và số lần xuất hiện

| Mô hình | Brier gốc | Brier hiệu chỉnh | Trọng số giữ điểm gốc | MSE số lần | Poisson deviance |
|---|---:|---:|---:|---:|---:|
| original_binary | 0.182223 | 0.181152 | 0.000000 | — | — |
| regularized_binary | 0.181241 | 0.181152 | 0.000000 | — | — |
| count_poisson | — | — | — | 0.267088 | 0.800035 |
| rolling_30 | 0.182545 | 0.181152 | 0.000000 | 0.269048 | 0.807489 |
| uniform_reference | 0.181152 | — | — | 0.266932 | 0.799451 |

Các thước đo trên càng thấp càng tốt. Trọng số hiệu chỉnh bằng 0 nghĩa là trả về xác suất nền cho mọi số; không còn tín hiệu xếp hạng từ xác suất đã hiệu chỉnh. Bảng Top K dưới đây giữ thứ hạng gốc để không tạo thứ hạng giả từ các điểm bằng nhau.

## Lãi/lỗ theo từng lần xuất hiện (đồng)

| Mô hình | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 | Top 6 |
|---|---:|---:|---:|---:|---:|---:|
| original_binary | -911,000 | -1,422,000 | -1,613,000 | -1,884,000 | -2,555,000 | -2,666,000 |
| regularized_binary | +129,000 | -622,000 | +307,000 | +36,000 | +325,000 | -266,000 |
| count_poisson | -1,951,000 | -1,982,000 | -3,613,000 | -4,204,000 | -3,675,000 | -3,546,000 |
| rolling_30 | +209,000 | +178,000 | -573,000 | -1,404,000 | -2,715,000 | -2,426,000 |

## Giới hạn và diễn giải

- Brier tốt hơn không đồng nghĩa Top K trúng nhiều hơn. Hiệu chỉnh đơn điệu chỉ sửa độ tin cậy; trọng số 0 không hỗ trợ việc chọn số.
- Mô hình count_poisson dự báo kỳ vọng số lần, không phải phần trăm xác suất. Tổng kỳ vọng của 100 số được chuẩn hóa về 27; không giả định các số độc lập để ép xác suất cộng thành 1.
- regularized_binary và count_poisson dùng cùng đặc trưng và lịch sử 3 năm để so ảnh hưởng của mục tiêu học. So với mô hình gốc còn thay cả đặc trưng, lịch sử và mức điều chuẩn, nên không quy hết thay đổi cho một nguyên nhân.
- Khoảng 95% trong ranking.csv là bootstrap khối 7 ngày cho chênh lệch tiền/ngày so mô hình gốc, 2.000 mẫu. Khoảng xấp xỉ và không điều chỉnh cho việc so nhiều mô hình/K; không phải cam kết lợi nhuận tương lai.
- Ngày thiếu kết quả hoặc thiếu ngày liền trước bị loại. Kết quả được tính trên 297 ngày có thể đánh giá.

[Chỉ số](./metrics.csv) · [Top K và khoảng bất định](./ranking.csv) · [Từng ngày](./daily.csv) · [Giao thức](./protocol.json) · [Hiệu chỉnh](./calibration.json)

Phương pháp: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html), [LightGBM objectives](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective).
