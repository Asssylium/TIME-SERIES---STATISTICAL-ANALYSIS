# Nghiên cứu đồng xuất hiện theo cặp

## Giao thức

- Giai đoạn khám phá: 2005-10-01 đến 2022-12-31.
- Giai đoạn đối chiếu: 2023-01-01 đến 2026-10-29.
- Kiểm tra toàn bộ 4,950 cặp không thứ tự của các đuôi số khác nhau trong mỗi giai đoạn.
- Giả thuyết không: 27 kết quả mỗi ngày là các lượt rút độc lập, đồng xác suất từ 00–99. Theo giả thuyết này, xác suất một cặp cố định cùng xuất hiện ít nhất một lần là
  `1 - 2(0.99)^27 + (0.98)^27 = 5.488210%` mỗi ngày có kết quả.
- Mỗi cặp dùng kiểm định nhị thức chính xác hai phía trên các ngày có kết quả. Hiệu chỉnh Holm kiểm soát sai số toàn cục trên cả 4.950 cặp trong từng giai đoạn, kể cả khi các phép thử phụ thuộc nhau.
- Hiệu ứng là tỷ lệ ngày cặp số cùng xuất hiện trừ tỷ lệ theo giả thuyết không. Khoảng Wilson 95% được tính cho tỷ lệ quan sát; trừ tỷ lệ giả thuyết không để ra khoảng hiệu ứng hiển thị.
- Dữ liệu lịch sử đã được xem trong nghiên cứu trước. Giai đoạn sau là đối chiếu hồi cứu, không phải holdout chưa xem hoặc xác nhận tiền cứu.

## Kết quả

- Giai đoạn khám phá: **0 / 4,950** cặp có p đã hiệu chỉnh Holm ≤ 0,05.
- Giai đoạn đối chiếu: **0 / 4,950** cặp có p đã hiệu chỉnh Holm ≤ 0,05.
- Có ý nghĩa ở cả hai giai đoạn và cùng chiều hiệu ứng: **0** cặp.
- Ngày lịch thiếu kết quả được loại ra, không nội suy. Mẫu số mỗi giai đoạn là số ngày có kết quả đã được xác thực.

| Giai đoạn | Cặp số | Số ngày cùng xuất hiện | Hiệu ứng so với giả thuyết không (điểm %) | p hiệu chỉnh Holm |
|---|---|---:|---:|---:|
| Khám phá | 31–38 | 420/6216 | +1.269 | 0.09948 |
| Khám phá | 50–51 | 273/6216 | -1.096 | 0.5298 |
| Khám phá | 01–38 | 409/6216 | +1.092 | 1 |
| Khám phá | 38–88 | 408/6216 | +1.075 | 1 |
| Khám phá | 51–80 | 278/6216 | -1.016 | 1 |
| Đối chiếu | 00–23 | 45/1382 | -2.232 | 0.584 |
| Đối chiếu | 09–43 | 109/1382 | +2.399 | 0.9513 |
| Đối chiếu | 01–71 | 48/1382 | -2.015 | 1 |
| Đối chiếu | 11–98 | 48/1382 | -2.015 | 1 |
| Đối chiếu | 32–61 | 106/1382 | +2.182 | 1 |

Bảng liệt kê năm p hiệu chỉnh nhỏ nhất mỗi giai đoạn; đây không phải phát hiện có ý nghĩa nếu p đã hiệu chỉnh lớn hơn 0,05. Kết quả đầy đủ của mọi cặp và giai đoạn nằm trong `discovery.csv`, `replication.csv` và `stability.csv`.

## Diễn giải và giới hạn

Phép sàng lọc này kiểm tra liệu tỷ lệ đồng xuất hiện của một cặp có khác mô hình rút công bằng, độc lập và đồng đều hay không. Phép thử **không** tách riêng tương tác cặp khỏi sai lệch nhỏ trong tần suất biên của từng số: một tỷ lệ cặp bất thường có thể do tần suất của một hoặc cả hai số. Audit tần suất biên trước đó không tìm thấy tín hiệu sau hiệu chỉnh, nhưng điều đó không chứng minh phân phối hoàn toàn đồng đều.

Các p-value nhị thức chính xác giả định các ngày quay độc lập và 27 kết quả trong mỗi ngày độc lập, đồng đều. Hiệu chỉnh Holm xử lý sự phụ thuộc giữa 4.950 phép thử cặp số, không xử lý tự tương quan theo thời gian giữa các ngày. Ngày thiếu kết quả được bỏ qua. Ngay cả khi thấy bất thường thống kê, vẫn cần tái lập tiền cứu bằng dữ liệu mới và paper-trial với quy tắc đóng băng trước khi có thể dùng để dự đoán hoặc tuyên bố có lợi nhuận. Kết quả không có ý nghĩa thống kê không chứng minh tính ngẫu nhiên.

**Khuyến nghị:** Không có cặp nào vượt hiệu chỉnh nhiều phép thử ở cả hai giai đoạn và cùng chiều hiệu ứng. Không thêm luật dự đoán theo cặp; giữ nguyên mặc định production hiện tại.

Không thay đổi hành vi dự đoán production hoặc mặc định Poisson.
