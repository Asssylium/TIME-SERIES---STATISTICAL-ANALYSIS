# Kiểm tra dấu hiệu thống kê trong dữ liệu XSMB

## Kết quả

Không có giả thuyết nào trong 600 giả thuyết mỗi giai đoạn đạt mức p đã hiệu chỉnh ≤ 0,05. Đây là kết quả của các phép thử được liệt kê dưới đây, không phải chứng minh rằng mọi quan hệ có thể có đều vắng mặt.

| Nhóm kiểm tra | Số giả thuyết mỗi giai đoạn | Đạt ngưỡng ở giai đoạn tìm kiếm | Đạt ngưỡng ở giai đoạn đối chiếu |
|---|---:|---:|---:|
| Tần suất riêng của từng số so với kỳ vọng 0,27 lần/ngày | 100 | 0 | 0 |
| Khác biệt theo thứ trong tuần cho từng số | 100 | 0 | 0 |
| Tự tương quan số lần xuất hiện ở độ trễ 1, 2, 3, 7 ngày | 400 | 0 | 0 |
| **Tổng** | **600** | **0** | **0** |

- Tìm kiếm: 01/10/2005–31/12/2022, **6.216 ngày** có dữ liệu.
- Đối chiếu: 01/01/2023–29/10/2026, **1.382 ngày** có dữ liệu.
- Không sử dụng dữ liệu sau 2022 để chọn danh sách số trong phép đối chiếu tần suất cố định.
- Dữ liệu đều là lịch sử đã có thể được xem trong nghiên cứu trước, không phải kiểm tra triển vọng hoàn toàn mới.

## Phương pháp và giới hạn

Mỗi giai đoạn có 499 lần hoán vị. Kiểm tra tần suất hoán vị độc lập nhãn 100 số trong từng ngày, giữ nguyên 27 kết quả và cấu trúc số lặp. Kiểm tra thời gian hoán vị toàn bộ dòng ngày trong từng quý, giữ nguyên phân bố số của quý và sự đồng xuất hiện trong ngày. Dữ liệu được trừ trung bình từng quý trước khi đo tác động thứ trong tuần và tự tương quan. Độ trễ khớp theo ngày lịch chính xác, không nhảy qua ngày thiếu.

P-value được hiệu chỉnh bằng thống kê lớn nhất trong từng nhóm và Bonferroni cho ba nhóm. Các phép hoán vị dựa trên giả định có thể hoán đổi nhãn số hoặc ngày trong quý dưới giả thuyết không có hiệu ứng tương ứng. Sự thay đổi chế độ mạnh bên trong quý, quan hệ chéo giữa các số, cấu trúc giải thưởng và quan hệ phi tuyến khác chưa được bao quát đầy đủ.

Không đạt ngưỡng thống kê không có nghĩa xác suất tồn tại tín hiệu bằng 0. Có thể còn hiệu ứng nhỏ hoặc quan hệ ngoài phạm vi kiểm tra. Ngược lại, một bất thường thống kê nếu tìm thấy cũng chưa đủ chứng minh lợi nhuận sau chi phí.

Kiểm tra đối chứng trên dữ liệu nhân tạo đã phát hiện được thiên lệch số, mẫu theo thứ và tự tương quan mạnh được cố ý tạo; kiểm tra cũng từ chối suy luận theo thứ khi dữ liệu thiếu hoàn toàn một thứ.

## Đối chiếu phụ: danh sách tần suất cố định

Chọn đúng 10 số xuất hiện nhiều nhất trong giai đoạn tìm kiếm và giữ nguyên danh sách trong toàn bộ giai đoạn đối chiếu:

**38, 62, 59, 54, 64, 76, 24, 29, 93, 36.**

Đây là kiểm tra thiên lệch tần suất bền vững, không phải dự báo của mô hình regularized và không phải khuyến nghị mua.

| Chỉ tiêu trên 1.382 ngày có kết quả | Giá trị |
|---|---:|
| Lần xuất hiện được trả thưởng, tính cả lặp | 3.679 |
| Tổng chi, 23.000đ mỗi số/ngày | 317.860.000đ |
| Tổng nhận, 80.000đ mỗi lần xuất hiện | 294.320.000đ |
| Lãi/lỗ | **−23.540.000đ** |
| ROI | **−7,41%** |

Phép kiểm tra danh sách cố định không cần kết quả ngày liền trước. Chỉ 16 ngày thiếu kết quả thực tế bị loại; danh sách và ngày thiếu nằm trong dữ liệu đính kèm. Không suy ra dòng tiền của các ngày thiếu.

## Dữ liệu kiểm tra

- [So sánh từng giả thuyết ở hai giai đoạn](./comparison.csv)
- [Kết quả giai đoạn tìm kiếm](./discovery.csv)
- [Kết quả giai đoạn đối chiếu](./replication.csv)
- [Giao thức, giả định và dấu kiểm tra nguồn dữ liệu](./protocol.json)
- [Tổng hợp](./summary.json)
- [Dòng tiền danh sách cố định](./fixed_top10_replication.csv)
