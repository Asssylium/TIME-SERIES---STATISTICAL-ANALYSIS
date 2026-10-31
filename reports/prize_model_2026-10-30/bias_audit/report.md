# Kiểm tra thiên kiến trong báo cáo mô hình — 30/10/2026

Kết luận: chưa phát hiện rò rỉ tương lai trong các kiểm tra đã thực hiện, đặc biệt với Binary quý III/2026. Tuy nhiên, kết quả là nghiên cứu hồi cứu trên dữ liệu đã được xem nhiều lần. Khoản lãi không phải bằng chứng độc lập về khả năng sinh lời tương lai.

## Kiểm tra thực thi

- Đối chiếu mã nguồn với 10 mã băm trong giao thức huấn luyện: khớp.
- Kiểm tra cả 24 mốc huấn luyện của 3 mô hình × 8 quý: ngày nhãn huấn luyện cuối luôn trước ngày dự báo đầu tiên.
- Đếm lại kết quả trực tiếp từ 27 giải gốc cho toàn bộ 5.913 dòng ngày/mô hình/Top-K: số trúng phân biệt, số lần xuất hiện và lợi nhuận đều khớp. Giá 23.000đ/số; nhận tổng 80.000đ cho mỗi lần xuất hiện, tính cả số lặp.
- Dựng lại đặc trưng bằng dữ liệu cắt tại 30/06 và 29/09/2026: mọi đặc trưng ở các ngày chung khớp chính xác với bản có dữ liệu đến 29/10. Nhãn được loại khỏi so sánh đặc trưng.
- Huấn luyện mới Binary bằng nguồn đã cắt tại 30/06/2026, cửa sổ nhãn 1.095 ngày. Toàn bộ 9.200 điểm dự báo của 92 ngày quý III khớp chính xác với bộ nhớ đệm của báo cáo. Mô hình không nhận nhãn quý III khi huấn luyện.
- Đặc trưng dự báo mỗi ngày sử dụng kết quả hoàn tất của ngày trước đó. Việc sử dụng kết quả đầu quý để dự báo ngày tiếp theo trong quý phù hợp dự báo cuốn chiếu hằng ngày, không phải dự báo trước toàn quý vào 30/06.

Đây là kiểm tra có phạm vi cụ thể, không chứng minh mọi nguồn sai lệch đều đã được loại bỏ. Chưa đối chiếu các phiên bản dữ liệu nguồn được lưu đúng vào từng ngày lịch sử.

## Lợi nhuận Binary Top 5 trong quý III/2026

| Tháng | Ngày | Số lần được trả thưởng | Lợi nhuận |
|---|---:|---:|---:|
| 07/2026 | 31 | 48 | +275.000đ |
| 08/2026 | 31 | 53 | +675.000đ |
| 09/2026 | 30 | 56 | +1.030.000đ |
| Tổng | 92 | 157 | +1.980.000đ |

Ba ngày lợi nhuận cao nhất đóng góp 935.000đ (47,2% tổng lãi). Nếu bỏ ba ngày này, 89 ngày còn lại vẫn lãi 1.045.000đ. Đây chỉ là kiểm tra độ nhạy sau khi quan sát, không phải quy tắc giao dịch hay phép kiểm định.

Bootstrap mô tả theo khối 7 ngày quan sát, 2.000 mẫu, cho khoảng 95% của lợi nhuận bình quân quý III từ +3.261 đến +40.652đ/ngày. Khoảng này dương, nhưng CHƯA điều chỉnh việc lựa chọn mô hình, Top-K và giai đoạn sau khi xem kết quả. Không được dùng nó như xác nhận độc lập về lợi thế.

## Những thiên kiến còn tồn tại

1. **Tái sử dụng tập kiểm thử:** 2025 và 2026 đã tham gia nhiều vòng phân tích, tạo ý tưởng và so sánh. Chia ngày đúng trong mã không xóa được ảnh hưởng của việc con người đã xem kết quả trước khi quyết định thiết kế tiếp theo.
2. **Chọn kết quả thuận lợi sau khi xem:** bảng theo quý có 72 tổ hợp (3 mô hình × 3 Top-K × 8 quý, bao gồm quý IV/2026 chưa đủ), trong đó 20 tổ hợp lãi. Các tổ hợp phụ thuộc nhau; đây không phải 72 phép thử độc lập, và cũng chưa bao gồm mọi thử nghiệm trước đó. Nhấn mạnh Binary Top 5 quý III có thể tạo ấn tượng quá tích cực.
3. **Khoảng tin cậy chưa điều chỉnh lựa chọn:** bootstrap từng kết quả không bao quát toàn bộ quá trình thử nhiều phương án. Không có căn cứ để gán một mức ý nghĩa thống kê đã điều chỉnh cho cả lịch sử nghiên cứu.
4. **Mục tiêu học khác mục tiêu tiền thưởng:** Binary học xác suất xuất hiện ít nhất một lần, trong khi lợi nhuận tính mỗi lần lặp. Poisson học số lần xuất hiện phù hợp hơn về mục tiêu, nhưng cũng chưa chứng minh sinh lời ổn định.
5. **Xác suất chưa cho thấy lợi thế đáng tin:** hiệu chỉnh Binary trên dự báo lịch sử 2025 cho hệ số co về mức nền bằng 0. Thứ hạng thô vẫn được xuất, nhưng không thể coi chênh lệch xác suất thô là đã được xác nhận.
6. **Vai trò thứ tự giải cần mô tả đúng:** đặc trưng hiện tại nhận diện từng vị trí giải; việc đặt tên vị trí theo thứ tự quay không tự chứng minh mô hình đã học quy luật chuyển tiếp giữa các lần quay. Thứ tự dựa trên xác nhận của người dùng.

Trên toàn bộ 657 ngày kiểm thử 2025–29/10/2026, Binary Top 5 vẫn lỗ **1.635.000đ**. Mốc này cũng là kết quả hồi cứu.

## Cách đánh giá tiếp theo

Khóa trước mô hình, Top-K, lịch tái huấn luyện, mục tiêu và thời hạn đánh giá. Lưu dự báo cùng dấu thời gian trước kết quả thực tế. Báo cáo toàn bộ ngày trong giai đoạn đã định, các đối chứng và ngày thiếu dữ liệu; không chọn lại quý bắt đầu hoặc mô hình dựa vào lợi nhuận đã thấy. Mọi thử nghiệm bổ sung phải được ghi nhận là nghiên cứu, tách khỏi phép thử tương lai đã khóa.

[Bằng chứng kiểm tra](./checks.json) · [Tất cả kết quả tháng](./monthly.csv) · [Tất cả kết quả quý](./quarterly.csv) · [Báo cáo gốc](../report.md)
