# Kiểm toán cú nhảy Binary 1095d trong Q3/2026

## Kết luận

Cú nhảy Q3 trong báo cáo cũ không phải do cộng sai tiền thưởng và cũng không phải do rò rỉ nhãn tương lai. Nguyên nhân là quy trình cũ **cắt dữ liệu gốc theo ngày bắt đầu đánh giá trước khi tạo các đặc trưng tích lũy**. Vì vậy, cùng Q3, cùng cửa sổ huấn luyện 1095 ngày và cùng tham số mô hình nhưng chạy toàn kỳ từ 01/01/2025 hoặc chạy riêng từ 01/07/2026 cho feature và dự đoán khác nhau.

Runner đã được sửa để luôn dựng feature nhân quả từ cùng mốc đầu của nguồn dữ liệu, rồi mới cắt các dòng cần huấn luyện/đánh giá. Sau sửa, replay toàn kỳ và replay riêng Q3 khớp **92/92 dự đoán Binary**.

## Kết quả Q3/2026

| Phiên bản | Ngày hit | Tỷ lệ hit | Lượt trúng | Lãi/lỗ | ROI |
|---|---:|---:|---:|---:|---:|
| Báo cáo cũ, chạy từ 01/01/2025 | 65/92 | 70,65% | 115 | +3.933.000đ | 52,78% |
| Runner cũ, chạy riêng từ 01/07/2026 | 62/92 | 67,39% | 90 | +1.458.000đ | 19,57% |
| **Runner đã sửa, lịch sử feature cố định** | **59/92** | **64,13%** | **102** | **+2.646.000đ** | **35,51%** |

Q3 vẫn là quý mạnh của Binary 1095d sau hiệu chỉnh, nhưng không còn mức 65 ngày hit và 115 lượt trúng của báo cáo cũ.

## Các kiểm tra đã thực hiện

- Nhãn huấn luyện Q3 kết thúc ngày 30/06/2026; không chồng lấn các ngày đánh giá 01/07–30/09/2026.
- Các feature tại ngày `t` chỉ dùng kết quả hoàn tất đến ngày `t`; không phát hiện future leakage.
- Công thức tiền: 81.000đ tiền cược/ngày, nhận gộp 99.000đ cho mỗi lượt xuất hiện. Đối chiếu độc lập từ 27 giải gốc khớp **2.628/2.628 dòng** của replay đã sửa.
- Báo cáo cũ tái lập đúng **2.628/2.628 dòng** khi giữ ngày bắt đầu 01/01/2025. Điều này xác nhận lỗi là độ nhạy ẩn theo mốc tạo feature, không phải file kết quả bị hỏng.
- Sau sửa, replay toàn kỳ và replay riêng Q3 khớp **92/92 dòng** Binary trong Q3.
- Bộ kiểm thử liên quan sau sửa: **5/5 đạt**; có thêm test buộc kết quả của quý chồng lấn không phụ thuộc ngày bắt đầu đánh giá.

## Ảnh hưởng đến so sánh 90d và 1095d

Trên 297 ngày năm 2026:

| Mô hình | 90d daily | 1095d quarterly đã sửa |
|---|---:|---:|
| Poisson | 179 ngày hit, 279 lượt, +3.564.000đ | 167 ngày hit, 260 lượt, +1.683.000đ |
| Binary | 174 ngày hit, 261 lượt, +1.782.000đ | 164 ngày hit, 264 lượt, +2.079.000đ |

Vì vậy, kết luận cũ rằng Binary 1095d nhảy vọt rõ rệt so với 90d về ngày hit không còn đứng vững. 90d có nhiều ngày hit hơn; Binary 1095d chỉ nhỉnh hơn 3 lượt xuất hiện và 297.000đ lãi trong mẫu này.

## Tệp kiểm chứng

- [Báo cáo Poisson/Binary 1095d đã hiệu chỉnh](../../current/poisson_binary_1095d_2025_2026/report.md)
- [So sánh 90d–1095d đã hiệu chỉnh](../../current/comparison_90d_vs_1095d_2026/report.md)
