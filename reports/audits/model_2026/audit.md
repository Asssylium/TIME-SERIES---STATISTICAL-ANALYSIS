# Kiểm tra mô hình và hướng cải thiện — 30/10/2026

Đã kiểm tra pipeline, bổ sung phương án cải tiến, chạy thí nghiệm cố định và kiểm thử.
Kết quả: mô hình nhỏ hơn với đặc trưng chuẩn hóa tăng tỷ lệ trúng quan sát được trong
thí nghiệm này, nhưng chưa chứng minh được lợi thế ổn định. Không tự thay mặc định.

## Những gì đúng và những gì cần sửa

| Hạng mục | Kết luận | Xử lý |
|---|---|---|
| Ngày, số 00, số lặp, 27 kết quả | Không tìm thấy lỗi đếm hoặc lệch nhãn ngày kế tiếp | Kiểm tra causal prefix và đối chiếu kết quả thực tế |
| Rò rỉ dữ liệu | Không tìm thấy lỗi dùng kết quả ngày dự báo trong nhãn/đặc trưng của chính dự báo đó | Giữ kiểm tra theo toàn bộ ngày và mốc nhãn |
| Cache backtest khi chọn ngày kết thúc sớm | Có lỗi truy cập các ngày ngoài phần lịch sử đã cắt | Đã sửa so sánh đúng phần lịch sử chung và thêm kiểm thử hồi quy |
| Mục tiêu học | Binary đúng cho xuất hiện ít nhất một lần, nhưng không trực tiếp tối ưu số lần trả thưởng | Thêm count target và Poisson riêng, không dùng binary target làm đặc trưng |
| Đặc trưng tích lũy | history_days/co_occurrence_total tăng theo thời gian; không phải tín hiệu cặp số riêng biệt | Phương án mới thay bằng mức trung bình theo số ngày quan sát |
| Nhịp gan khi thiếu ngày | Chỉ biết lần xuất hiện cuối được quan sát, không biết chắc số đã vắng suốt khoảng trống | Thêm gap_observation_complete |
| Xác suất | Điểm gốc chưa được hiệu chỉnh; không đáng tin chỉ vì lớn hơn 28,75% | Thêm hiệu chỉnh co về xác suất nền; đánh giá riêng Brier và thứ hạng |
| Chính sách đánh giá | CLI giữ mô hình cố định trong kỳ test, backtest trước huấn luyện lại hàng ngày | In rõ chính sách trong CLI; thí nghiệm này dùng cùng lịch trước mỗi quý cho các ML |
| Lãi/lỗ trong CLI | Trước chỉ báo chỉ số trúng khác nhau | Đã bổ sung tiền thưởng theo từng lần xuất hiện, đúng mức 23.000/80.000đ |

## Phương án đã kiểm tra

1. **Mô hình gốc:** binary LightGBM, 200 cây, 15 lá, learning rate 0,05, L2=1; toàn bộ lịch sử.
2. **Regularized:** binary LightGBM, 100 cây, 7 lá, learning rate 0,03, min_child_samples=200,
   L2=20; lịch sử 1.095 ngày; rolling và co-occurrence chuẩn hóa, cờ nhịp gan có đủ quan sát.
3. **Count Poisson:** cùng đặc trưng/lịch sử và mức điều chuẩn như regularized, nhưng nhãn là
   số lần xuất hiện. Tổng kỳ vọng mỗi ngày của 100 số bằng 27, không gọi điểm này là xác suất.
4. **Rolling 30 ngày:** baseline cập nhật theo kết quả đã biết; thêm mốc xác suất nền và kỳ vọng 0,27 lần/số.

Không tìm thêm tham số theo kết quả 2026. Hiệu chỉnh xác suất khớp trên dự báo ngoài mẫu
theo thời gian của 2025, sau đó giữ nguyên. Đánh giá 297 ngày năm 2026 đến 29/10,
huấn luyện ML trước mỗi quý và giữ cố định trong quý. Năm 2026 đã từng được phân tích,
nên đây là kiểm tra hồi cứu, không phải bằng chứng độc lập từ dữ liệu tương lai.

## Kết quả thực nghiệm

Precision là tỷ lệ số được chọn có xuất hiện, không tính lặp. Tất cả dòng dùng cùng lịch huấn luyện theo quý.

| Mô hình | Precision@1 | Precision@3 | Precision@6 | Lãi/lỗ Top 3 | Lãi/lỗ Top 6 |
|---|---:|---:|---:|---:|---:|
| Gốc | 22,56% | 23,23% | 23,46% | −1.613.000đ | −2.666.000đ |
| Regularized | 25,25% | 25,36% | 25,08% | +307.000đ | −266.000đ |
| Count Poisson | 18,18% | 20,76% | 22,90% | −3.613.000đ | −3.546.000đ |
| Rolling 30 ngày | 25,93% | 24,13% | 24,13% | −573.000đ | −2.426.000đ |

Regularized cải thiện quan sát ở mọi K=1..6 so với mô hình gốc trong phép thử này.
Tuy nhiên, khoảng bootstrap 95% cho chênh lệch tiền/ngày vẫn chứa 0 ở mọi K:
chưa có đủ bằng chứng về lợi thế ổn định. Top 3 regularized lỗ ở quý 1 và phần tháng 10;
Top 6 cũng lỗ ở hai khoảng đó. Không chọn K chỉ vì một dòng trong bảng có lãi.

MSE dự báo số lần của Poisson là 0,267088, còn mốc cố định 0,27 lần/số là 0,266932
(thấp hơn tốt hơn). Đổi đúng mục tiêu học không tự tạo ra tín hiệu dự báo.
Do đó Poisson được giữ như mô hình thử nghiệm, không đề xuất thay thế mô hình hiện tại.

Brier của mô hình gốc là 0,182223; regularized là 0,181241; xác suất nền là 0,181152.
Hiệu chỉnh của cả ba mô hình có đầu ra xác suất đều chọn trọng số 0 trên 2025:
trả về mức nền khoảng 23,84% cho mọi số. Điều này giảm sai số xác suất trong lần kiểm tra,
nhưng không tạo thứ hạng để chọn số. Các bảng Top K giữ thứ hạng gốc, không tạo thứ hạng
giả từ các xác suất bằng nhau. Xem [tài liệu hiệu chỉnh](https://scikit-learn.org/stable/modules/calibration.html).

## Cách dùng phương án regularized

Đây là lựa chọn thử nghiệm; `lightgbm` vẫn là mặc định. Mọi phần trăm được ghi rõ chưa hiệu chỉnh.
Mở PowerShell tại thư mục dự án và chạy bằng Python đang có trên máy:

```powershell
@'
import os, sys
sys.path.insert(0, 'D:/xsmb-model-check/deps')
sys.path.insert(0, os.getcwd())
from main import main
raise SystemExit(main(['--model', 'regularized', '--top-k', '6']))
'@ | & 'C:\Program Files\PostgreSQL\18\pgAdmin 4\python\python.exe' -
```

Ngày dự báo bằng ngày mới nhất trong nguồn cộng một ngày. `--top-k` chỉ đổi số lượng lựa chọn,
không phải cam kết số lượng đó có lãi. Kiểm thử 90 ngày in trong CLI là một lần fit cố định,
khác phép so sánh theo quý trong báo cáo và khác backtest huấn luyện lại mỗi ngày.

Đã chạy CLI thực tế bằng snapshot đến 29/10: phép kiểm tra cố định từ 01/08–29/10 cho
regularized Top 6 có Precision@6=22,78%, 139 lần xuất hiện, lỗ 1.300.000đ. Kết quả này
không phải phép so sánh mô hình cùng kỳ, nhưng cho thấy không thể lấy cải thiện trung bình
theo quý để khẳng định mọi lịch huấn luyện hoặc mọi giai đoạn đều tốt hơn.

Để kiểm chứng bước tiếp: chọn trước một cấu hình, K, lịch huấn luyện và số ngày đánh giá;
lưu dự báo trước khi có kết quả; không thay tham số theo các khoản lãi/lỗ tạm thời.
Đánh giá đồng thời số lần xuất hiện, Precision@K, sai số xác suất và dòng tiền. Chỉ có
kiểm chứng tiếp trên các ngày chưa xem mới bổ sung bằng chứng về hiệu quả tương lai.

## Bằng chứng và mã nguồn

- [Thí nghiệm đầy đủ](./experiment.md), [theo quý](./quarterly.csv), [tất cả Top K và khoảng bất định](./ranking.csv).
- [Giao thức cố định](./protocol.json), [hiệu chỉnh trên 2025](./calibration.json), [mốc nhãn huấn luyện](./training_cutoffs.csv).
- [Đặc trưng chuẩn hóa](../../../src/count_features.py), [mô hình regularized](../../../src/regularized_models.py),
  [Poisson](../../../src/count_models.py), [hiệu chỉnh](../../../src/calibration.py), [runner](../../../audit_experiment.py).
- Toàn bộ 84 kiểm thử đạt. Đã đối chiếu dự báo từ lớp regularized dùng trong CLI với cache thí nghiệm:
  khớp đến sai số số học; đã kiểm tra 24 bảng mô hình/K và toàn bộ mốc nhãn huấn luyện của thí nghiệm.
