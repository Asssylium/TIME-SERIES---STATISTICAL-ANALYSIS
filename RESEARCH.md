# Chạy nghiên cứu và đọc kết quả

Kết quả đã chọn nằm trong [báo cáo](reports/research/baseline_model_2026/report.md).
Xem [từng ngày](reports/research/baseline_model_2026/daily_results.csv) và
[chỉ số tổng hợp](reports/research/baseline_model_2026/metrics.csv) để kiểm tra lại.

## Chạy bằng môi trường Python hiện có trên máy này

Mở PowerShell tại thư mục dự án rồi chạy:

```powershell
@'
import os, sys, runpy
sys.dont_write_bytecode = True
sys.path.insert(0, 'D:/xsmb-model-check/deps')
sys.path.insert(0, os.getcwd())
runpy.run_path('research.py', run_name='__main__')
'@ | & 'C:\Program Files\PostgreSQL\18\pgAdmin 4\python\python.exe' -
```

Nếu Python chuẩn đã được cài cùng các thư viện cần thiết:

```powershell
python research.py
```

Chương trình dùng lại cache khi mã nguồn, dữ liệu và giao thức không thay đổi.
Nếu nguồn CSV đã cập nhật, chương trình từ chối trộn dữ liệu mới với kết quả cũ.
Dùng `--output work/reports/research_new` với Python chuẩn để ghi lần chạy mới.
Với lệnh PowerShell ở trên, thêm dòng
`sys.argv = ['research.py', '--output', 'work/reports/research_new']`
trước dòng `runpy.run_path(...)`.

Đây là thí nghiệm có các mốc thời gian cố định, không phải hệ thống tự động
tối ưu lại sau mỗi lần thấy kết quả. Mỗi lần nghiên cứu mới cần xác định trước
tập phát triển, hiệu chỉnh và kiểm tra riêng; chạy lại trên cùng dữ liệu không
tạo thêm bằng chứng độc lập.

## Diễn giải đúng

- Dữ liệu có 100 số từ 00 đến 99; mỗi ngày có 27 kết quả, có thể trùng.
- `Precision@5` đếm số trúng khác nhau trong 5 số đã chọn.
- `occurrences_per_selection` tính cả số lần lặp, phù hợp khi trả 99.000đ
  cho mỗi lần xuất hiện. Hai cách trả thưởng có thể dẫn tới Top 5 khác nhau.
- 99.000đ được hiểu là tổng tiền nhận; giá mua 27.000đ mỗi số.
- Tỷ lệ ngày trúng ít nhất một số không phải tỷ lệ ngày có lãi:
  mua 5 số tốn 135.000đ, nhận một khoản thưởng 99.000đ vẫn lỗ 36.000đ.
- Các báo cáo/giao thức đã tạo trước khi sửa quy tắc giữ nguyên số liệu lịch sử.
- Phần trăm của LightGBM trong lệnh dự báo thông thường là ước lượng chưa
  hiệu chỉnh. Bản nghiên cứu kiểm tra hiệu chỉnh riêng nhưng không tự động
  đưa mô hình hoặc quy tắc hiệu chỉnh sang lệnh dự báo.
- Lượt nghiên cứu huấn luyện ML trước từng khối thời gian rồi giữ cố định
  trong khối. Do đó kết quả không đồng nhất với backtest huấn luyện lại mỗi ngày.

## Đối chiếu sau khi chạy

Đã kiểm tra 7.597 ngày nguồn, 23.324 dòng kết quả theo ngày, 140 dòng tổng hợp
và 24 mốc huấn luyện. Số trúng, số lần lặp, tiền chi, tiền nhận và lãi/lỗ đều
khớp dữ liệu nguồn; ngày cuối nhãn huấn luyện luôn trước ngày dự báo đầu khối.

Với LightGBM cũ, Top 5 theo xác suất xuất hiện có kết quả:

| Giai đoạn | Xác suất dự báo trung bình | Tỷ lệ xuất hiện thực tế |
|---|---:|---:|
| 206 ngày kiểm tra, 01/01–30/07/2026 | 30,91% | 26,02% |
| 90 ngày hồi cứu, 31/07–28/10/2026 | 31,12% | 23,78% |

Đây là đối chiếu trung bình nhóm Top 5, không phải kết luận rằng mọi số được
dự báo 30% đều có cùng xác suất thực tế. Hiệu chỉnh sigmoid của mô hình này
không vượt qua kiểm tra trên giai đoạn hiệu chỉnh riêng nên không được áp dụng.
