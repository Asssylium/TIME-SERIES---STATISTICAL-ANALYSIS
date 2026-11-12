# Sổ dự báo và chấm kết quả giả lập

Chỉ mô phỏng, không có chức năng mua vé, chuyển tiền hay gửi dự báo ra ngoài.

## Chạy trên máy hiện tại

```powershell
Set-Location 'D:\TIME-SERIES & STATISTICAL ANALYSIS'
.\run_paper.ps1 init
.\run_paper.ps1 run
.\run_paper.ps1 status
.\run_paper.ps1 verify
```

`init` tạo giao thức đúng một lần. Nếu tạo trước 18:00 UTC+7, ngày bắt đầu là hôm nay; nếu muộn hơn thì là ngày mai. Thời hạn là 90 ngày lịch. Chạy lại `init` giữ nguyên giao thức và ngày bắt đầu.

`run` tải dữ liệu giải gốc, chấm các dự báo cũ đã có đủ kết quả, rồi ghi dự báo hôm nay nếu đủ điều kiện. Chạy nhiều lần không thay dự báo hoặc nhân đôi tiền thưởng. Không có tùy chọn ghi lùi ngày. `status` tái tạo báo cáo, `verify` kiểm tra chuỗi sự kiện.

[Lệnh chạy](./run_paper.ps1) sử dụng Python của pgAdmin và thư viện tại `D:/xsmb-model-check/deps` đang có trên máy. Không phụ thuộc lệnh `python` trong PATH. Máy khác cần cấu hình lại hai đường dẫn này trước khi bắt đầu thử nghiệm.

## Quy tắc đã khóa

- Phương án chính: **Poisson Top 5**; Binary Top 5 và Regularized Top 5 là hai đối chứng. Ba tài khoản giả lập riêng, không cộng chúng thành một chiến lược.
- 27.000đ/số/ngày; nhận tổng 99.000đ **mỗi lần xuất hiện**. Không cộng thêm hoàn vốn. Một số về hai lần nhận 198.000đ.
- Mỗi ngày phải ghi hoàn tất trước **18:00 UTC+7**. Mô hình chỉ nhận kết quả đến hôm trước. Nếu nguồn đã có kết quả hôm nay, từ chối ghi dự báo mang tính tương lai.
- Chấm kết quả từ **19:00** trở đi và chỉ khi nguồn có đủ 27 giải hợp lệ. Kết quả ngày cũ có thể được chấm vào lần chạy sau.
- Huấn luyện ban đầu với nhãn có trước ngày bắt đầu, sau đó huấn luyện lại vào quý lịch mới, dùng nhãn trước quý mới. Cửa sổ nhãn 1.095 ngày; tham số, phiên bản thư viện và mã nguồn được ghi trong giao thức. Không tìm tham số mới trên kết quả thử nghiệm.
- Đặc trưng được cập nhật bằng kết quả hoàn tất của hôm trước, dù mô hình giữ nguyên trong quý.
- Ngày bỏ lỡ dự báo, ngày chờ kết quả và ngày đã chấm là các trạng thái khác nhau. Không tính ngày thiếu thành thua hoặc ngầm coi là quyết định không mua.
- Điểm Poisson là số lần kỳ vọng; điểm Binary/Regularized là xác suất thô chưa hiệu chỉnh. Không đổi phương án chính sau khi thấy đối chứng có lãi hơn.

Sổ đã khởi tạo giữ nguyên giao thức và số tiền đã khóa; không sửa các kết quả lịch sử. Sau khi mã nguồn đổi, mã băm khóa sổ cũ sẽ từ chối ghi hoặc chấm tiếp bằng mã mới. Muốn chạy theo quy tắc mới, hãy khởi tạo sổ riêng.

## Chạy tự động

```powershell
.\install_paper_schedule.ps1
Get-ScheduledTask -TaskName 'XSMB-Paper-90Day-Trial'
```

[Trình cài lịch](./install_paper_schedule.ps1) tạo một tác vụ Windows chạy hằng ngày lúc **17:00** và **19:30**, kết thúc sau ngày thứ 90. Không ghi đè tác vụ đã tồn tại. Cửa sổ chạy được ẩn, quyền người dùng thường, không lưu mật khẩu. Nhật ký mỗi lần chạy nằm trong `reports/paper_trial/logs/`.

**Máy phải đang bật và tài khoản Windows này đang đăng nhập.** Không bảo đảm chạy khi máy ngủ hoặc tắt. Tác vụ có thể chạy bù khi khả dụng, nhưng chương trình vẫn từ chối ghi dự báo quá 18:00. Nguồn cập nhật chậm sẽ được chấm trong lần chạy sau. Trong Task Scheduler, mã thoát khác 0 là lỗi; đọc log để xử lý, không sửa sổ để che ngày bỏ lỡ.

Tạm dừng lịch khi cần:

```powershell
Disable-ScheduledTask -TaskName 'XSMB-Paper-90Day-Trial'
```

## Dữ liệu và tính toàn vẹn

Sổ gốc là `reports/paper_trial/journal.sqlite3`. Mỗi dự báo lưu đủ 100 điểm của từng mô hình, Top 5, nguồn đầu vào nén, mã băm nguồn, phiên bản mô hình và thời điểm ghi UTC. Kết quả được thêm bằng sự kiện riêng, không sửa dự báo. Mô hình đã đóng băng cũng được lưu trong sổ dưới dạng văn bản LightGBM.

SQLite chặn UPDATE/DELETE và khóa duy nhất mỗi ngày. Chuỗi SHA-256 giúp phát hiện sửa đổi nội dung; giao dịch nguyên tử chống ghi dở hoặc ghi trùng khi chạy đồng thời. Việc huấn luyện vượt giờ chốt sẽ hủy cả lần ghi. Nếu nguồn sửa lịch sử/kết quả đã chấm, chương trình báo lỗi và giữ nguyên dữ liệu đã ghi.

**Giới hạn:** dấu thời gian lấy từ đồng hồ máy. Người có toàn quyền tệp vẫn có thể xóa cơ sở dữ liệu hoặc dựng lại chuỗi băm; đây không phải chứng nhận thời gian độc lập. Không xóa/tạo lại sổ để chọn một ngày bắt đầu đẹp hơn. Nên sao lưu sổ khi không có tiến trình đang ghi.

Các báo cáo `report.md`, `summary.json`, `daily.csv`, `coverage.csv` được tái tạo từ sổ và có thể ghi đè. Chúng không phải bản ghi gốc bất biến. Báo cáo hiện tại không coi lãi ngắn hạn là bằng chứng thống kê; hết 90 ngày cần đánh giá toàn kỳ, độ bao phủ, rủi ro và chênh lệch với đối chứng.

Khi mã nguồn hoặc thư viện đã khóa thay đổi, chương trình từ chối thêm dự báo/chấm kết quả. Khôi phục đúng phiên bản để tiếp tục; không sửa giao thức cũ. Nếu cần thay phương pháp, ghi rõ thay đổi và quản lý như một thử nghiệm khác, không trộn kết quả.
