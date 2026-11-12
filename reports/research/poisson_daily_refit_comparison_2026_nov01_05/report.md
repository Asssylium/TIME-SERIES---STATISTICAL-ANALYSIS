# Poisson 90 ngày và mặc định: kết quả 01–05/11/2026

**Phạm vi:** 01–05/11/2026, 5 ngày liên tiếp. Mỗi dự báo được fit trước ngày dự báo và chỉ dùng kết quả đã có trước ngày đó. Top-3 mỗi model mỗi ngày.

**Giao thức:** Poisson 90 ngày dùng nhãn của 90 ngày lịch trước ngày dự báo; Poisson mặc định dùng toàn bộ nhãn có trước ngày dự báo. Cả hai refit hằng ngày với cùng `CountLGBMPredictor` và bộ feature như báo cáo trước.

**Quy tắc P&L:** 27.000đ/số/ngày, tổng cược 81.000đ/ngày/model; trả gộp 99.000đ mỗi lần xuất hiện, tính lặp. `✓` trúng, `✗` trượt; `×2`/`×3` là số lần số đó xuất hiện trong 27 giải.

## Tổng kết

| Model | Ngày có hit | Lượt trúng | Chi | Thu gộp | Lãi/lỗ | ROI |
|---|---:|---:|---:|---:|---:|---:|
| poisson_90d_daily | 4/5 | 7 | 405.000đ | 693.000đ | +288.000đ | 71.11% |
| poisson_all_history_daily | 3/5 | 4 | 405.000đ | 396.000đ | -9.000đ | -2.22% |

## Từng ngày: 3 số dự báo và kết quả

| Ngày | Poisson 90d (refit hằng ngày) | Poisson mặc định (refit hằng ngày) |
|---|---|---|
| 2026-11-01 | 18✗, 33✗, 38✗ | 44✗, 72✗, 59✗ |
| 2026-11-02 | 18✗, 22✓, 32✗ | 44✗, 59✗, 72✓ |
| 2026-11-03 | 50✓, 17✗, 38✗ | 59✗, 72✗, 44✗ |
| 2026-11-04 | 17✓×3, 50✗, 32✗ | 44✗, 59✓, 79✓ |
| 2026-11-05 | 33✓, 93✓, 21✗ | 59✗, 72✗, 44✓ |

## Nguồn và kiểm tra

- Feed kết quả [xsmb.csv](https://github.com/khiemdoan/vietnam-lottery-xsmb-analysis/blob/main/data/xsmb.csv) có dữ liệu đến 04/11 tại thời điểm tải. Kết quả 05/11 được lấy từ [trang XSMB ngày 05/11/2026](https://xskt.com.vn/xsmb/ngay-5-11-2026); đã kiểm tra đủ 27 giải trước khi thêm vào snapshot.
- Snapshot dùng cho phép chạy: [xsmb_prizes_2026-11-06.csv](../../../data/processed/xsmb_prizes_2026-11-06.csv); SHA-256 `24b847a8b4f3db5acecbc9f93c8b6478b617460401b4000cf842f5dc7e46f9cc`.
- Đã xác nhận cutoff nhãn không vượt ngày trước dự báo và tính lại hit/lượt trực tiếp từ kết quả 27 giải.
- Đây là kết quả hồi cứu của 5 ngày, không đảm bảo hiệu quả tương lai.

Chi tiết máy đọc: [daily.csv](./daily.csv) | [summary.csv](./summary.csv) | [protocol.json](./protocol.json).
