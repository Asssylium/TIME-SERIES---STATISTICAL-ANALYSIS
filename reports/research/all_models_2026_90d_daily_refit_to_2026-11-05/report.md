# So sánh 4 model: cửa sổ huấn luyện 90 ngày, 2026

**Phạm vi:** 01/01–05/11/2026. Cả bốn model chọn Top-3, dùng nhãn của tối đa 90 ngày lịch ngay trước từng ngày dự báo và refit hằng ngày. Có **304 ngày dự báo ghép cặp trên 309 ngày lịch**; loại 5 ngày 16–20/02 do thiếu kết quả/ngày liền trước.

**Cách tính:** 27.000đ cho mỗi số mỗi ngày (81.000đ/ngày/model); thu gộp 99.000đ cho mỗi lần xuất hiện trúng, tính cả số lặp; lãi/lỗ = thu − chi, ROI = lãi/lỗ ÷ chi.

## Các model được so sánh

| Tên báo cáo | Ý nghĩa model |
|---|---|
| `regularized` | LightGBM nhị phân, feature count chuẩn hóa và regularization; dự đoán số có xuất hiện hay không. |
| `poisson` | **Poisson Prize 90d**: hồi quy Poisson trên feature theo vị trí 27 giải; xếp hạng theo số lần xuất hiện kỳ vọng. Đây là biến thể Poisson cửa sổ 90 ngày của session cũ, **không phải** `main.py --model poisson`. |
| `binary` | Binary Prize 90d trên feature theo vị trí 27 giải; dự đoán hiện diện/không hiện diện. |
| `lightgbm` | LightGBM nhị phân trên bộ feature tần suất/gap thông thường. |

**Lưu ý:** cả bốn được cấp cùng cửa sổ nhãn 90 ngày và cùng ngày dự báo; feature/model family khác nhau theo định nghĩa ở trên. `main.py --model poisson` là Poisson Count khác, không nằm trong phép so sánh này.

## Tổng kết cả kỳ

| Model | Ngày có hit | Lượt trúng (kể cả lặp) | Ngày có lãi | Tổng chi | Tổng thu | Lãi/lỗ ròng | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regularized | 174/304 (57,24%) | 259 | 174 | 24.624.000đ | 25.641.000đ | +1.017.000đ | 4,13% |
| Poisson (Prize) | 183/304 (60,20%) | 285 | 183 | 24.624.000đ | 28.215.000đ | +3.591.000đ | 14,58% |
| Binary (Prize) | 179/304 (58,88%) | 270 | 179 | 24.624.000đ | 26.730.000đ | +2.106.000đ | 8,55% |
| LightGBM | 173/304 (56,91%) | 270 | 173 | 24.624.000đ | 26.730.000đ | +2.106.000đ | 8,55% |

## Lãi/lỗ theo từng tháng

Mỗi ô ghi **lãi/lỗ (ROI)**. Dấu `*` đánh dấu tháng chưa đủ ngày: tháng 2 thiếu 5 ngày; tháng 11 mới có kết quả đến ngày 05.

| Tháng | Ngày chạy | Regularized | Poisson (Prize) | Binary (Prize) | LightGBM |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 31/31 | +261.000đ (10,39%) | +558.000đ (22,22%) | +162.000đ (6,45%) | +162.000đ (6,45%) |
| 2026-02* | 23/28 | −378.000đ (−20,29%) | −675.000đ (−36,23%) | −279.000đ (−14,98%) | +117.000đ (6,28%) |
| 2026-03 | 31/31 | +162.000đ (6,45%) | +558.000đ (22,22%) | −36.000đ (−1,43%) | −1.026.000đ (−40,86%) |
| 2026-04 | 30/30 | +441.000đ (18,15%) | +1.431.000đ (58,89%) | +936.000đ (38,52%) | −153.000đ (−6,30%) |
| 2026-05 | 31/31 | −234.000đ (−9,32%) | +261.000đ (10,39%) | +63.000đ (2,51%) | +459.000đ (18,28%) |
| 2026-06 | 30/30 | +45.000đ (1,85%) | −153.000đ (−6,30%) | +342.000đ (14,07%) | −54.000đ (−2,22%) |
| 2026-07 | 31/31 | −135.000đ (−5,38%) | +558.000đ (22,22%) | +360.000đ (14,34%) | +261.000đ (10,39%) |
| 2026-08 | 31/31 | +162.000đ (6,45%) | −135.000đ (−5,38%) | +63.000đ (2,51%) | +855.000đ (34,05%) |
| 2026-09 | 30/30 | +243.000đ (10,00%) | +441.000đ (18,15%) | −153.000đ (−6,30%) | +441.000đ (18,15%) |
| 2026-10 | 31/31 | +558.000đ (22,22%) | +657.000đ (26,16%) | +459.000đ (18,28%) | +855.000đ (34,05%) |
| 2026-11* | 5/30 | −108.000đ (−26,67%) | +90.000đ (22,22%) | +189.000đ (46,67%) | +189.000đ (46,67%) |

## Hit, lượt trúng và dòng tiền từng tháng

Trong mỗi ô: **ngày có hit/ngày chạy; số lượt trúng; chi → thu**. Ngày có hit nghĩa là ít nhất một trong ba số dự báo xuất hiện; lượt trúng tính lặp nếu một số xuất hiện nhiều lần.

| Tháng | Regularized | Poisson (Prize) | Binary (Prize) | LightGBM |
|---|---|---|---|---|
| 2026-01 | 19/31 ngày; 28 lượt; 2.511.000đ → 2.772.000đ | 15/31; 31 lượt; 2.511.000đ → 3.069.000đ | 17/31; 27 lượt; 2.511.000đ → 2.673.000đ | 17/31; 27 lượt; 2.511.000đ → 2.673.000đ |
| 2026-02 | 11/23 ngày; 15 lượt; 1.863.000đ → 1.485.000đ | 9/23; 12 lượt; 1.863.000đ → 1.188.000đ | 10/23; 16 lượt; 1.863.000đ → 1.584.000đ | 12/23; 20 lượt; 1.863.000đ → 1.980.000đ |
| 2026-03 | 19/31 ngày; 27 lượt; 2.511.000đ → 2.673.000đ | 22/31; 31 lượt; 2.511.000đ → 3.069.000đ | 19/31; 25 lượt; 2.511.000đ → 2.475.000đ | 12/31; 15 lượt; 2.511.000đ → 1.485.000đ |
| 2026-04 | 22/30 ngày; 29 lượt; 2.430.000đ → 2.871.000đ | 22/30; 39 lượt; 2.430.000đ → 3.861.000đ | 21/30; 34 lượt; 2.430.000đ → 3.366.000đ | 17/30; 23 lượt; 2.430.000đ → 2.277.000đ |
| 2026-05 | 17/31 ngày; 23 lượt; 2.511.000đ → 2.277.000đ | 19/31; 28 lượt; 2.511.000đ → 2.772.000đ | 19/31; 26 lượt; 2.511.000đ → 2.574.000đ | 19/31; 30 lượt; 2.511.000đ → 2.970.000đ |
| 2026-06 | 17/30 ngày; 25 lượt; 2.430.000đ → 2.475.000đ | 16/30; 23 lượt; 2.430.000đ → 2.277.000đ | 17/30; 28 lượt; 2.430.000đ → 2.772.000đ | 17/30; 24 lượt; 2.430.000đ → 2.376.000đ |
| 2026-07 | 14/31 ngày; 24 lượt; 2.511.000đ → 2.376.000đ | 19/31; 31 lượt; 2.511.000đ → 3.069.000đ | 18/31; 29 lượt; 2.511.000đ → 2.871.000đ | 17/31; 28 lượt; 2.511.000đ → 2.772.000đ |
| 2026-08 | 16/31 ngày; 27 lượt; 2.511.000đ → 2.673.000đ | 18/31; 24 lượt; 2.511.000đ → 2.376.000đ | 18/31; 26 lượt; 2.511.000đ → 2.574.000đ | 20/31; 34 lượt; 2.511.000đ → 3.366.000đ |
| 2026-09 | 17/30 ngày; 27 lượt; 2.430.000đ → 2.673.000đ | 16/30; 29 lượt; 2.430.000đ → 2.871.000đ | 16/30; 23 lượt; 2.430.000đ → 2.277.000đ | 16/30; 29 lượt; 2.430.000đ → 2.871.000đ |
| 2026-10 | 20/31 ngày; 31 lượt; 2.511.000đ → 3.069.000đ | 24/31; 32 lượt; 2.511.000đ → 3.168.000đ | 21/31; 30 lượt; 2.511.000đ → 2.970.000đ | 22/31; 34 lượt; 2.511.000đ → 3.366.000đ |
| 2026-11 | 2/5 ngày; 3 lượt; 405.000đ → 297.000đ | 3/5; 5 lượt; 405.000đ → 495.000đ | 3/5; 6 lượt; 405.000đ → 594.000đ | 4/5; 6 lượt; 405.000đ → 594.000đ |

## Nhận xét

- Poisson Prize đạt lãi cộng dồn cao nhất trong kỳ: **+3.591.000đ**, với 183/304 ngày có hit.
- LightGBM và Binary Prize cùng lãi **+2.106.000đ**, nhưng số ngày có hit lần lượt là 173 và 179.
- Regularized lãi **+1.017.000đ**.
- Đây là kết quả hồi cứu, không chứng minh lợi thế dự đoán hay đảm bảo lợi nhuận trong tương lai.

## Nguồn và kiểm chứng

- Snapshot dữ liệu [xsmb_prizes_2026-11-06.csv](../../../data/processed/xsmb_prizes_2026-11-06.csv) kết thúc ngày 05/11/2026. Feed thô có đến 04/11 khi tải; kết quả ngày 05/11 lấy từ [trang XSMB 05/11](https://xskt.com.vn/xsmb/ngay-5-11-2026), ghép vào snapshot sau khi xác nhận đủ 27 giải.
- 297 ngày đến 29/10 dùng lại score cache của báo cáo 90d cũ. Xếp hạng Top-3 khớp CSV đã lưu cho cả bốn model trên toàn bộ 297 ngày; hit và lượt trúng được tính lại khớp với snapshot hiện tại. Bảy ngày 30/10–05/11 được fit lại hằng ngày cùng giao thức.
- Đã đối chiếu tổng ngày/tháng và chi, thu, lãi/lỗ; tổng các tháng khớp với tổng kỳ.

Chi tiết máy đọc: [daily.csv](./daily.csv) · [monthly.csv](./monthly.csv) · [monthly_pivot.csv](./monthly_pivot.csv) · [summary.csv](./summary.csv) · [protocol.json](./protocol.json).
