# So sánh đã hiệu chỉnh: 90d daily và 1095d quarterly

> Bản này thay thế so sánh trước; nhánh 1095d đã sửa lỗi phụ thuộc ngày bắt đầu đánh giá.

| Mô hình | Giao thức | Ngày hit | Tỷ lệ hit | Lượt trúng | Lãi/lỗ | ROI |
|---|---|---:|---:|---:|---:|---:|
| Poisson | 90d_daily | 179/297 | 60,27% | 279 | +3.564.000đ | 14,81% |
| Binary | 90d_daily | 174/297 | 58,59% | 261 | +1.782.000đ | 7,41% |
| Poisson | 1095d_quarterly | 167/297 | 56,23% | 260 | +1.683.000đ | 7,00% |
| Binary | 1095d_quarterly | 164/297 | 55,22% | 264 | +2.079.000đ | 8,64% |

## Chênh lệch ghép cặp 90d − 1095d

- **Poisson**: hit-rate 4,04% (block 95% -5,05% đến 13,13%; McNemar p=0.379); lượt trúng +19; lãi/lỗ +1.881.000đ.
- **Binary**: hit-rate 3,37% (block 95% -4,38% đến 11,45%; McNemar p=0.453); lượt trúng -3; lãi/lỗ -297.000đ.

[Tổng hợp](./summary.csv) · [So sánh ghép cặp](./paired_comparison.csv) · [Theo quý](./quarterly.csv) · [Giao thức](./protocol.json)
