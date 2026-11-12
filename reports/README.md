# Chỉ mục báo cáo

Thư mục này chỉ giữ các kết quả đã được chọn để tham khảo lâu dài. Các lần chạy thử và cache phải ghi vào `work/reports/`; chỉ chuyển kết quả vào đây sau khi đã kiểm tra.

## Báo cáo hiện hành

- [So sánh 90d và 1095d năm 2026](./current/comparison_90d_vs_1095d_2026/report.md)
- [Poisson và Binary 1095d, giai đoạn 2025–2026](./current/poisson_binary_1095d_2025_2026/report.md)
- [So sánh mô hình 90d mới nhất](./current/model_comparison_90d_2026/report.md)

## Nghiên cứu được giữ lại

- [Giới hạn Top-3 trong 000–099](./research/top3_000_099_2025_2026/report.md)
- [Top-3 ba chữ số 000–999](./research/top3_3digit_2025_2026/report.md)
- [Nghiên cứu mô hình nền](./research/baseline_model_2026/report.md)
- [Phụ thuộc cặp số](./research/pair_dependence_2026/report.md)
- [Chiến lược tránh số lặp](./research/repeat_strategy_90d/report.md)
- [Tín hiệu tần suất 90 ngày trên lịch sử dài](./research/short_horizon_persistence_2005_2026/report.md)
- [So sánh tần suất 90 ngày với Poisson chính](./research/short_horizon_persistence_2005_2026/poisson_comparison.md)
- [Báo cáo lãi/lỗ từng tháng năm 2026](./research/short_horizon_persistence_2005_2026/monthly_comparison_2026.md)
- [Poisson cửa sổ 90 ngày so với Poisson toàn lịch sử, 2024–2026](./research/poisson_window_90d_vs_expanding_2024_2026/report.md)
- [Poisson toàn lịch sử cố định, holdout năm 2026](./research/default_poisson_2026/report.md)
- [So sánh các biến thể Poisson năm 2026](./research/poisson_variants_2026_to_2026-11-05/report.md)
- [Báo cáo Poisson daily cũ (đã thay thế)](./research/poisson_daily_refit_comparison_2026/report.md)
- [Đối chiếu Poisson 90d và mặc định, 01–05/11/2026](./research/poisson_daily_refit_comparison_2026_nov01_05/report.md)
- [So sánh đủ 4 model với cửa sổ train 90 ngày, 2026 đến 05/11](./research/all_models_2026_90d_daily_refit_to_2026-11-05/report.md)

## Kiểm toán

- [Kiểm toán Binary 1095d Q3/2026](./audits/binary_1095d_q3_2026/report.md)
- [Kiểm toán mô hình 2026](./audits/model_2026/audit.md)
- [Chẩn đoán tín hiệu](./audits/signal_2026/report.md)
- Dữ liệu kiểm toán nguồn: [`raw_prize_2026/audit.json`](./audits/raw_prize_2026/audit.json)

## Dữ liệu vận hành

- [`paper_trial/`](./paper_trial/) là sổ paper-trading đang hoạt động, không phải báo cáo lịch sử để dọn tự động.
- [`prize_model_2026-10-30/`](./prize_model_2026-10-30/) chứa snapshot nguồn, mô hình và kết quả huấn luyện cần cho khả năng tái lập.

Các báo cáo cũ và cache đã được chuyển ra kho lưu trữ cạnh dự án, không bị xóa vĩnh viễn.
