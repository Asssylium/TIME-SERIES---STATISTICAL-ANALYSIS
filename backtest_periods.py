"""Daily-refit LightGBM backtest grouped into fixed calendar-day periods.

Example: python backtest_periods.py --start 2026-01-01 --end 2026-10-29
All Top 1..6 use the same pre-outcome ranking. Payouts count every occurrence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data_loader import DEFAULT_XSMB_URL, XSMBDataLoader
from src.diagnostics import evaluate_payouts, rank_top_k
from src.features import build_feature_dataset
from src.models import LGBMPredictor, TARGET


def calendar_periods(start: str, end: str, period_days: int = 90) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return inclusive, nonoverlapping calendar periods, retaining a partial tail."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first != first.normalize() or last != last.normalize() or first > last:
        raise ValueError('Expected midnight dates with start <= end.')
    if isinstance(period_days, bool) or not isinstance(period_days, int) or period_days < 1:
        raise ValueError('period_days must be a positive integer.')
    return [(date, min(date + pd.Timedelta(days=period_days - 1), last))
            for date in pd.date_range(first, last, freq=f'{period_days}D')]


def _verify_prior_history(frequency: pd.DataFrame, prior_frequency: pd.DataFrame) -> None:
    """Verify a cached snapshot's relevant prefix before reusing its forecasts.

    The current matrix is already truncated to the requested end date. A prior
    snapshot can extend beyond it; those later dates are not part of this run.
    Within the shared prefix, require identical dates and counts, including no
    newly filled historical gaps that would change the causal features.
    """
    if frequency.empty or prior_frequency.empty:
        raise ValueError('Current and prior history must both contain observations.')
    prior_prefix = prior_frequency.loc[:frequency.index[-1]]
    if prior_prefix.empty:
        raise ValueError('Current and prior history have no shared prefix.')
    missing = prior_prefix.index.difference(frequency.index)
    if len(missing):
        raise ValueError('Current history is missing dates from the prior snapshot: '
                         + ', '.join(str(date.date()) for date in missing))
    pd.testing.assert_frame_equal(frequency.loc[:prior_prefix.index[-1]], prior_prefix)


def run(start: str, end: str, output: Path, workers: int = 2, period_days: int = 90) -> None:
    """Freeze input, fit separately before each target, and save auditable cash flows."""
    if workers not in (1, 2, 3, 4):
        raise ValueError('Use one to four workers to limit memory usage.')
    periods = calendar_periods(start, end, period_days)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'cache'
    cache.mkdir(exist_ok=True)
    snapshot = output / 'source_snapshot.csv'
    loader = XSMBDataLoader(url=snapshot.resolve().as_uri() if snapshot.exists() else DEFAULT_XSMB_URL)
    raw = loader.load_data()
    frequency, binary = loader.preprocess_and_build_matrices(raw)
    if frequency.index.max() < pd.Timestamp(end):
        raise ValueError('Source is not yet updated through requested end date.')
    frequency, binary = frequency.loc[:end], binary.loc[:end]
    digest = hashlib.sha256(frequency.to_csv().encode()).hexdigest()
    params = LGBMPredictor().params
    root = Path(__file__).parent
    protocol = {
        'start':start, 'end':end, 'period_days':period_days, 'top_k':[1,2,3,4,5,6],
        'source_url':DEFAULT_XSMB_URL, 'frequency_sha256':digest,
        'model_parameters':params, 'stake':23000, 'gross_payout':80000,
        'payout_mode':'per_occurrence',
        'training':'Fresh all-history fit before every target date; training labels end before target; same ranking for all K; no tuning.',
        'missing_dates':'Require actual target and preceding calendar day. Exclude unavailable evaluations, itemize them; do not impute zero outcomes or assume abstention.',
        'code_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in ['backtest_periods.py','src/data_loader.py','src/features.py','src/models.py','src/diagnostics.py']},
    }
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Protocol changed. Use a separate output directory.')
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if not snapshot.exists():
        raw.to_csv(snapshot,index=False)
    dates = pd.date_range(start,end)
    valid = dates.isin(frequency.index) & (dates-pd.Timedelta(days=1)).isin(frequency.index)
    targets = dates[valid]
    skipped = [{'date':str(t.date()), 'reason':'actual outcome missing' if t not in frequency.index else 'preceding calendar day missing'} for t in dates[~valid]]
    (output/'excluded_dates.json').write_text(json.dumps(skipped,indent=2),encoding='utf-8')
    print(f'PROTOCOL {len(targets)} evaluable days / {len(dates)} calendar days; exclusions={skipped}; workers={workers}',flush=True)

    # Reuse only the exact daily-refit forecasts, never the fixed-block research fits.
    old_path = root/'reports/backtest_top6_2026-09-29_2026-10-28.json'
    old_snapshot = root/'reports/research_2026-10-29/source_snapshot.csv'
    if old_path.exists() and old_snapshot.exists():
        old = json.loads(old_path.read_text(encoding='utf-8'))
        old_loader = XSMBDataLoader(url=old_snapshot.resolve().as_uri())
        old_raw = old_loader.load_data()
        old_freq, _ = old_loader.preprocess_and_build_matrices(old_raw)
        assert hashlib.sha256(old_raw.to_csv(index=False).encode()).hexdigest() == old['source_sha256']
        _verify_prior_history(frequency, old_freq)
        if old['model_parameters'] != params or not old['complete']:
            raise ValueError('Prior daily-fit model parameters differ.')
        for record in old['daily_results']:
            target = pd.Timestamp(record['target_date'])
            destination = cache/f'{target:%Y-%m-%d}.json'
            if target in targets and not destination.exists():
                saved = {key:record[key] for key in ['target_date','history_end','training_label_end','all_scores']}
                saved['provenance'] = str(old_path.relative_to(root))
                destination.write_text(json.dumps(saved),encoding='utf-8')

    data = build_feature_dataset(frequency,binary,include_unlabeled=True)
    origins = data.index.get_level_values(0)
    labeled = data.loc[data[TARGET].notna()]
    labeled_origins = labeled.index.get_level_values(0)
    model_columns = [name for name in data.columns if name != TARGET]

    def fit_day(target: pd.Timestamp) -> dict:
        cutoff = target-pd.Timedelta(days=1)
        train = labeled.iloc[:labeled_origins.searchsorted(cutoff)]
        model = LGBMPredictor().fit(train)
        latest = data.loc[origins==cutoff,model_columns]
        assert len(latest)==100 and model.trained_through_<target
        scores = model.predict_proba(latest).droplevel(0).reindex(frequency.columns).to_numpy()
        assert scores.shape==(100,) and np.isfinite(scores).all()
        return {'target_date':str(target.date()),'history_end':str(cutoff.date()),
                'training_label_end':str(model.trained_through_.date()),
                'all_scores':scores.tolist(),'provenance':'fresh_daily_refit'}

    records = {}
    for target in targets:
        destination = cache/f'{target:%Y-%m-%d}.json'
        if destination.exists():
            record = json.loads(destination.read_text(encoding='utf-8'))
            assert record['target_date']==str(target.date())
            assert pd.Timestamp(record['history_end'])==target-pd.Timedelta(days=1)
            assert pd.Timestamp(record['training_label_end'])<target
            assert len(record['all_scores'])==100
            records[target]=record
    remaining = [target for target in targets if target not in records]
    print(f'CACHE {len(records)} reusable; {len(remaining)} daily fits required',flush=True)
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fit_day,target):target for target in remaining}
        for done,future in enumerate(as_completed(futures),1):
            target=futures[future]
            record=future.result()
            destination=cache/f'{target:%Y-%m-%d}.json'
            temporary=destination.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record),encoding='utf-8')
            temporary.replace(destination)
            records[target]=record
            print(f'FIT {done}/{len(remaining)} target={target:%Y-%m-%d} elapsed={time.monotonic()-began:.0f}s',flush=True)

    scores=np.array([records[t]['all_scores'] for t in targets])
    counts=frequency.loc[targets].to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(counts.sum(axis=1),27)
    report_rows,daily_rows=[],[]
    groups=[(f'period_{i}',a,b) for i,(a,b) in enumerate(periods,1)] + [('total',pd.Timestamp(start),pd.Timestamp(end))]
    for k in range(1,7):
        selected=rank_top_k(scores,k)
        daily=evaluate_payouts(counts,selected,payout_mode='per_occurrence').per_day
        daily.insert(0,'target_date',targets.strftime('%Y-%m-%d'))
        daily['top_k']=k
        daily['prediction']=[' '.join(f'{n:02d}' for n in np.argsort(-row,kind='stable')[:k]) for row in scores]
        daily['hits_with_counts']=[' '.join(f'{n:02d}x{actual[n]}' for n in np.flatnonzero(mask & (actual>0))) for mask,actual in zip(selected,counts)]
        daily_rows.append(daily)
        for label,a,b in groups:
            mask=(targets>=a)&(targets<=b)
            if not mask.any():
                raise ValueError(f'No evaluable dates in {label}.')
            summary=evaluate_payouts(counts[mask],selected[mask],payout_mode='per_occurrence').summary
            report_rows.append({'period':label,'start':str(a.date()),'end':str(b.date()),
                                'calendar_days':(b-a).days+1,'excluded_days':int(((dates>=a)&(dates<=b)&~valid).sum()),
                                'top_k':k,**summary})
    summary=pd.DataFrame(report_rows)
    daily=pd.concat(daily_rows,ignore_index=True)
    for row in summary.itertuples():
        subset=daily.loc[(daily.top_k==row.top_k)&daily.target_date.between(row.start,row.end)]
        assert subset.profit.sum()==row.profit and subset.paid_occurrences.sum()==row.occurrences
    summary.to_csv(output/'summary.csv',index=False,encoding='utf-8-sig')
    daily.to_csv(output/'daily.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{key:value for key,value in records[t].items() if key!='all_scores'} for t in targets]).to_csv(output/'training_cutoffs.csv',index=False)
    lines=['# Lãi/lỗ Top 1–6 theo mốc 90 ngày từ đầu năm 2026','',
           f'Giai đoạn {start} đến {end}. Giá mua 23.000đ/số/ngày; tổng nhận 80.000đ mỗi lần xuất hiện, tính cả lặp. Cùng thứ hạng LightGBM, huấn luyện lại trước từng ngày bằng toàn bộ lịch sử đã có. Không thay tham số hay chọn K theo kết quả.',
           '', '90 ngày là độ dài khoảng báo cáo, không phải giới hạn lịch sử huấn luyện. Khoảng cuối có thể chưa đủ 90 ngày. Các ngày thiếu kết quả hoặc thiếu ngày liền trước bị loại khỏi phép đánh giá, không được coi là ngày trượt hay tự động bỏ mua.',
           '',f'{len(targets)} ngày đủ dữ liệu / {len(dates)} ngày lịch. Ngày bị loại: '+('; '.join(x['date']+' ('+x['reason']+')' for x in skipped) or 'không có')+'.',
           '', '## Lãi/lỗ (đồng)', '', '| Khoảng | Ngày đánh giá/ngày lịch | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 | Top 6 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,a,b in groups:
        block=summary.loc[summary.period==label].sort_values('top_k')
        name='Tổng' if label=='total' else f'{a:%d/%m}–{b:%d/%m}'
        lines.append(f'| {name} | {int(block.iloc[0].days)}/{(b-a).days+1} | '+' | '.join(f'{v:+,.0f}' for v in block.profit)+' |')
    lines+=['','## Tổng tiền và tỷ lệ','', '| Top K | Chi | Nhận | Lãi/lỗ | ROI | Lần xuất hiện | Ngày có lãi | Sụt giảm vốn lớn nhất |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary.loc[summary.period=='total'].itertuples():
        lines.append(f'| {row.top_k} | {row.cost:,.0f} | {row.receipts:,.0f} | {row.profit:+,.0f} | {row.roi:.2%} | {row.occurrences} | {row.profitable_days}/{row.days} | {row.max_drawdown:,.0f} |')
    lines+=['','## Diễn giải và kiểm tra','',
            '- Đây là backtest hồi cứu. Các khoảng đã được dùng trong nghiên cứu trước; không phải tập kiểm tra hoàn toàn mới. Không suy ra K tối ưu cho tương lai từ K có lãi cao nhất trong bảng.',
            '- Không có ngày cần dự báo trong nhãn huấn luyện của chính ngày đó. Đặc trưng tại t chỉ dùng kết quả đến t, dự báo t+1.',
            '- Khi chọn K số đều ngẫu nhiên, kỳ vọng số lần xuất hiện là 0,27 mỗi lượt mua; kỳ vọng lỗ 1.400đ/lượt với mức trả trên. Đây là kỳ vọng, không phải kết quả của một danh sách ngẫu nhiên cụ thể.',
            '- Tổng tiền trong bảng chỉ cho các ngày có thể đánh giá. Không thể suy ra dòng tiền thực tế của những ngày bị thiếu kết quả.',
            '', '[Tổng hợp đầy đủ](./summary.csv) · [Từng ngày](./daily.csv) · [Giao thức](./protocol.json) · [Ngày bị loại](./excluded_dates.json) · [Mốc nhãn huấn luyện](./training_cutoffs.csv)', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('SUMMARY',summary[['period','start','end','days','top_k','occurrences','profit','roi']].to_json(orient='records'),flush=True)
    print('REPORT',output/'report.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',default='2026-01-01')
    parser.add_argument('--end',default='2026-10-29')
    parser.add_argument('--period-days',type=int,default=90)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path('reports/backtest_90d_2026_to_2026-10-29'))
    args=parser.parse_args()
    run(args.start,args.end,args.output,args.workers,args.period_days)
