"""Bounded model audit: 2025 chronological calibration, 2026 retrospective check.

Every model is refitted before each quarter; none is tuned on 2026 outcomes.
This is deliberately separate from the old daily-refit performance reports.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from src.calibration import ProbabilityShrinkageCalibrator
from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.count_models import CountLGBMPredictor
from src.data_loader import XSMBDataLoader
from src.diagnostics import block_bootstrap_mean_interval, evaluate_payouts, probability_diagnostics, rank_top_k
from src.features import build_feature_dataset
from src.models import LGBMPredictor, TARGET


MODELS = ('original_binary', 'regularized_binary', 'count_poisson', 'rolling_30')
CALIBRATION = [('2025-01-01','2025-03-31'),('2025-04-01','2025-06-30'),
               ('2025-07-01','2025-09-30'),('2025-10-01','2025-12-31')]
EVALUATION = [('2026-01-01','2026-03-31'),('2026-04-01','2026-06-30'),
              ('2026-07-01','2026-09-30'),('2026-10-01','2026-10-29')]


def run() -> None:
    """Persist protocol before fitting, calibration before evaluating, and all evidence."""
    root = Path(__file__).resolve().parents[2]
    out = root/'work/reports/model_audit_2026-10-30'
    out.mkdir(parents=True,exist_ok=True)
    cache=out/'cache'
    cache.mkdir(exist_ok=True)
    source=root/'data/processed/xsmb_prizes_2026-10-30.csv'
    loader=XSMBDataLoader(url=source.resolve().as_uri())
    raw=loader.load_data()
    frequency,binary=loader.preprocess_and_build_matrices(raw)
    frequency,binary=frequency.loc[:'2026-10-29'],binary.loc[:'2026-10-29']
    base=build_feature_dataset(frequency,binary)
    count_data=build_count_feature_dataset(frequency,binary)
    assert base.index.equals(count_data.index)
    origins=pd.DatetimeIndex(base.index.get_level_values(0).unique())
    targets=origins+pd.Timedelta(days=1)
    actual=frequency.loc[targets].to_numpy(dtype=int)
    np.testing.assert_array_equal(count_data[COUNT_TARGET].to_numpy(dtype=int).reshape(-1,100),actual)
    anchor=float(binary.loc[:'2024-12-31'].to_numpy().mean())
    regularized_params={**CountLGBMPredictor().params,'objective':'binary'}
    protocol={
        'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'calibration_blocks':CALIBRATION,'retrospective_blocks':EVALUATION,
        'models':MODELS,'original_parameters':LGBMPredictor().params,
        'regularized_parameters':regularized_params,'count_parameters':CountLGBMPredictor().params,
        'calibration_anchor_from_history_before_2025':anchor,
        'training':'Original: all history. Regularized binary and Poisson: preceding 1095 calendar days, identical normalized features. All ML refit at quarter start only; training target dates strictly before quarter start.',
        'selection':'No parameter/model/K search; K1..6 all reported. Calibration fitted solely on 2025 OOF predictions, frozen for 2026. No automatic replacement of production model.',
        'caution':'2026 has already been examined. This is retrospective method comparison, not untouched or prospective evidence. Calibration can improve Brier without improving ranking.',
        'code_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in [
            'scripts/operations/audit_experiment.py','src/models.py','src/features.py','src/count_models.py',
            'src/count_features.py','src/calibration.py','src/diagnostics.py']},
    }
    protocol=json.loads(json.dumps(protocol))
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Protocol/code changed; do not reuse incompatible experiment cache.')
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rolls=frequency.rolling('30D').sum()
    observations=pd.Series(1.,index=frequency.index).rolling('30D').sum()
    rolling_count=((rolls+30*0.27).div(observations+30,axis=0)).loc[origins].to_numpy()
    rolling_p=((binary.rolling('30D').sum()+30*anchor).div(observations+30,axis=0)).loc[origins].to_numpy()
    cutoffs=[]

    def block(first: str, last: str) -> dict:
        start,end=pd.Timestamp(first),pd.Timestamp(last)
        test=(targets>=start)&(targets<=end)
        train=targets<start
        recent=train&(targets>=start-pd.Timedelta(days=1095))
        assert targets[train].max()<targets[test].min()
        result={'dates':targets[test],'actual':actual[test],'scores':{},'probabilities':{},'counts':{}}
        for name in MODELS:
            if name=='rolling_30':
                result['scores'][name]=rolling_count[test]
                result['counts'][name]=rolling_count[test]
                result['probabilities'][name]=rolling_p[test]
                continue
            path=cache/f'{first}_{name}.npz'
            selected_train=train if name=='original_binary' else recent
            cutoffs.append({'model':name,'block_start':first,'block_end':last,
                            'training_label_end':str(targets[selected_train].max().date()),
                            'first_test_target':str(targets[test].min().date())})
            if path.exists():
                with np.load(path) as stored:
                    score=stored['scores']
            else:
                print(f'FIT {first}..{last} {name}',flush=True)
                if name=='original_binary':
                    model=LGBMPredictor().fit(base.loc[np.repeat(train,100)])
                    score=model.predict_proba(base.loc[np.repeat(test,100)]).to_numpy().reshape(-1,100)
                elif name=='regularized_binary':
                    features=count_data.drop(columns=COUNT_TARGET).astype(float)
                    model=LGBMClassifier(**regularized_params)
                    model.fit(features.loc[np.repeat(recent,100)],(actual[recent]>0).ravel().astype(int),categorical_feature=['number_id'])
                    score=model.predict_proba(features.loc[np.repeat(test,100)])[:,1].reshape(-1,100)
                else:
                    model=CountLGBMPredictor().fit(count_data.loc[np.repeat(recent,100)])
                    score=model.predict_expected_counts(count_data.loc[np.repeat(test,100)]).to_numpy().reshape(-1,100)
                np.savez_compressed(path,scores=score)
            assert score.shape==actual[test].shape and np.isfinite(score).all()
            result['scores'][name]=score
            result['counts' if name=='count_poisson' else 'probabilities'][name]=score
        return result

    calibration_blocks=[block(a,b) for a,b in CALIBRATION]
    cal_actual=np.concatenate([x['actual'] for x in calibration_blocks])
    calibrators={}
    config={}
    for name in ('original_binary','regularized_binary','rolling_30'):
        p=np.concatenate([x['probabilities'][name] for x in calibration_blocks])
        calibrator=ProbabilityShrinkageCalibrator(anchor).fit(p,cal_actual>0)
        calibrators[name]=calibrator
        config[name]={'anchor':anchor,'weight':calibrator.weight_}
    (out/'calibration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('FROZEN CALIBRATION',config,flush=True)
    evaluated=[block(a,b) for a,b in EVALUATION]
    dates=evaluated[0]['dates'].append([x['dates'] for x in evaluated[1:]])
    counts=np.concatenate([x['actual'] for x in evaluated])
    metrics,ranking,daily=[],[],[]
    daily_profits={}
    for name in MODELS:
        scores=np.concatenate([x['scores'][name] for x in evaluated])
        row={'model':name,'days':len(dates)}
        if name in calibrators:
            p=np.concatenate([x['probabilities'][name] for x in evaluated])
            raw_stats,_=probability_diagnostics(counts>0,p)
            adjusted_stats,_=probability_diagnostics(counts>0,calibrators[name].transform(p))
            row.update(raw_brier=raw_stats['brier'],calibrated_brier=adjusted_stats['brier'],
                       raw_log_loss=raw_stats['log_loss'],calibrated_log_loss=adjusted_stats['log_loss'],calibration_weight=calibrators[name].weight_)
        if name in ('count_poisson','rolling_30'):
            mu=np.concatenate([x['counts'][name] for x in evaluated])
            np.testing.assert_allclose(mu.sum(axis=1),27,rtol=1e-10)
            row.update(count_mse=float(np.square(mu-counts).mean()),
                       poisson_deviance=float(mean_poisson_deviance(counts.ravel(),mu.ravel())))
        metrics.append(row)
        for k in range(1,7):
            selection=rank_top_k(scores,k)
            payout=evaluate_payouts(counts,selection,payout_mode='per_occurrence')
            ranking.append({'model':name,'top_k':k,**payout.summary})
            frame=payout.per_day.copy()
            frame.insert(0,'date',dates.strftime('%Y-%m-%d'))
            frame['model'],frame['top_k']=name,k
            daily.append(frame)
            daily_profits[(name,k)]=frame.profit.to_numpy()
    uniform_p=np.full(counts.shape,anchor)
    reference,_=probability_diagnostics(counts>0,uniform_p)
    metrics.append({'model':'uniform_reference','days':len(dates),'raw_brier':reference['brier'],
                    'raw_log_loss':reference['log_loss'],'count_mse':float(np.square(counts-0.27).mean()),
                    'poisson_deviance':float(mean_poisson_deviance(counts.ravel(),np.full(counts.size,0.27)))})
    for row in ranking:
        delta=daily_profits[(row['model'],row['top_k'])]-daily_profits[('original_binary',row['top_k'])]
        lo,hi=block_bootstrap_mean_interval(delta)
        row.update(profit_delta_vs_original=float(delta.sum()),delta_mean_profit_95_low=lo,delta_mean_profit_95_high=hi)
    metrics_frame=pd.DataFrame(metrics)
    rankings=pd.DataFrame(ranking)
    metrics_frame.to_csv(out/'metrics.csv',index=False)
    rankings.to_csv(out/'ranking.csv',index=False)
    pd.concat(daily,ignore_index=True).to_csv(out/'daily.csv',index=False)
    pd.DataFrame(cutoffs).to_csv(out/'training_cutoffs.csv',index=False)
    lines=['# Kiểm tra mô hình và thí nghiệm cải tiến — 30/10/2026','',
           'Hiệu chỉnh trên dự báo ngoài mẫu theo thời gian của năm 2025; đánh giá hồi cứu trên 2026 đến 29/10. Tất cả ML được huấn luyện lại trước mỗi quý, cùng lịch. Không so trực tiếp tiền lời với báo cáo cũ huấn luyện lại hàng ngày.',
           '', 'Không tìm thêm tham số sau khi xem kết quả. Không tuyên bố năm 2026 là tập kiểm tra chưa từng được xem. Các so sánh Top 1–6 đều là chẩn đoán, không tự chọn cấu hình có lãi cao nhất để triển khai.',
           '', '## Độ chính xác xác suất và số lần xuất hiện','',
           '| Mô hình | Brier gốc | Brier hiệu chỉnh | Trọng số giữ điểm gốc | MSE số lần | Poisson deviance |',
           '|---|---:|---:|---:|---:|---:|']
    for row in metrics:
        def value(key):
            return f'{row[key]:.6f}' if key in row else '—'
        lines.append('| '+str(row['model'])+' | '+' | '.join(value(key) for key in ['raw_brier','calibrated_brier','calibration_weight','count_mse','poisson_deviance'])+' |')
    lines+=['','Các thước đo trên càng thấp càng tốt. Trọng số hiệu chỉnh bằng 0 nghĩa là trả về xác suất nền cho mọi số; không còn tín hiệu xếp hạng từ xác suất đã hiệu chỉnh. Bảng Top K dưới đây giữ thứ hạng gốc để không tạo thứ hạng giả từ các điểm bằng nhau.',
            '', '## Lãi/lỗ theo từng lần xuất hiện (đồng)','', '| Mô hình | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 | Top 6 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        lines.append('| '+name+' | '+' | '.join(f'{x:+,.0f}' for x in rankings.loc[rankings.model==name].sort_values('top_k').profit)+' |')
    lines+=['','## Giới hạn và diễn giải','',
            '- Brier tốt hơn không đồng nghĩa Top K trúng nhiều hơn. Hiệu chỉnh đơn điệu chỉ sửa độ tin cậy; trọng số 0 không hỗ trợ việc chọn số.',
            '- Mô hình count_poisson dự báo kỳ vọng số lần, không phải phần trăm xác suất. Tổng kỳ vọng của 100 số được chuẩn hóa về 27; không giả định các số độc lập để ép xác suất cộng thành 1.',
            '- regularized_binary và count_poisson dùng cùng đặc trưng và lịch sử 3 năm để so ảnh hưởng của mục tiêu học. So với mô hình gốc còn thay cả đặc trưng, lịch sử và mức điều chuẩn, nên không quy hết thay đổi cho một nguyên nhân.',
            '- Khoảng 95% trong ranking.csv là bootstrap khối 7 ngày cho chênh lệch tiền/ngày so mô hình gốc, 2.000 mẫu. Khoảng xấp xỉ và không điều chỉnh cho việc so nhiều mô hình/K; không phải cam kết lợi nhuận tương lai.',
            '- Ngày thiếu kết quả hoặc thiếu ngày liền trước bị loại. Kết quả được tính trên '+str(len(dates))+' ngày có thể đánh giá.',
            '', '[Chỉ số](./metrics.csv) · [Top K và khoảng bất định](./ranking.csv) · [Từng ngày](./daily.csv) · [Giao thức](./protocol.json) · [Hiệu chỉnh](./calibration.json)',
            '', 'Phương pháp: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html), [LightGBM objectives](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective).', '']
    (out/'experiment.md').write_text('\n'.join(lines),encoding='utf-8')
    print('METRICS',metrics_frame.to_json(orient='records'),flush=True)
    print('RANKING',rankings[['model','top_k','profit','precision_at_k','occurrences_per_selection','delta_mean_profit_95_low','delta_mean_profit_95_high']].to_json(orient='records'),flush=True)
    print('REPORT',out/'experiment.md',flush=True)


if __name__=='__main__':
    run()
