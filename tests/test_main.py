"""End-to-end pipeline tests with an offline CSV and real model fitting."""

from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from main import main, run_pipeline
from src.data_loader import NUMBER_COLUMNS
from src.count_features import build_count_feature_dataset
from src.features import build_feature_dataset
from src.models import BaselinePredictor, LGBMPredictor, TARGET
from src.regularized_models import RegularizedLGBMPredictor


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(91)
        self.raw = pd.DataFrame(
            rng.multinomial(27, np.full(100, 0.01), size=65), columns=NUMBER_COLUMNS
        )
        self.raw.insert(0, "date", pd.date_range("2026-01-01", periods=len(self.raw)))

    def test_both_models_and_separate_final_refit(self) -> None:
        training_frames = []
        baseline_histories = []
        original_fit = LGBMPredictor.fit
        original_baseline_fit = BaselinePredictor.fit

        def record_fit(model, dataset):
            training_frames.append(dataset.copy())
            return original_fit(model, dataset)

        def record_baseline_fit(model, frequency):
            baseline_histories.append(frequency.copy())
            return original_baseline_fit(model, frequency)

        output = io.StringIO()
        payload = self.raw.to_csv(index=False).encode()
        with patch("src.data_loader.urlopen", return_value=io.BytesIO(payload)) as download, \
                patch.object(LGBMPredictor, "fit", new=record_fit), \
                patch.object(BaselinePredictor, "fit", new=record_baseline_fit), \
                redirect_stdout(output):
            result = run_pipeline(model="both", test_days=5, n_splits=3, n_estimators=8)
        download.assert_called_once()
        self.assertEqual(set(result.forecasts), {"baseline", "lightgbm"})
        self.assertEqual(result.prediction_date, self.raw.date.iloc[-1] + pd.Timedelta(days=1))
        self.assertEqual(len(result.test_feature_dates), 5)
        self.assertLess(result.train_feature_dates[-1] + pd.Timedelta(days=1), result.test_feature_dates[0])
        # The evaluation fit contains only historical labeled data. A separate
        # final fit may use test outcomes only AFTER evaluation has completed.
        self.assertEqual(len(training_frames), 2)
        self.assertTrue(training_frames[0][TARGET].notna().all())
        self.assertEqual(training_frames[0].index.get_level_values(0).max(), result.train_feature_dates[-1])
        self.assertEqual(training_frames[1].index.get_level_values(0).max(), self.raw.date.iloc[-1])
        self.assertEqual(len(baseline_histories), 6)
        for date, history in zip(result.test_feature_dates, baseline_histories[:-1]):
            self.assertEqual(history.index.max(), date)
            self.assertGreater(history.index.min(), date - pd.Timedelta(days=30))
        for name, evaluation in result.evaluations.items():
            self.assertEqual(evaluation.summary["n_days"], 5)
            self.assertEqual(evaluation.summary["k"], 5)
            self.assertTrue(evaluation.per_day.n_actual.eq(27).all())
            self.assertTrue(evaluation.per_day.index.equals(result.test_feature_dates + pd.Timedelta(days=1)))
            self.assertEqual(len(result.forecasts[name]), 5)
            self.assertEqual(len({number for number, _ in result.forecasts[name]}), 5)
        self.assertTrue(all(0 <= score <= 1 for _, score in result.forecasts["lightgbm"]))
        self.assertIn("30-day count", output.getvalue())
        self.assertIn("P(at least one appearance)", output.getvalue())
        self.assertIn("uncalibrated", output.getvalue())
        self.assertIn("fitted once before the holdout", output.getvalue())
        self.assertEqual(set(result.payouts), {"baseline", "lightgbm"})
        self.assertEqual(output.getvalue().count("TOP-5 EVALUATION REPORT"), 2)

    def test_pipeline_payout_counts_repeated_occurrences(self) -> None:
        raw = self.raw.copy()
        raw.loc[:, NUMBER_COLUMNS] = 0
        raw["00"] = 27
        with patch("src.data_loader.urlopen", return_value=io.BytesIO(raw.to_csv(index=False).encode())), \
                redirect_stdout(io.StringIO()):
            result = run_pipeline(model="baseline", top_k=1, test_days=2, n_splits=2)
        money = result.payouts["baseline"].summary
        self.assertEqual(money["distinct_hits"], 2)
        self.assertEqual(money["occurrences"], 54)
        self.assertEqual(money["cost"], 46_000)
        self.assertEqual(money["receipts"], 4_320_000)
        self.assertEqual(money["profit"], 4_274_000)

    def test_regularized_pipeline_uses_count_features_without_target_leakage(self) -> None:
        payload = self.raw.to_csv(index=False).encode()
        output = io.StringIO()
        with patch("src.data_loader.urlopen", return_value=io.BytesIO(payload)), redirect_stdout(output):
            result = run_pipeline(model="regularized", top_k=3, test_days=5, n_splits=3, n_estimators=8)
        self.assertEqual(set(result.forecasts), {"regularized"})
        self.assertEqual(len(result.forecasts["regularized"]), 3)
        self.assertTrue(all(0 <= score <= 1 for _, score in result.forecasts["regularized"]))
        self.assertEqual(result.payouts["regularized"].summary["cost"], 5 * 3 * 23_000)
        self.assertIn("Experimental regularized classifier", output.getvalue())

    def test_missing_calendar_dates_are_not_mislabeled(self) -> None:
        raw = self.raw.drop(index=[10, 20, 60])
        with patch("src.data_loader.urlopen", return_value=io.BytesIO(raw.to_csv(index=False).encode())), \
                redirect_stdout(io.StringIO()):
            result = run_pipeline(model="baseline", test_days=5, n_splits=3)
        for origin in result.test_feature_dates:
            self.assertIn(origin + pd.Timedelta(days=1), set(raw.date))
        self.assertEqual(result.prediction_date, raw.date.iloc[-1] + pd.Timedelta(days=1))

    def test_requested_forecast_uses_only_the_historical_prefix_for_every_model(self) -> None:
        requested = pd.Timestamp("2026-02-20")
        truncated = self.raw.loc[self.raw.date < requested].copy()
        changed_future = self.raw.copy()
        future = changed_future.date >= requested
        changed_future.loc[future, NUMBER_COLUMNS] = 0
        changed_future.loc[future, "99"] = 27

        def offline_run(raw, model, **options):
            with patch("src.data_loader.urlopen", return_value=io.BytesIO(raw.to_csv(index=False).encode())), \
                    redirect_stdout(io.StringIO()):
                return run_pipeline(model=model, top_k=3, test_days=3, n_splits=2, n_estimators=5, **options)

        for model in ["both", "regularized"]:
            with self.subTest(model=model):
                builder = build_count_feature_dataset if model == "regularized" else build_feature_dataset
                builder_name = "main.build_count_feature_dataset" if model == "regularized" else "main.build_feature_dataset"
                with patch(builder_name, wraps=builder) as build:
                    reference = offline_run(truncated, model)
                    requested_result = offline_run(self.raw, model, forecast_date="2026-02-20")
                    altered_result = offline_run(changed_future, model, forecast_date="2026-02-20")
                self.assertEqual(build.call_count, 3)
                for call in build.call_args_list:
                    frequency, binary = call.args[:2]
                    self.assertEqual(frequency.index[-1], requested - pd.Timedelta(days=1))
                    self.assertTrue((frequency.index < requested).all())
                    self.assertTrue(binary.index.equals(frequency.index))
                    self.assertEqual(len(frequency), len(truncated))
                for actual in [requested_result, altered_result]:
                    self.assertEqual(actual.prediction_date, requested)
                    self.assertEqual(actual.forecasts, reference.forecasts)
                    self.assertTrue(actual.train_feature_dates.equals(reference.train_feature_dates))
                    self.assertTrue(actual.test_feature_dates.equals(reference.test_feature_dates))
                    self.assertLess(actual.test_feature_dates[-1] + pd.Timedelta(days=1), requested)
                    for name in reference.forecasts:
                        pd.testing.assert_frame_equal(actual.evaluations[name].per_day, reference.evaluations[name].per_day)
                        pd.testing.assert_frame_equal(actual.payouts[name].per_day, reference.payouts[name].per_day)

    def test_requested_date_requires_previous_calendar_day_before_features_or_fitting(self) -> None:
        requested = "2026-02-20"
        missing_previous = self.raw.loc[self.raw.date != pd.Timestamp("2026-02-19")]
        cases = [
            (missing_previous, requested),
            (self.raw, "2026-01-01"),  # No history before the requested date.
            (self.raw, (self.raw.date.iloc[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")),
        ]
        for raw, date in cases:
            for model in ["both", "regularized"]:
                with self.subTest(date=date, model=model), \
                        patch("src.data_loader.urlopen", return_value=io.BytesIO(raw.to_csv(index=False).encode())), \
                        patch("main.build_feature_dataset") as features, \
                        patch("main.build_count_feature_dataset") as count_features, \
                        patch.object(LGBMPredictor, "fit") as fit, \
                        patch.object(RegularizedLGBMPredictor, "fit") as regularized_fit, \
                        patch.object(BaselinePredictor, "fit") as baseline_fit, \
                        redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        run_pipeline(model=model, forecast_date=date, test_days=3, n_splits=2)
                    features.assert_not_called()
                    count_features.assert_not_called()
                    fit.assert_not_called()
                    regularized_fit.assert_not_called()
                    baseline_fit.assert_not_called()

    def test_forecast_date_requires_strict_iso_format_before_download(self) -> None:
        for invalid in ["2026-2-20", "20/02/2026", "20260220", "2026-02-30", "2026-02-20T00:00:00",
                        "2026-02-20Z", " 2026-02-20", "2026-02-20 ", "", 20260220, pd.Timestamp("2026-02-20")]:
            with self.subTest(forecast_date=invalid), patch("src.data_loader.urlopen") as download:
                with self.assertRaises(ValueError):
                    run_pipeline(forecast_date=invalid)
                download.assert_not_called()

    def test_insufficient_history_fails_before_training(self) -> None:
        payload = self.raw.head(5).to_csv(index=False).encode()
        with patch("src.data_loader.urlopen", return_value=io.BytesIO(payload)), \
                patch.object(LGBMPredictor, "fit") as fit, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Reduce --test-days"):
                run_pipeline()
            fit.assert_not_called()

    def test_invalid_options_fail_before_downloading(self) -> None:
        for options in [{"model": "unknown"}, {"top_k": 101}, {"gap": 0},
                        {"n_splits": 1}, {"n_jobs": 0}, {"test_days": -1}]:
            with self.subTest(options=options), patch("src.data_loader.urlopen") as download:
                with self.assertRaises(ValueError):
                    run_pipeline(**options)
                download.assert_not_called()

    def test_cli_defaults_and_options(self) -> None:
        with patch("main.run_pipeline") as run:
            self.assertEqual(main([]), 0)
            self.assertEqual(run.call_args.kwargs["model"], "lightgbm")
            self.assertEqual(run.call_args.kwargs["top_k"], 5)
            self.assertIsNone(run.call_args.kwargs["forecast_date"])
            self.assertEqual(main(["--model", "both", "--test-days", "10", "--inspect"]), 0)
            self.assertEqual(run.call_args.kwargs["test_days"], 10)
            self.assertTrue(run.call_args.kwargs["inspect"])
            self.assertEqual(main(["--model", "regularized", "--forecast-date", "2026-02-20"]), 0)
            self.assertEqual(run.call_args.kwargs["forecast_date"], "2026-02-20")
            self.assertEqual(run.call_args.kwargs["model"], "regularized")

    def test_cli_rejects_invalid_forecast_date_without_downloading(self) -> None:
        with patch("src.data_loader.urlopen") as download, redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--forecast-date", "2026-02-30"]), 1)
            download.assert_not_called()

    def test_cli_error_and_interrupt_exit_codes(self) -> None:
        for error, code in [(ValueError("bad data"), 1), (OSError("network failed"), 1), (KeyboardInterrupt(), 130)]:
            with self.subTest(error=error), patch("main.run_pipeline", side_effect=error), \
                    redirect_stderr(io.StringIO()) as output:
                self.assertEqual(main([]), code)
                self.assertIn("Pipeline", output.getvalue())


if __name__ == "__main__":
    unittest.main()
