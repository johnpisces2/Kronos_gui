import unittest

import pandas as pd

from paper_strategy import (
    PaperPosition,
    PaperStrategyConfig,
    build_entry_decision,
    build_exit_decision,
    build_signal_snapshot,
)


def build_payload(
    *,
    current_price=100.0,
    forecast_price=104.0,
    validation_history=(96.0, 100.0),
    validation_pred=(101.0, 103.0),
    validation_actual=(100.5, 102.0),
):
    base_times = pd.date_range("2026-04-14 00:00:00", periods=4, freq="4h")
    context_df = pd.DataFrame(
        {
            "timestamps": base_times[:2],
            "close": [98.0, current_price],
        }
    )
    future_pred_df = pd.DataFrame(
        {
            "timestamps": base_times[2:],
            "close": [102.0, forecast_price],
        }
    )
    validation_history_df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-04-11 00:00:00", periods=2, freq="4h"),
            "close": list(validation_history),
        }
    )
    validation_pred_df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-04-11 08:00:00", periods=2, freq="4h"),
            "close": list(validation_pred),
        }
    )
    validation_actual_df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-04-11 08:00:00", periods=2, freq="4h"),
            "close": list(validation_actual),
        }
    )
    return {
        "context_df": context_df,
        "future_pred_df": future_pred_df,
        "validation_history_df": validation_history_df,
        "validation_pred_df": validation_pred_df,
        "validation_actual_df": validation_actual_df,
    }


class TestPaperStrategy(unittest.TestCase):
    def test_build_signal_snapshot_computes_returns_and_deltas(self):
        snapshot = build_signal_snapshot(build_payload())

        self.assertAlmostEqual(snapshot.forecast_return, 0.04, places=6)
        self.assertAlmostEqual(snapshot.validation_pred_return, 0.03, places=6)
        self.assertEqual(snapshot.validation_history_delta, 4.0)
        self.assertEqual(snapshot.validation_pred_delta, 2.0)

    def test_entry_long_uses_half_of_forecast_edge_as_stop_distance(self):
        config = PaperStrategyConfig(
            entry_threshold_pct=0.03,
            stop_loss_fraction=0.5,
            min_stop_loss_pct=0.01,
            max_stop_loss_pct=0.05,
        )
        decision = build_entry_decision(build_signal_snapshot(build_payload()), config)

        self.assertEqual(decision.action, "enter_long")
        self.assertAlmostEqual(decision.stop_distance_pct, 0.02, places=6)
        self.assertAlmostEqual(decision.stop_price, 98.0, places=6)

    def test_entry_short_requires_reverse_alignment(self):
        payload = build_payload(
            current_price=100.0,
            forecast_price=96.0,
            validation_history=(104.0, 100.0),
            validation_pred=(99.0, 97.0),
            validation_actual=(98.0, 96.0),
        )
        decision = build_entry_decision(build_signal_snapshot(payload), PaperStrategyConfig())

        self.assertEqual(decision.action, "enter_short")
        self.assertAlmostEqual(decision.stop_price, 102.0, places=6)

    def test_exit_long_when_validation_pred_invalidates_signal(self):
        snapshot = build_signal_snapshot(
            build_payload(validation_pred=(101.0, 99.0))
        )
        position = PaperPosition(
            side="long",
            entry_price=100.0,
            stop_price=98.0,
            take_profit_price=104.0,
            entry_time=pd.Timestamp("2026-04-14 04:00:00"),
        )

        decision = build_exit_decision(position, snapshot, PaperStrategyConfig())

        self.assertEqual(decision.action, "exit_long")
        self.assertEqual(decision.reason, "long_signal_invalidated")

    def test_exit_short_when_stop_loss_is_hit(self):
        payload = build_payload(
            current_price=103.0,
            forecast_price=96.0,
            validation_history=(104.0, 100.0),
            validation_pred=(99.0, 97.0),
            validation_actual=(98.0, 96.0),
        )
        position = PaperPosition(
            side="short",
            entry_price=100.0,
            stop_price=102.0,
            take_profit_price=96.0,
            entry_time=pd.Timestamp("2026-04-14 04:00:00"),
        )

        decision = build_exit_decision(position, build_signal_snapshot(payload), PaperStrategyConfig())

        self.assertEqual(decision.action, "exit_short")
        self.assertEqual(decision.reason, "stop_loss_hit")


if __name__ == "__main__":
    unittest.main()
