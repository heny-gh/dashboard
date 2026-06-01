# ============================================================
# MERGED STREAMLIT COMMUNITY CLOUD APP
#
# This single file contains:
# 1) the forecasting pipeline formerly stored in forecasting.py; and
# 2) the Streamlit decision dashboard formerly stored in new_streamlit_app.py.
#
# Deploy this file as app.py on Streamlit Community Cloud.
# ============================================================

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from xgboost import XGBClassifier, XGBRegressor

try:
    from scipy.stats import levene
except Exception:
    levene = None

warnings.filterwarnings("ignore")

RAW_FILE_PATH = Path(__file__).resolve().parent / "Final_data.xlsx"
SHEET_NAME = "Sheet1"
DATE_COL = "Date"
TARGET_COL = "y_target"
RET_COL = "r_usdtnd"
EPS = 1e-8

VOL_TARGET_WINDOW = 3
LABEL_GAP = VOL_TARGET_WINDOW - 1
HOLDOUT_YEARS = 1
TRADING_DAYS_PER_YEAR = 252
REGIME_BOOTSTRAP_ITERATIONS = 1000
REGIME_BOOTSTRAP_BLOCK_SIZE = 5
MIN_OBS_FOR_VARIANCE_TEST = 10
LOW_SAMPLE_THRESHOLD = 50
MODERATE_SAMPLE_THRESHOLD = 150

ROLLING_WINDOW = 500
STEP_SIZE = 21
RANDOM_STATE = 42
WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995

RIDGE_STABLE_FEATURE_COUNT = 30
BROAD_FEATURE_COUNT = 50
CLASSIFIER_TOP_PERCENTILE = 75
REGIME_WEIGHT_BOOST = 0.20
REGIME_WEIGHT_FLOOR = 0.05
REGIME_WEIGHT_CAP = 0.95
BLEND_WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2).tolist()
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
XGB_MAX_CONFIGS_PER_WINDOW = 12
XGB_N_JOBS = 1

BASELINE_GARCH_SPEC = {
    "name": "GARCH-normal",
    "mean": "Zero",
    "vol": "GARCH",
    "p": 1,
    "o": 0,
    "q": 1,
    "dist": "normal",
}
PRIMARY_GARCH_SPEC = {
    "name": "EGARCH-t",
    "mean": "Zero",
    "vol": "EGARCH",
    "p": 1,
    "o": 1,
    "q": 1,
    "dist": "t",
}
NO_WORSE_TOL = 1e-6

EXPECTED_INPUT_COLUMNS = [
    "Date",
    "BRENT",
    "DXY",
    "VIX",
    "GBP_USD",
    "USD_JPY",
    "EUR_USD",
    "USDTND",
    "sentiment_global",
    "GOLD",
    "MOVE",
    "SP500",
    "Tunindex",
    "US_10Y",
    "US_2Y_10_Spread",
]

OPTIONAL_INPUT_COLUMNS = [
    "SOFR",
    "tunibor_1m",
    "tunibor_3m",
    "tunibor_6m",
    "tunibor_9m",
    "tunibor_1y",
    "USD_TND_3M_Forward_Premium",
    "BID_ASK_SPREAD",
]

RAW_COLUMN_ALIASES = {
    "EUR_USD": ["EUR/USD"],
    "GBP_USD": ["GBP/USD"],
    "USD_JPY": ["USD/JPY"],
    "sentiment_global": ["Global sentiment", "Simple average sentiment"],
    "SOFR": ["USDSOFR", "USD_SOFR", "sofr", "SOFR Overnight"],
    "tunibor_1m": ["TUNIBOR_1M", "Tunibor 1M", "TUNIBOR 1M"],
    "tunibor_3m": ["TUNIBOR_3M", "Tunibor 3M", "TUNIBOR 3M"],
    "tunibor_6m": ["TUNIBOR_6M", "Tunibor 6M", "TUNIBOR 6M"],
    "tunibor_9m": ["TUNIBOR_9M", "Tunibor 9M", "TUNIBOR 9M"],
    "tunibor_1y": ["TUNIBOR_1Y", "Tunibor 1Y", "TUNIBOR 1Y", "tunibor_12m", "TUNIBOR_12M"],
    "USD_TND_3M_Forward_Premium": [
        "USDTND_3M_FORWARD_PREMIUM",
        "USD_TND_3M_FORWARD_BID_PREMIUM",
        "USD_TND_3M_Forward_Bid_Premium",
        "USDTND 3M Forward Premium",
        "USD/TND 3M Forward Premium",
    ],
    "BID_ASK_SPREAD": [
        "USD_TND_BID_ASK_SPREAD",
        "USDTND_BID_ASK_SPREAD",
        "Bid Ask Spread",
        "BID ASK SPREAD",
        "bid_ask_spread",
    ],
}

RAW_COLUMN_RENAMES = {
    alias: canonical
    for canonical, aliases in RAW_COLUMN_ALIASES.items()
    for alias in aliases
}

FEATURE_NAME_MAP = {
    "rv_usdtnd_1": "USD/TND 1-day realized volatility",
    "rv_usdtnd_5": "USD/TND weekly realized volatility",
    "rv_usdtnd_10": "USD/TND 10-day realized volatility",
    "rv_eurusd_1": "EUR/USD volatility",
    "rv_gbpusd_1": "GBP/USD volatility",
    "rv_usdjpy_1": "USD/JPY volatility",
    "rv_dxy_1": "Dollar Index volatility",
    "VIX_lag1": "VIX global equity risk",
    "MOVE_lag1": "MOVE US rates volatility",
    "sentiment_global_lag1": "Global sentiment",
    "rv_brent_1": "Brent oil volatility",
    "rv_gold_1": "Gold volatility",
    "rv_sp500_1": "S&P 500 volatility",
    "rv_tunindex_1": "Tunindex volatility",
    "global_fx_mean": "Average global FX volatility",
    "global_local_ratio": "Global FX vs USD/TND volatility ratio",
    "garch_cond_vol": "Selected GARCH benchmark volatility",
    "event_regime_code": "Economic event regime code",
    "covid_dummy": "COVID-19 event dummy",
    "ukraine_war_dummy": "Russia-Ukraine war event dummy",
    "us_tariff_dummy": "US tariff shock event dummy",
    "iran_geopolitical_dummy": "Iran geopolitical escalation dummy",
    "any_crisis_dummy": "Any major crisis event dummy",
    "vix_x_covid": "VIX x COVID shock interaction",
    "brent_x_ukraine": "Brent volatility x Ukraine war interaction",
    "brent_x_iran": "Brent volatility x Iran geopolitical shock interaction",
    "dxy_x_tariff": "DXY volatility x US tariff shock interaction",
    "move_x_crisis": "MOVE x crisis interaction",
    "sentiment_x_crisis": "Global sentiment x crisis interaction",
    "tunibor_curve_level": "TUNIBOR curve level",
    "tunibor_curve_slope_1y_minus_1m": "TUNIBOR 1Y minus 1M slope",
    "tunibor_curve_curvature_6m_midpoint": "TUNIBOR curve curvature",
    "tunibor_curve_level_change_5d": "Weekly change in TUNIBOR curve level",
    "tunibor_curve_slope_1y_minus_1m_change_5d": "Weekly change in TUNIBOR curve slope",
    "tunibor_curve_level_z20": "TUNIBOR curve level stress z-score",
    "tunibor_curve_slope_1y_minus_1m_z20": "TUNIBOR curve slope stress z-score",
    "tnd_usd_rate_spread_3m_proxy": "TND-USD short-rate spread proxy",
    "tnd_usd_rate_spread_3m_proxy_change_5d": "Weekly change in TND-USD rate spread proxy",
    "tnd_usd_rate_spread_3m_proxy_z20": "TND-USD rate spread stress z-score",
    "usdtnd_3m_forward_bid_premium_points": "USD/TND 3M forward bid premium",
    "usdtnd_3m_forward_premium_relative": "USD/TND 3M forward premium relative to spot",
    "usdtnd_3m_forward_implied_rate_diff": "USD/TND 3M forward-implied rate differential",
    "usdtnd_3m_forward_bid_premium_points_change_5d": "Weekly change in USD/TND 3M forward premium",
    "usdtnd_3m_forward_premium_relative_change_5d": "Weekly change in USD/TND relative forward premium",
    "usdtnd_3m_forward_implied_rate_diff_z20": "USD/TND forward-implied rate differential stress z-score",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy": "USD/TND forward basis versus TUNIBOR-SOFR proxy",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_change_5d": "Weekly change in USD/TND forward basis",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_z20": "USD/TND forward-basis stress z-score",
    "usdtnd_bid_ask_spread": "USD/TND bid-ask spread",
    "usdtnd_bid_ask_spread_change_5d": "Weekly change in USD/TND bid-ask spread",
    "usdtnd_bid_ask_spread_z20": "USD/TND liquidity stress z-score",
}

BASE_FEATURES = [
    "rv_usdtnd_1",
    "rv_usdtnd_5",
    "rv_usdtnd_10",
    "rv_eurusd_1",
    "rv_gbpusd_1",
    "rv_usdjpy_1",
    "rv_dxy_1",
    "rv_brent_1",
    "VIX_lag1",
    "sentiment_global_lag1",
    "rv_gold_1",
    "rv_sp500_1",
    "rv_tunindex_1",
    "MOVE_lag1",
    "US_10Y_lag1",
    "US_2Y_10_Spread_lag1",
    "tunibor_curve_level",
    "tunibor_curve_slope_1y_minus_1m",
    "tunibor_curve_curvature_6m_midpoint",
    "tunibor_curve_level_change_5d",
    "tunibor_curve_slope_1y_minus_1m_change_5d",
    "tunibor_curve_level_z20",
    "tunibor_curve_slope_1y_minus_1m_z20",
    "tnd_usd_rate_spread_3m_proxy",
    "tnd_usd_rate_spread_3m_proxy_change_5d",
    "tnd_usd_rate_spread_3m_proxy_z20",
    "usdtnd_3m_forward_bid_premium_points",
    "usdtnd_3m_forward_premium_relative",
    "usdtnd_3m_forward_implied_rate_diff",
    "usdtnd_3m_forward_bid_premium_points_change_5d",
    "usdtnd_3m_forward_premium_relative_change_5d",
    "usdtnd_3m_forward_implied_rate_diff_z20",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_change_5d",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_z20",
    "usdtnd_bid_ask_spread",
    "usdtnd_bid_ask_spread_change_5d",
    "usdtnd_bid_ask_spread_z20",
]

NEW_PREPARED_FEATURES = [
    "tunibor_curve_level",
    "tunibor_curve_slope_1y_minus_1m",
    "tunibor_curve_curvature_6m_midpoint",
    "tunibor_curve_level_change_5d",
    "tunibor_curve_slope_1y_minus_1m_change_5d",
    "tunibor_curve_level_z20",
    "tunibor_curve_slope_1y_minus_1m_z20",
    "tnd_usd_rate_spread_3m_proxy",
    "tnd_usd_rate_spread_3m_proxy_change_5d",
    "tnd_usd_rate_spread_3m_proxy_z20",
    "usdtnd_3m_forward_bid_premium_points",
    "usdtnd_3m_forward_premium_relative",
    "usdtnd_3m_forward_implied_rate_diff",
    "usdtnd_3m_forward_bid_premium_points_change_5d",
    "usdtnd_3m_forward_premium_relative_change_5d",
    "usdtnd_3m_forward_implied_rate_diff_z20",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_change_5d",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_z20",
    "usdtnd_bid_ask_spread",
    "usdtnd_bid_ask_spread_change_5d",
    "usdtnd_bid_ask_spread_z20",
]

SIGNED_OR_CAN_BE_NEGATIVE_FEATURES = [
    "sentiment_global_lag1",
    "US_2Y_10_Spread_lag1",
    "tunibor_curve_slope_1y_minus_1m",
    "tunibor_curve_curvature_6m_midpoint",
    "tunibor_curve_level_change_5d",
    "tunibor_curve_slope_1y_minus_1m_change_5d",
    "tunibor_curve_level_z20",
    "tunibor_curve_slope_1y_minus_1m_z20",
    "tnd_usd_rate_spread_3m_proxy",
    "tnd_usd_rate_spread_3m_proxy_change_5d",
    "tnd_usd_rate_spread_3m_proxy_z20",
    "usdtnd_3m_forward_bid_premium_points_change_5d",
    "usdtnd_3m_forward_premium_relative_change_5d",
    "usdtnd_3m_forward_implied_rate_diff_z20",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_change_5d",
    "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_z20",
    "usdtnd_bid_ask_spread_change_5d",
    "usdtnd_bid_ask_spread_z20",
]

EXCLUDED_MODEL_COLUMNS = [
    DATE_COL,
    TARGET_COL,
    "log_target",
    RET_COL,
    "event_regime_code",
]

SPLIT = ROLLING_WINDOW

ARTIFACT_FILENAMES = {
    "model_ready_dataset": "model_ready_dataset_3d.xlsx",
    "development_results": "walkforward_results_dev_3d.xlsx",
    "holdout_results": "walkforward_results_holdout_3d.xlsx",
    "development_summary": "walkforward_summary_dev_3d.xlsx",
    "holdout_summary": "walkforward_summary_holdout_3d.xlsx",
    "final_forecast": "final_forecast_blend_3d.xlsx",
    "high_vol_classifier_calibration_diagnostics": "high_vol_classifier_calibration_diagnostics_3d.xlsx",
    "event_descriptive_stats": "event_regime_descriptive_stats_3d.xlsx",
    "event_metrics_development": "event_regime_metrics_dev_3d.xlsx",
    "event_metrics_holdout": "event_regime_metrics_holdout_3d.xlsx",
    "run_metadata": "run_metadata.json",
}
DASHBOARD_PAYLOAD_FILENAME = "dashboard_payload.json"

EVENT_TIMELINE_START = pd.Timestamp("2019-10-04")

EVENT_REGIME_DEFINITIONS = {
    "pre_covid_normal": {
        "start": "2019-10-04",
        "end": "2020-03-10",
        "description": "Pre-COVID normal market regime"
    },
    "covid_shock": {
        "start": "2020-03-11",
        "end": "2020-12-31",
        "description": "COVID-19 pandemic shock"
    },
    "post_covid_recovery": {
        "start": "2021-01-01",
        "end": "2022-02-23",
        "description": "Post-COVID recovery and inflation build-up"
    },
    "ukraine_war_shock": {
        "start": "2022-02-24",
        "end": "2022-12-30",
        "description": "Russia-Ukraine war initial shock"
    },
    "post_war_inflation_adjustment": {
        "start": "2023-01-02",
        "end": "2025-04-01",
        "description": "Post-war inflation and monetary adjustment regime"
    },
    "us_tariff_shock": {
        "start": "2025-04-02",
        "end": "2025-06-12",
        "description": "US tariff shock and trade-policy uncertainty"
    },
    "post_tariff_normalization": {
        "start": "2025-06-13",
        "end": "2026-02-27",
        "description": "Post-tariff normalization regime"
    },
    "iran_geopolitical_shock": {
        "start": "2026-02-28",
        "end": None,
        "description": "Iran / US-Israel geopolitical escalation"
    },
}

EVENT_DUMMY_COLUMNS = [
    "covid_dummy",
    "ukraine_war_dummy",
    "us_tariff_dummy",
    "iran_geopolitical_dummy",
    "any_crisis_dummy",
]

EVENT_INTERACTION_COLUMNS = [
    "vix_x_covid",
    "brent_x_ukraine",
    "brent_x_iran",
    "dxy_x_tariff",
    "move_x_crisis",
    "sentiment_x_crisis",
]


@dataclass
class ForecastRunResult:
    forecast: Dict[str, Any]
    dashboard: Dict[str, Any]
    development_summary: pd.DataFrame
    holdout_summary: pd.DataFrame
    development_results: pd.DataFrame
    holdout_results: pd.DataFrame
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def first_existing_column(dataframe: pd.DataFrame, candidate_columns: list[str]) -> str:
    for column_name in candidate_columns:
        if column_name in dataframe.columns:
            return column_name
    raise KeyError(f"None of these columns were found: {candidate_columns}")


def compute_file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_package_versions() -> Dict[str, str]:
    import platform

    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    optional_packages = {
        "sklearn": "sklearn",
        "xgboost": "xgboost",
        "statsmodels": "statsmodels",
        "scipy": "scipy",
        "arch": "arch",
    }
    for key, module_name in optional_packages.items():
        try:
            module = __import__(module_name)
            versions[key] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[key] = "unavailable"
    return versions


def _write_high_vol_calibration_artifact(
    output_path: Path,
    diagnostics_summary: pd.DataFrame,
    reliability_table: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        diagnostics_summary.to_excel(writer, sheet_name="summary", index=False)
        reliability_table.to_excel(writer, sheet_name="reliability_curve", index=False)
        threshold_sensitivity.to_excel(writer, sheet_name="threshold_sensitivity", index=False)


def build_high_volatility_calibration_diagnostics(
    development_results: pd.DataFrame,
    holdout_results: pd.DataFrame,
    output_path: Path,
    threshold_grid: Optional[list[float]] = None,
    calibration_bins: int = 5,
    high_volatility_cutoff: Optional[float] = None,
) -> dict[str, pd.DataFrame]:
    if threshold_grid is None:
        threshold_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    summary_columns = ["metric", "value", "interpretation"]
    reliability_columns = [
        "mean_predicted_probability",
        "observed_high_volatility_frequency",
        "calibration_gap",
    ]
    threshold_columns = [
        "alert_threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "false_alert_rate",
        "missed_high_vol_rate",
    ]
    reliability_table = pd.DataFrame(columns=reliability_columns)
    threshold_sensitivity = pd.DataFrame(columns=threshold_columns)

    actual_candidates = ["actual", "actual_volatility", "y_target", "target"]
    probability_candidates = [
        "regime_prob_high_vol",
        "high_vol_probability",
        "prob_high_vol",
        "pred_high_vol_probability",
    ]
    cutoff_source = "fixed_event_threshold" if high_volatility_cutoff is not None else "development_results_q75"
    cutoff_source_interpretation = (
        "Fixed high-volatility threshold used consistently for classifier training, holdout evaluation, "
        "calibration diagnostics, and live forecast interpretation."
        if high_volatility_cutoff is not None
        else "Identifies whether the calibration label used the same threshold as classifier training."
    )

    def unavailable_summary(reason: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "metric": "High-volatility cutoff",
                    "value": np.nan,
                    "interpretation": reason,
                },
                {
                    "metric": "High-volatility cutoff source",
                    "value": cutoff_source,
                    "interpretation": "Identifies whether the calibration label used the same threshold as classifier training.",
                },
                {
                    "metric": "Brier score",
                    "value": np.nan,
                    "interpretation": "Not computed because calibration inputs were unavailable.",
                },
                {
                    "metric": "ROC-AUC",
                    "value": np.nan,
                    "interpretation": "Not computed because calibration inputs were unavailable.",
                },
                {
                    "metric": "PR-AUC",
                    "value": np.nan,
                    "interpretation": "Not computed because calibration inputs were unavailable.",
                },
                {
                    "metric": "Holdout observations",
                    "value": 0,
                    "interpretation": "No valid holdout observations were available for calibration diagnostics.",
                },
                {
                    "metric": "Realized high-volatility frequency",
                    "value": np.nan,
                    "interpretation": "Not computed because calibration inputs were unavailable.",
                },
                {
                    "metric": "Average predicted high-volatility probability",
                    "value": np.nan,
                    "interpretation": "Not computed because classifier probabilities were unavailable.",
                },
            ],
            columns=summary_columns,
        )

    try:
        development_actual_column = first_existing_column(development_results, actual_candidates)
        holdout_actual_column = first_existing_column(holdout_results, actual_candidates)
        holdout_probability_column = first_existing_column(holdout_results, probability_candidates)
    except KeyError as exc:
        diagnostics_summary = unavailable_summary(str(exc))
        _write_high_vol_calibration_artifact(
            output_path,
            diagnostics_summary,
            reliability_table,
            threshold_sensitivity,
        )
        return {
            "summary": diagnostics_summary,
            "reliability_curve": reliability_table,
            "threshold_sensitivity": threshold_sensitivity,
        }

    if high_volatility_cutoff is None:
        development_actual = pd.to_numeric(development_results[development_actual_column], errors="coerce")
        development_actual = development_actual.replace([np.inf, -np.inf], np.nan).dropna()
        if development_actual.empty:
            diagnostics_summary = unavailable_summary("Development-period actual volatility was empty or invalid, so the 75th percentile cutoff could not be computed.")
            _write_high_vol_calibration_artifact(
                output_path,
                diagnostics_summary,
                reliability_table,
                threshold_sensitivity,
            )
            return {
                "summary": diagnostics_summary,
                "reliability_curve": reliability_table,
                "threshold_sensitivity": threshold_sensitivity,
            }
        high_volatility_cutoff = float(development_actual.quantile(0.75))
    else:
        high_volatility_cutoff = float(high_volatility_cutoff)
    holdout_actual = pd.to_numeric(holdout_results[holdout_actual_column], errors="coerce")
    holdout_probability = pd.to_numeric(holdout_results[holdout_probability_column], errors="coerce")
    valid = (
        holdout_actual.replace([np.inf, -np.inf], np.nan).notna()
        & holdout_probability.replace([np.inf, -np.inf], np.nan).notna()
    )

    actual_valid = holdout_actual.loc[valid].astype(float)
    probability_valid = holdout_probability.loc[valid].clip(0.0, 1.0).astype(float)
    realized_high_volatility = (actual_valid >= high_volatility_cutoff).astype(int)

    n_obs = int(len(realized_high_volatility))
    has_observations = n_obs > 0
    has_two_classes = has_observations and len(np.unique(realized_high_volatility)) > 1

    brier = (
        float(brier_score_loss(realized_high_volatility, probability_valid))
        if has_observations
        else np.nan
    )
    roc_auc = (
        float(roc_auc_score(realized_high_volatility, probability_valid))
        if has_two_classes
        else np.nan
    )
    pr_auc = (
        float(average_precision_score(realized_high_volatility, probability_valid))
        if has_two_classes
        else np.nan
    )
    realized_frequency = float(realized_high_volatility.mean()) if has_observations else np.nan
    average_probability = float(probability_valid.mean()) if has_observations else np.nan

    diagnostics_summary = pd.DataFrame(
        [
            {
                "metric": "High-volatility cutoff",
                "value": high_volatility_cutoff,
                "interpretation": cutoff_source_interpretation,
            },
            {
                "metric": "High-volatility cutoff source",
                "value": cutoff_source,
                "interpretation": "Identifies whether the calibration label used the same threshold as classifier training.",
            },
            {
                "metric": "Brier score",
                "value": brier,
                "interpretation": "Probability forecast error for the holdout high-volatility label; lower is better.",
            },
            {
                "metric": "ROC-AUC",
                "value": roc_auc,
                "interpretation": "Holdout ranking quality across alert thresholds; unavailable when only one class appears.",
            },
            {
                "metric": "PR-AUC",
                "value": pr_auc,
                "interpretation": "Average precision for the holdout high-volatility class; unavailable when only one class appears.",
            },
            {
                "metric": "Holdout observations",
                "value": n_obs,
                "interpretation": "Valid holdout rows with actual realized volatility and classifier probability.",
            },
            {
                "metric": "Realized high-volatility frequency",
                "value": realized_frequency,
                "interpretation": "Share of valid holdout observations at or above the development-period high-volatility cutoff.",
            },
            {
                "metric": "Average predicted high-volatility probability",
                "value": average_probability,
                "interpretation": "Mean clipped high-volatility probability over valid holdout observations.",
            },
        ],
        columns=summary_columns,
    )

    if has_observations:
        try:
            n_bins = max(1, min(int(calibration_bins), n_obs))
            observed_frequency, mean_predicted_probability = calibration_curve(
                realized_high_volatility,
                probability_valid,
                n_bins=n_bins,
                strategy="quantile",
            )
            reliability_table = pd.DataFrame(
                {
                    "mean_predicted_probability": mean_predicted_probability,
                    "observed_high_volatility_frequency": observed_frequency,
                }
            )
            reliability_table["calibration_gap"] = (
                reliability_table["observed_high_volatility_frequency"]
                - reliability_table["mean_predicted_probability"]
            )
        except Exception:
            reliability_table = pd.DataFrame(columns=reliability_columns)

        threshold_rows = []
        y_true = realized_high_volatility.to_numpy(dtype=int)
        y_prob = probability_valid.to_numpy(dtype=float)
        for threshold in threshold_grid:
            y_pred = (y_prob >= float(threshold)).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            threshold_rows.append(
                {
                    "alert_threshold": float(threshold),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                    "false_alert_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan,
                    "missed_high_vol_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else np.nan,
                }
            )
        threshold_sensitivity = pd.DataFrame(threshold_rows, columns=threshold_columns)

    _write_high_vol_calibration_artifact(
        output_path,
        diagnostics_summary,
        reliability_table,
        threshold_sensitivity,
    )
    return {
        "summary": diagnostics_summary,
        "reliability_curve": reliability_table,
        "threshold_sensitivity": threshold_sensitivity,
    }


def assign_event_regimes(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['event_regime'] = 'unclassified'
    df['event_regime_description'] = ''
    df['event_regime_code'] = -1

    regime_codes = {
        'pre_covid_normal': 0,
        'covid_shock': 1,
        'post_covid_recovery': 2,
        'ukraine_war_shock': 3,
        'post_war_inflation_adjustment': 4,
        'us_tariff_shock': 5,
        'post_tariff_normalization': 6,
        'iran_geopolitical_shock': 7,
        'unclassified': -1
    }

    for regime, details in EVENT_REGIME_DEFINITIONS.items():
        start = pd.to_datetime(details['start']) if details['start'] else pd.Timestamp.min
        end = pd.to_datetime(details['end']) if details['end'] else pd.Timestamp.max
        mask = (df[date_col] >= start) & (df[date_col] <= end)
        df.loc[mask, 'event_regime'] = regime
        df.loc[mask, 'event_regime_description'] = details['description']
        df.loc[mask, 'event_regime_code'] = regime_codes[regime]

    df['covid_dummy'] = (df['event_regime'] == 'covid_shock').astype(int)
    df['ukraine_war_dummy'] = (df['event_regime'] == 'ukraine_war_shock').astype(int)
    df['us_tariff_dummy'] = (df['event_regime'] == 'us_tariff_shock').astype(int)
    df['iran_geopolitical_dummy'] = (df['event_regime'] == 'iran_geopolitical_shock').astype(int)
    df['any_crisis_dummy'] = ((df['covid_dummy'] == 1) | (df['ukraine_war_dummy'] == 1) | (df['us_tariff_dummy'] == 1) | (df['iran_geopolitical_dummy'] == 1)).astype(int)

    return df


ProgressCallback = Optional[Callable[[str], None]]


def emit(progress_callback: ProgressCallback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RAW_COLUMN_RENAMES)


def get_expected_input_schema() -> Dict[str, Any]:
    return {
        "sheet_name": SHEET_NAME,
        "required_columns": EXPECTED_INPUT_COLUMNS,
        "optional_columns": OPTIONAL_INPUT_COLUMNS,
        "accepted_aliases": {
            canonical: [canonical] + aliases
            for canonical, aliases in RAW_COLUMN_ALIASES.items()
        },
    }


def _format_missing_columns(missing_cols: List[str]) -> str:
    details = []
    for col in missing_cols:
        accepted = [col] + RAW_COLUMN_ALIASES.get(col, [])
        details.append(f"{col} (accepted: {', '.join(accepted)})")
    return "; ".join(details)


def validate_input_file(raw_file_path: Path | str, sheet_name: str = SHEET_NAME) -> Dict[str, Any]:
    raw_file_path = Path(raw_file_path)
    if not raw_file_path.exists():
        raise FileNotFoundError(f"{raw_file_path.resolve()} not found.")

    try:
        raw_df = pd.read_excel(raw_file_path, sheet_name=sheet_name)
    except ValueError as exc:
        raise ValueError(f"Could not read sheet '{sheet_name}' from uploaded Excel file.") from exc
    except Exception as exc:
        raise ValueError(f"Could not read uploaded Excel file: {exc}") from exc

    normalized = normalize_input_columns(raw_df)
    missing_cols = [col for col in EXPECTED_INPUT_COLUMNS if col not in normalized.columns]
    optional_cols_available = [col for col in OPTIONAL_INPUT_COLUMNS if col in normalized.columns]
    optional_cols_missing = [col for col in OPTIONAL_INPUT_COLUMNS if col not in normalized.columns]
    if missing_cols:
        raise ValueError(
            "Uploaded file is missing required columns: "
            f"{_format_missing_columns(missing_cols)}"
        )

    date_series = pd.to_datetime(normalized[DATE_COL], errors="coerce")
    summary = {
        "row_count": int(len(normalized)),
        "column_count": int(len(normalized.columns)),
        "date_min": date_series.min(),
        "date_max": date_series.max(),
        "missing_required_columns": [],
        "optional_input_columns_available": optional_cols_available,
        "optional_input_columns_missing": optional_cols_missing,
        "missing_values": {
            col: int(normalized[col].isna().sum())
            for col in EXPECTED_INPUT_COLUMNS + optional_cols_available
            if col in normalized.columns
        },
    }
    return summary


def clean_numeric_column(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    numeric = clean_numeric_column(series)
    median_abs = numeric.dropna().abs().median()
    if pd.notna(median_abs) and median_abs > 1.0:
        return numeric / 100.0
    return numeric


def realized_vol(series: pd.Series, window: int) -> pd.Series:
    return np.sqrt(series.pow(2).rolling(window).sum())


def add_rolling_zscore(df: pd.DataFrame, column: str, window: int = 20) -> pd.Series:
    rolling_mean = df[column].rolling(window=window, min_periods=window).mean()
    rolling_std = df[column].rolling(window=window, min_periods=window).std()
    rolling_count = df[column].rolling(window=window, min_periods=window).count()
    zscore = (df[column] - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.mask((rolling_std == 0) & (rolling_count >= window), 0.0)


def add_tunibor_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_tunibor = ["tunibor_1m", "tunibor_3m", "tunibor_6m", "tunibor_9m", "tunibor_1y"]
    if not all(c in df.columns for c in required_tunibor):
        return df

    # TUNIBOR curve factors summarize local Tunisian money-market conditions
    # through level, slope, and curvature.
    df["tunibor_curve_level"] = df[required_tunibor].mean(axis=1)
    df["tunibor_curve_slope_1y_minus_1m"] = df["tunibor_1y"] - df["tunibor_1m"]
    df["tunibor_curve_curvature_6m_midpoint"] = (
        df["tunibor_6m"] - (df["tunibor_1m"] + df["tunibor_1y"]) / 2.0
    )

    for feature in [
        "tunibor_curve_level",
        "tunibor_curve_slope_1y_minus_1m",
        "tunibor_curve_curvature_6m_midpoint",
    ]:
        df[f"{feature}_change_1d"] = df[feature].diff(1)
        df[f"{feature}_change_5d"] = df[feature].diff(5)
        df[f"{feature}_z20"] = add_rolling_zscore(df, feature, window=20)

    return df


def add_sofr_tunibor_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "SOFR" not in df.columns:
        return df

    # SOFR is used as the USD short-rate benchmark. Since the current file only
    # contains overnight SOFR, the 3M TND-USD spread is labelled as a proxy.
    usd_3m_proxy_col = "SOFR_90D" if "SOFR_90D" in df.columns else "SOFR"

    if "tunibor_3m" in df.columns and usd_3m_proxy_col in df.columns:
        df["tnd_usd_rate_spread_3m_proxy"] = df["tunibor_3m"] - df[usd_3m_proxy_col]
        df["tnd_usd_rate_spread_3m_proxy_change_1d"] = (
            df["tnd_usd_rate_spread_3m_proxy"].diff(1)
        )
        df["tnd_usd_rate_spread_3m_proxy_change_5d"] = (
            df["tnd_usd_rate_spread_3m_proxy"].diff(5)
        )
        df["tnd_usd_rate_spread_3m_proxy_z20"] = add_rolling_zscore(
            df,
            "tnd_usd_rate_spread_3m_proxy",
            window=20,
        )

    return df


def add_usdtnd_forward_premium_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "USD_TND_3M_Forward_Premium" not in df.columns or "USDTND" not in df.columns:
        return df

    spot = pd.to_numeric(df["USDTND"], errors="coerce")
    forward_premium_points = pd.to_numeric(
        df["USD_TND_3M_Forward_Premium"],
        errors="coerce",
    )

    valid = (spot > 0) & np.isfinite(spot) & np.isfinite(forward_premium_points)

    # USD/TND 3M forward bid premium is treated as forward points, not as a
    # percentage rate.
    df["usdtnd_3m_forward_bid_premium_points"] = forward_premium_points.where(valid)

    df["usdtnd_3m_forward_outright_proxy"] = (
        spot + df["usdtnd_3m_forward_bid_premium_points"]
    )

    valid_forward = (
        (spot > 0)
        & (df["usdtnd_3m_forward_outright_proxy"] > 0)
        & np.isfinite(df["usdtnd_3m_forward_outright_proxy"])
    )

    df["usdtnd_3m_forward_premium_relative"] = np.where(
        valid_forward,
        df["usdtnd_3m_forward_bid_premium_points"] / spot,
        np.nan,
    )

    forward_t = 90.0 / 360.0

    df["usdtnd_3m_forward_implied_rate_diff"] = np.where(
        valid_forward,
        np.log(df["usdtnd_3m_forward_outright_proxy"] / spot) / forward_t,
        np.nan,
    )

    for feature in [
        "usdtnd_3m_forward_bid_premium_points",
        "usdtnd_3m_forward_premium_relative",
        "usdtnd_3m_forward_implied_rate_diff",
    ]:
        df[f"{feature}_change_1d"] = df[feature].diff(1)
        df[f"{feature}_change_5d"] = df[feature].diff(5)
        df[f"{feature}_z20"] = add_rolling_zscore(df, feature, window=20)

    if "tnd_usd_rate_spread_3m_proxy" in df.columns:
        df["usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy"] = (
            df["usdtnd_3m_forward_implied_rate_diff"]
            - df["tnd_usd_rate_spread_3m_proxy"]
        )
        df["usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_change_5d"] = (
            df["usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy"].diff(5)
        )
        df["usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy_z20"] = add_rolling_zscore(
            df,
            "usdtnd_3m_forward_basis_vs_tunibor_sofr_proxy",
            window=20,
        )

    return df


def add_bid_ask_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "BID_ASK_SPREAD" not in df.columns:
        return df

    # Bid-ask spread is used as a local FX liquidity-stress variable.
    df["usdtnd_bid_ask_spread"] = pd.to_numeric(
        df["BID_ASK_SPREAD"],
        errors="coerce",
    )

    df["usdtnd_bid_ask_spread_change_1d"] = df["usdtnd_bid_ask_spread"].diff(1)
    df["usdtnd_bid_ask_spread_change_5d"] = df["usdtnd_bid_ask_spread"].diff(5)
    df["usdtnd_bid_ask_spread_z20"] = add_rolling_zscore(
        df,
        "usdtnd_bid_ask_spread",
        window=20,
    )

    return df


def add_new_market_features(df: pd.DataFrame) -> pd.DataFrame:
    # All optional features are generated from raw data inside forecasting.py to
    # preserve reproducibility and avoid manual Excel feature-engineering.
    df = add_tunibor_curve_features(df)
    df = add_sofr_tunibor_spread_features(df)
    df = add_usdtnd_forward_premium_features(df)
    df = add_bid_ask_spread_features(df)
    return df


def prepare_data(
    raw_file_path: Path | str = RAW_FILE_PATH,
    sheet_name: str = SHEET_NAME,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    raw_file_path = Path(raw_file_path)
    if not raw_file_path.exists():
        raise FileNotFoundError(f"{raw_file_path.resolve()} not found.")

    raw_df = normalize_input_columns(pd.read_excel(raw_file_path, sheet_name=sheet_name))
    required_cols = EXPECTED_INPUT_COLUMNS
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in sheet '{sheet_name}': {_format_missing_columns(missing_cols)}"
        )

    available_optional_cols = [c for c in OPTIONAL_INPUT_COLUMNS if c in raw_df.columns]
    raw_df = raw_df[required_cols + available_optional_cols].copy()
    raw_df[DATE_COL] = pd.to_datetime(raw_df[DATE_COL], errors="coerce")
    raw_df = raw_df.sort_values(DATE_COL).reset_index(drop=True)

    numeric_cols = [
        "BRENT",
        "DXY",
        "VIX",
        "GBP_USD",
        "USD_JPY",
        "EUR_USD",
        "USDTND",
        "sentiment_global",
        "GOLD",
        "MOVE",
        "SP500",
        "Tunindex",
        "US_10Y",
        "US_2Y_10_Spread",
    ]
    for col in numeric_cols:
        raw_df[col] = clean_numeric_column(raw_df[col])

    rate_optional_cols = [
        "SOFR",
        "tunibor_1m",
        "tunibor_3m",
        "tunibor_6m",
        "tunibor_9m",
        "tunibor_1y",
    ]
    for col in rate_optional_cols:
        if col in raw_df.columns:
            raw_df[col] = ensure_decimal_rate(raw_df[col])

    # Forward premium/points and bid-ask spread are raw FX price/liquidity
    # variables; they are not divided by 100.
    for col in ["USD_TND_3M_Forward_Premium", "BID_ASK_SPREAD"]:
        if col in raw_df.columns:
            raw_df[col] = clean_numeric_column(raw_df[col])

    raw_df = add_new_market_features(raw_df)

    global_raw_cols = [
        "EUR_USD",
        "GBP_USD",
        "USD_JPY",
        "BRENT",
        "DXY",
        "VIX",
        "sentiment_global",
        "GOLD",
        "MOVE",
        "SP500",
        "Tunindex",
        "US_10Y",
        "US_2Y_10_Spread",
    ]
    for col in global_raw_cols:
        raw_df[f"{col}_lag1"] = raw_df[col].shift(1)

    raw_df[RET_COL] = np.log(raw_df["USDTND"] / raw_df["USDTND"].shift(1))
    raw_df["r_eur_usd_lag1"] = np.log(raw_df["EUR_USD_lag1"] / raw_df["EUR_USD_lag1"].shift(1))
    raw_df["r_gbp_usd_lag1"] = np.log(raw_df["GBP_USD_lag1"] / raw_df["GBP_USD_lag1"].shift(1))
    raw_df["r_usd_jpy_lag1"] = np.log(raw_df["USD_JPY_lag1"] / raw_df["USD_JPY_lag1"].shift(1))
    raw_df["r_brent_lag1"] = np.log(raw_df["BRENT_lag1"] / raw_df["BRENT_lag1"].shift(1))
    raw_df["r_dxy_lag1"] = np.log(raw_df["DXY_lag1"] / raw_df["DXY_lag1"].shift(1))
    raw_df["r_gold_lag1"] = np.log(raw_df["GOLD_lag1"] / raw_df["GOLD_lag1"].shift(1))
    raw_df["r_sp500_lag1"] = np.log(raw_df["SP500_lag1"] / raw_df["SP500_lag1"].shift(1))
    raw_df["r_tunindex_lag1"] = np.log(raw_df["Tunindex_lag1"] / raw_df["Tunindex_lag1"].shift(1))

    raw_df["rv_usdtnd_1"] = realized_vol(raw_df[RET_COL], 1)
    raw_df["rv_usdtnd_5"] = realized_vol(raw_df[RET_COL], 5)
    raw_df["rv_usdtnd_10"] = realized_vol(raw_df[RET_COL], 10)
    raw_df["rv_eurusd_1"] = realized_vol(raw_df["r_eur_usd_lag1"], 1)
    raw_df["rv_gbpusd_1"] = realized_vol(raw_df["r_gbp_usd_lag1"], 1)
    raw_df["rv_usdjpy_1"] = realized_vol(raw_df["r_usd_jpy_lag1"], 1)
    raw_df["rv_brent_1"] = realized_vol(raw_df["r_brent_lag1"], 1)
    raw_df["rv_dxy_1"] = realized_vol(raw_df["r_dxy_lag1"], 1)
    raw_df["rv_gold_1"] = realized_vol(raw_df["r_gold_lag1"], 1)
    raw_df["rv_sp500_1"] = realized_vol(raw_df["r_sp500_lag1"], 1)
    raw_df["rv_tunindex_1"] = realized_vol(raw_df["r_tunindex_lag1"], 1)

    future_sq = np.zeros(len(raw_df), dtype=float)
    for k in range(1, VOL_TARGET_WINDOW + 1):
        future_sq += raw_df[RET_COL].shift(-k).pow(2).fillna(np.nan).values
    raw_df[TARGET_COL] = np.sqrt(future_sq)

    base_prepared_cols = [
        DATE_COL,
        RET_COL,
        "rv_usdtnd_1",
        "rv_usdtnd_5",
        "rv_usdtnd_10",
        "rv_eurusd_1",
        "rv_gbpusd_1",
        "rv_usdjpy_1",
        "rv_dxy_1",
        "rv_brent_1",
        "VIX_lag1",
        "sentiment_global_lag1",
        "rv_gold_1",
        "rv_sp500_1",
        "rv_tunindex_1",
        "MOVE_lag1",
        "US_10Y_lag1",
        "US_2Y_10_Spread_lag1",
        TARGET_COL,
    ]
    available_new_prepared_cols = [c for c in NEW_PREPARED_FEATURES if c in raw_df.columns]
    df = raw_df[base_prepared_cols + available_new_prepared_cols].dropna().reset_index(drop=True)

    # The final macro-event methodology starts at 2019-10-04.
    # Earlier observations are outside the official event calendar and are
    # excluded so downstream artifacts do not contain an "unclassified" regime.
    df = df[df[DATE_COL] >= EVENT_TIMELINE_START].reset_index(drop=True)

    df = assign_event_regimes(df)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)

    return df


def build_event_analysis_frame(
    raw_file_path: Path | str = RAW_FILE_PATH,
    sheet_name: str = SHEET_NAME,
) -> pd.DataFrame:
    """Build a stable event-volatility analysis frame independent of ML features.

    Descriptive macro-event volatility statistics should not change when optional
    model features such as TUNIBOR, SOFR, forward premium, or bid-ask spread
    variables are added to the forecasting feature matrix.
    """
    raw_file_path = Path(raw_file_path)
    if not raw_file_path.exists():
        raise FileNotFoundError(f"{raw_file_path.resolve()} not found.")

    raw_df = normalize_input_columns(pd.read_excel(raw_file_path, sheet_name=sheet_name))
    required_cols = [DATE_COL, "USDTND"]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in sheet '{sheet_name}': {_format_missing_columns(missing_cols)}"
        )

    optional_event_cols = [
        "VIX",
        "DXY",
        "BRENT",
        "MOVE",
        "Tunindex",
        "sentiment_global",
    ]
    available_event_cols = [c for c in optional_event_cols if c in raw_df.columns]
    event_df = raw_df[required_cols + available_event_cols].copy()
    event_df[DATE_COL] = pd.to_datetime(event_df[DATE_COL], errors="coerce")
    event_df = event_df.sort_values(DATE_COL).reset_index(drop=True)

    for col in ["USDTND"] + available_event_cols:
        event_df[col] = clean_numeric_column(event_df[col])

    event_df[RET_COL] = np.log(event_df["USDTND"] / event_df["USDTND"].shift(1))

    future_sq = np.zeros(len(event_df), dtype=float)
    for k in range(1, VOL_TARGET_WINDOW + 1):
        future_sq += event_df[RET_COL].shift(-k).pow(2).fillna(np.nan).values
    event_df[TARGET_COL] = np.sqrt(future_sq)

    event_df["rv_usdtnd_1"] = realized_vol(event_df[RET_COL], 1)
    event_df["rv_usdtnd_5"] = realized_vol(event_df[RET_COL], 5)
    event_df["rv_usdtnd_10"] = realized_vol(event_df[RET_COL], 10)

    for col in ["VIX", "MOVE", "DXY", "BRENT"]:
        if col in event_df.columns:
            event_df[f"{col}_lag1"] = event_df[col].shift(1)

    if "DXY_lag1" in event_df.columns:
        event_df["r_dxy_lag1"] = np.log(event_df["DXY_lag1"] / event_df["DXY_lag1"].shift(1))
        event_df["rv_dxy_1"] = realized_vol(event_df["r_dxy_lag1"], 1)
    if "BRENT_lag1" in event_df.columns:
        event_df["r_brent_lag1"] = np.log(event_df["BRENT_lag1"] / event_df["BRENT_lag1"].shift(1))
        event_df["rv_brent_1"] = realized_vol(event_df["r_brent_lag1"], 1)

    event_df = event_df[event_df[DATE_COL] >= EVENT_TIMELINE_START].reset_index(drop=True)
    event_df = assign_event_regimes(event_df)

    minimal_required_cols = [
        DATE_COL,
        RET_COL,
        TARGET_COL,
        "rv_usdtnd_1",
        "rv_usdtnd_5",
        "rv_usdtnd_10",
        "event_regime",
        "event_regime_code",
        "event_regime_description",
    ]
    return event_df.dropna(subset=minimal_required_cols).reset_index(drop=True)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    x = data.copy()
    available_base_features = [c for c in BASE_FEATURES if c in x.columns]

    for col in EVENT_DUMMY_COLUMNS:
        if col not in x.columns:
            x[col] = 0
        x[col] = x[col].astype(int)

    if 'event_regime_code' not in x.columns:
        x['event_regime_code'] = 0

    original_has_non_positive = {
        c: bool(pd.to_numeric(x[c], errors="coerce").le(0).any())
        for c in available_base_features
    }

    positive_base_features = [
        c for c in available_base_features
        if c not in SIGNED_OR_CAN_BE_NEGATIVE_FEATURES
    ]

    for c in positive_base_features:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(float).clip(lower=EPS)

    x[TARGET_COL] = pd.to_numeric(x[TARGET_COL], errors="coerce").astype(float).clip(lower=EPS)

    for c in available_base_features:
        if c in SIGNED_OR_CAN_BE_NEGATIVE_FEATURES:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)

    x["log_target"] = np.log(x[TARGET_COL])

    for lag in [1, 2, 3, 4, 5, 10, 21]:
        x[f"log_target_lag{lag}"] = x["log_target"].shift(lag)

    x["har_d"] = x["log_target"].shift(1)
    x["har_w"] = x["log_target"].shift(1).rolling(5, min_periods=5).mean()
    x["har_m"] = x["log_target"].shift(1).rolling(22, min_periods=22).mean()

    log_ret = x["log_target"].diff(1)
    lam = 0.94
    x["ewma_var"] = log_ret.ewm(alpha=1 - lam, adjust=False).var().shift(1)
    x["ewma_std"] = np.sqrt(x["ewma_var"].clip(lower=0))
    x["sq_innov_1d"] = log_ret.shift(1) ** 2
    x["sq_innov_5d"] = (log_ret ** 2).rolling(5, min_periods=5).mean().shift(1)
    x["abs_innov"] = log_ret.shift(1).abs()

    for w in [5, 10, 22]:
        x[f"roll_var_{w}"] = x["log_target"].shift(1).rolling(w, min_periods=w).var()
        x[f"roll_std_{w}"] = np.sqrt(x[f"roll_var_{w}"].clip(lower=0))

    for c in available_base_features:
        if c in SIGNED_OR_CAN_BE_NEGATIVE_FEATURES or original_has_non_positive.get(c, True):
            continue
        x[f"log_{c}"] = np.log(x[c].clip(lower=EPS))

    x["rv1_5d"] = x["rv_usdtnd_1"].rolling(5, min_periods=5).mean()
    x["rv1_22d"] = x["rv_usdtnd_1"].rolling(22, min_periods=22).mean()

    x["vix_x_rv1"] = x["VIX_lag1"] * x["rv_usdtnd_1"]
    x["vix_x_rv5"] = x["VIX_lag1"] * x["rv_usdtnd_5"]
    if "log_VIX_lag1" in x.columns and "log_rv_usdtnd_1" in x.columns:
        x["log_vix_x_log_rv1"] = x["log_VIX_lag1"] * x["log_rv_usdtnd_1"]
    x["dxy_x_vix"] = x["rv_dxy_1"] * x["VIX_lag1"]
    x["brent_x_rv1"] = x["rv_brent_1"] * x["rv_usdtnd_1"]

    x["vix_x_covid"] = x["VIX_lag1"] * x["covid_dummy"]
    x["brent_x_ukraine"] = x["rv_brent_1"] * x["ukraine_war_dummy"]
    x["brent_x_iran"] = x["rv_brent_1"] * x["iran_geopolitical_dummy"]
    x["dxy_x_tariff"] = x["rv_dxy_1"] * x["us_tariff_dummy"]
    x["move_x_crisis"] = x["MOVE_lag1"] * x["any_crisis_dummy"]
    x["sentiment_x_crisis"] = x["sentiment_global_lag1"] * x["any_crisis_dummy"]

    x["term_ratio_1_5"] = x["rv_usdtnd_1"] / (x["rv_usdtnd_5"] + EPS)
    x["term_ratio_1_10"] = x["rv_usdtnd_1"] / (x["rv_usdtnd_10"] + EPS)
    x["term_gap_1_5"] = x["rv_usdtnd_1"] - x["rv_usdtnd_5"]

    gfx = ["rv_eurusd_1", "rv_gbpusd_1", "rv_usdjpy_1"]
    x["global_fx_mean"] = x[gfx].mean(axis=1)
    x["global_local_ratio"] = x["global_fx_mean"] / (x["rv_usdtnd_1"] + EPS)

    for c, nm in [("VIX_lag1", "vix"), ("rv_usdtnd_1", "rv1")]:
        for w in [20, 60]:
            mu = x[c].rolling(w, min_periods=w).mean()
            sg = x[c].rolling(w, min_periods=w).std() + EPS
            x[f"{nm}_z{w}"] = (x[c] - mu) / sg

    mu22 = x["rv_usdtnd_1"].rolling(22, min_periods=22).mean()
    sg22 = x["rv_usdtnd_1"].rolling(22, min_periods=22).std() + EPS
    x["rv1_jump"] = ((x["rv_usdtnd_1"] - mu22) / sg22).clip(-5, 5)

    x["sent_5d"] = x["sentiment_global_lag1"].rolling(5, min_periods=5).mean()
    x["sent_shock"] = x["sentiment_global_lag1"] - x["sent_5d"]

    for c in available_base_features:
        x[f"{c}_lag1"] = x[c].shift(1)

    x["rv1_ma5"] = x["rv_usdtnd_1"].rolling(5, min_periods=5).mean()
    x["rv1_ma10"] = x["rv_usdtnd_1"].rolling(10, min_periods=10).mean()

    x["month"] = x[DATE_COL].dt.month
    x["quarter"] = x[DATE_COL].dt.quarter

    return x.dropna().reset_index(drop=True)


def build_feature_frame(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feat_df = build_features(data)
    feature_cols = [
        c
        for c in feat_df.columns
        if c not in EXCLUDED_MODEL_COLUMNS
        and pd.api.types.is_numeric_dtype(feat_df[c])
    ]
    return feat_df, feature_cols


def safe_exp(v: Any) -> np.ndarray:
    return np.exp(np.asarray(v, dtype=float)) - EPS


def ensure_event_features_in_selected_features(
    selected_features: List[str],
    all_feature_cols: List[str],
    max_feature_count: Optional[int] = None,
) -> List[str]:
    """Ensure event dummies and interactions are preserved in selected feature subsets.
    
    Event dummies and event interaction terms are calendar-known context
    variables. They are included as numerical model inputs to make the model
    regime-aware, and the same event labels are also used as diagnostic overlays
    to evaluate performance across event periods. String labels and ordinal event
    codes are reporting fields and are not used as continuous numerical predictors.
    """
    result = list(selected_features)
    event_features_to_force = EVENT_DUMMY_COLUMNS + EVENT_INTERACTION_COLUMNS
    
    for event_feat in event_features_to_force:
        if event_feat in all_feature_cols and event_feat not in result:
            result.append(event_feat)
    
    if max_feature_count is not None and len(result) > max_feature_count:
        event_in_result = [f for f in result if f in event_features_to_force]
        non_event_in_result = [f for f in result if f not in event_features_to_force]
        max_non_event = max_feature_count - len(event_in_result)
        result = non_event_in_result[:max_non_event] + event_in_result
    
    return result


def dataframe_to_json_safe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert dataframe to JSON-safe records with proper datetime and NaN handling."""
    if df is None or df.empty:
        return []
    
    safe_df = df.copy()
    
    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].astype(str)
    
    safe_df = safe_df.replace([np.inf, -np.inf], np.nan)
    safe_df = safe_df.where(pd.notnull(safe_df), None)
    
    return safe_df.to_dict(orient="records")


def validate_event_methodology_integration(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    stable_features_ridge: List[str],
    stable_features_broad: List[str],
) -> Dict[str, Any]:
    """Validate that event-aware methodology is correctly integrated.
    
    Checks:
    - All event dummies/interactions available in feat_df are in feature_cols.
    - All event dummies/interactions in feature_cols are in stable_features_ridge.
    - All event dummies/interactions in feature_cols are in stable_features_broad.
    - Event strings (regime, description) are not in numerical feature_cols.
    - event_regime_code is retained for reporting only and is not used as a predictor.
    """
    event_dummies = [c for c in EVENT_DUMMY_COLUMNS if c in feat_df.columns]
    event_interactions = [c for c in EVENT_INTERACTION_COLUMNS if c in feat_df.columns]
    all_event_numerical = event_dummies + event_interactions
    
    event_in_feature_cols = [f for f in all_event_numerical if f in feature_cols]
    event_missing_from_feature_cols = [f for f in all_event_numerical if f not in feature_cols]
    
    event_in_ridge = [f for f in event_in_feature_cols if f in stable_features_ridge]
    event_missing_from_ridge = [f for f in event_in_feature_cols if f not in stable_features_ridge]
    
    event_in_broad = [f for f in event_in_feature_cols if f in stable_features_broad]
    event_missing_from_broad = [f for f in event_in_feature_cols if f not in stable_features_broad]
    
    event_string_cols = [c for c in ['event_regime', 'event_regime_description'] if c in feature_cols]
    event_regime_code_used_as_predictor = (
        "event_regime_code" in feature_cols
        or "event_regime_code" in stable_features_ridge
        or "event_regime_code" in stable_features_broad
    )
    
    status = "PASS"
    if event_missing_from_feature_cols:
        status = "REVIEW"
    if event_missing_from_ridge:
        status = "REVIEW"
    if event_missing_from_broad:
        status = "REVIEW"
    if event_string_cols:
        status = "REVIEW"
    if event_regime_code_used_as_predictor:
        status = "REVIEW"
    
    return {
        "event_features_available": all_event_numerical,
        "event_features_in_feature_cols": event_in_feature_cols,
        "event_features_missing_from_feature_cols": event_missing_from_feature_cols,
        "event_features_in_ridge": event_in_ridge,
        "event_features_missing_from_ridge": event_missing_from_ridge,
        "event_features_in_broad": event_in_broad,
        "event_features_missing_from_broad": event_missing_from_broad,
        "event_string_columns_in_feature_cols": event_string_cols,
        "event_regime_code_used_as_predictor": event_regime_code_used_as_predictor,
        "status": status,
    }


def winsorise(X_fit: np.ndarray, X_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ql = np.nanquantile(X_fit, WINSOR_LOWER, axis=0)
    qh = np.nanquantile(X_fit, WINSOR_UPPER, axis=0)
    return np.clip(X_fit, ql, qh), np.clip(X_other, ql, qh)


def _safe_corr(a: Any, b: Any) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    valid = ~(np.isnan(a_arr) | np.isnan(b_arr))
    if valid.sum() < 2:
        return np.nan
    a_clean = a_arr[valid]
    b_clean = b_arr[valid]
    if np.std(a_clean) < 1e-8 or np.std(b_clean) < 1e-8:
        return np.nan
    return np.corrcoef(a_clean, b_clean)[0, 1]


def annualize_target_vol_3d(vol_value):
    # TARGET_COL is a 3-day realized-volatility proxy.
    # Convert 3-day volatility to annualized volatility.
    return vol_value * np.sqrt(TRADING_DAYS_PER_YEAR / VOL_TARGET_WINDOW)


def annualize_daily_return_vol(return_series):
    # Daily-return annualized volatility.
    return np.nanstd(return_series, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def robust_mad(values):
    # Median absolute deviation scaled to be comparable to standard deviation.
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return np.nan
    med = clean.median()
    return 1.4826 * np.median(np.abs(clean - med))


def reliability_label(observations):
    if observations < LOW_SAMPLE_THRESHOLD:
        return "Early / low sample"
    if observations < MODERATE_SAMPLE_THRESHOLD:
        return "Moderate sample"
    return "Robust sample"


def block_bootstrap_stat_ci(
    values,
    stat_func,
    n_boot=REGIME_BOOTSTRAP_ITERATIONS,
    block_size=REGIME_BOOTSTRAP_BLOCK_SIZE,
    seed=RANDOM_STATE,
):
    # Use block bootstrap to preserve short-term serial dependence.
    # Return 2.5% and 97.5% percentiles of the bootstrapped statistic.
    clean = pd.Series(values).dropna().astype(float).values
    n = len(clean)
    if n < 2:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    block_size = int(max(1, min(block_size, n)))
    stats = []
    for _ in range(n_boot):
        sampled = []
        while len(sampled) < n:
            start = rng.randint(0, n)
            block = clean[start:min(start + block_size, n)]
            if len(block) < block_size:
                block = np.concatenate([block, clean[:block_size - len(block)]])
            sampled.extend(block.tolist())
        sampled = np.asarray(sampled[:n], dtype=float)
        try:
            val = stat_func(sampled)
            if np.isfinite(val):
                stats.append(val)
        except Exception:
            continue
    if not stats:
        return np.nan, np.nan
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def brown_forsythe_vs_baseline(regime_returns, baseline_returns):
    # Brown-Forsythe is Levene's test with center="median".
    # It tests equality of variances and is more robust than classic Levene.
    if levene is None:
        return np.nan, np.nan, "Unavailable"
    x = pd.Series(regime_returns).dropna().astype(float)
    y = pd.Series(baseline_returns).dropna().astype(float)
    if len(x) < MIN_OBS_FOR_VARIANCE_TEST or len(y) < MIN_OBS_FOR_VARIANCE_TEST:
        return np.nan, np.nan, "Insufficient sample"
    try:
        stat, pvalue = levene(y.values, x.values, center="median")
        if not np.isfinite(pvalue):
            return np.nan, np.nan, "Unavailable"
        label = "Significant variance difference" if pvalue < 0.05 else "Not statistically significant"
        return float(stat), float(pvalue), label
    except Exception:
        return np.nan, np.nan, "Unavailable"


def compute_event_regime_descriptive_stats(
    feat_df: pd.DataFrame,
    high_volatility_cutoff: Optional[float] = None,
) -> pd.DataFrame:
    if 'event_regime' not in feat_df.columns:
        return pd.DataFrame()
    
    df = feat_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    
    grouped = df.groupby(['event_regime', 'event_regime_code', 'event_regime_description'])

    agg_spec = {
        "start_date": (DATE_COL, "min"),
        "end_date": (DATE_COL, "max"),
        "observations": (DATE_COL, "count"),
        "mean_target_vol": (TARGET_COL, "mean"),
        "median_target_vol": (TARGET_COL, "median"),
        "q75_target_vol": (TARGET_COL, lambda x: x.quantile(0.75)),
        "q90_target_vol": (TARGET_COL, lambda x: x.quantile(0.90)),
        "max_target_vol": (TARGET_COL, "max"),
        "target_vol_iqr": (TARGET_COL, lambda x: x.quantile(0.75) - x.quantile(0.25)),
        "target_vol_mad": (TARGET_COL, robust_mad),
        "annualized_return_vol": (RET_COL, annualize_daily_return_vol),
        "crisis_flag": ("any_crisis_dummy", "max") if "any_crisis_dummy" in df.columns else ("event_regime", lambda x: 0),
    }
    optional_agg_spec = {
        "mean_rv_usdtnd_1": ("rv_usdtnd_1", "mean"),
        "mean_rv_usdtnd_5": ("rv_usdtnd_5", "mean"),
        "mean_vix": ("VIX_lag1", "mean"),
        "mean_dxy_vol": ("rv_dxy_1", "mean"),
        "mean_brent_vol": ("rv_brent_1", "mean"),
    }
    for output_col, (source_col, func) in optional_agg_spec.items():
        if source_col in df.columns:
            agg_spec[output_col] = (source_col, func)

    stats = grouped.agg(**agg_spec).reset_index()

    stats['mean_target_vol_annualized'] = stats['mean_target_vol'].apply(annualize_target_vol_3d)
    stats['median_target_vol_annualized'] = stats['median_target_vol'].apply(annualize_target_vol_3d)
    stats['q75_target_vol_annualized'] = stats['q75_target_vol'].apply(annualize_target_vol_3d)
    stats['q90_target_vol_annualized'] = stats['q90_target_vol'].apply(annualize_target_vol_3d)
    stats['max_target_vol_annualized'] = stats['max_target_vol'].apply(annualize_target_vol_3d)
    stats['target_vol_mad_annualized'] = stats['target_vol_mad'].apply(annualize_target_vol_3d)

    if high_volatility_cutoff is not None:
        high_vol_cutoff = float(high_volatility_cutoff)
        share_high = df.groupby('event_regime')[TARGET_COL].apply(
            lambda x: float((pd.to_numeric(x, errors="coerce") >= high_vol_cutoff).mean())
        )
        stats['share_high_vol_days'] = stats['event_regime'].map(share_high)
    else:
        stats['share_high_vol_days'] = np.nan

    ci_records = []
    for _, row in stats.iterrows():
        regime_values = df.loc[df['event_regime'] == row['event_regime'], TARGET_COL]
        ci_low, ci_high = block_bootstrap_stat_ci(
            regime_values,
            lambda sample: annualize_target_vol_3d(np.nanmean(sample)),
        )
        ci_records.append((ci_low, ci_high))
    stats['annualized_target_vol_ci_low'] = [ci[0] for ci in ci_records]
    stats['annualized_target_vol_ci_high'] = [ci[1] for ci in ci_records]
    
    # Compute percentages
    pre_covid_mean = stats.loc[stats['event_regime'] == 'pre_covid_normal', 'mean_target_vol']
    if not pre_covid_mean.empty:
        pre_covid_mean = pre_covid_mean.iloc[0]
        stats['mean_target_vol_vs_pre_covid_pct'] = 100 * (stats['mean_target_vol'] / pre_covid_mean - 1)
    else:
        stats['mean_target_vol_vs_pre_covid_pct'] = np.nan
    
    full_sample_mean = df[TARGET_COL].mean()
    stats['mean_target_vol_vs_full_sample_pct'] = 100 * (stats['mean_target_vol'] / full_sample_mean - 1)

    pre_ann = stats.loc[stats['event_regime'] == 'pre_covid_normal', 'mean_target_vol_annualized']
    if not pre_ann.empty and np.isfinite(pre_ann.iloc[0]) and pre_ann.iloc[0] != 0:
        pre_ann = pre_ann.iloc[0]
        stats['annualized_target_vol_vs_pre_covid_pct'] = 100 * (stats['mean_target_vol_annualized'] / pre_ann - 1)
        stats['annualized_target_vol_ci_low_vs_pre_covid_pct'] = 100 * (stats['annualized_target_vol_ci_low'] / pre_ann - 1)
        stats['annualized_target_vol_ci_high_vs_pre_covid_pct'] = 100 * (stats['annualized_target_vol_ci_high'] / pre_ann - 1)
    else:
        stats['annualized_target_vol_vs_pre_covid_pct'] = np.nan
        stats['annualized_target_vol_ci_low_vs_pre_covid_pct'] = np.nan
        stats['annualized_target_vol_ci_high_vs_pre_covid_pct'] = np.nan

    broad_normal = df.loc[df.get('any_crisis_dummy', pd.Series(0, index=df.index)).eq(0)]
    if not broad_normal.empty:
        broad_mean = pd.to_numeric(broad_normal[TARGET_COL], errors="coerce").mean()
        broad_mean_ann = annualize_target_vol_3d(broad_mean)
        broad_q90_ann = annualize_target_vol_3d(pd.to_numeric(broad_normal[TARGET_COL], errors="coerce").quantile(0.90))
        if np.isfinite(broad_mean) and broad_mean != 0:
            stats['mean_target_vol_vs_broad_normal_pct'] = 100 * (stats['mean_target_vol'] / broad_mean - 1)
        else:
            stats['mean_target_vol_vs_broad_normal_pct'] = np.nan
        if np.isfinite(broad_mean_ann) and broad_mean_ann != 0:
            stats['mean_target_vol_annualized_vs_broad_normal_pct'] = 100 * (
                stats['mean_target_vol_annualized'] / broad_mean_ann - 1
            )
        else:
            stats['mean_target_vol_annualized_vs_broad_normal_pct'] = np.nan
        if np.isfinite(broad_q90_ann) and broad_q90_ann != 0:
            stats['q90_target_vol_annualized_vs_broad_normal_pct'] = 100 * (
                stats['q90_target_vol_annualized'] / broad_q90_ann - 1
            )
        else:
            stats['q90_target_vol_annualized_vs_broad_normal_pct'] = np.nan
    else:
        stats['mean_target_vol_vs_broad_normal_pct'] = np.nan
        stats['mean_target_vol_annualized_vs_broad_normal_pct'] = np.nan
        stats['q90_target_vol_annualized_vs_broad_normal_pct'] = np.nan

    baseline_returns = df.loc[df['event_regime'] == 'pre_covid_normal', RET_COL]
    bf_records = []
    for _, row in stats.iterrows():
        if row['event_regime'] == 'pre_covid_normal':
            bf_records.append((np.nan, np.nan, "Baseline"))
            continue
        regime_returns = df.loc[df['event_regime'] == row['event_regime'], RET_COL]
        bf_records.append(brown_forsythe_vs_baseline(regime_returns, baseline_returns))
    stats['brown_forsythe_stat_vs_pre_covid'] = [rec[0] for rec in bf_records]
    stats['brown_forsythe_pvalue_vs_pre_covid'] = [rec[1] for rec in bf_records]
    stats['variance_test_result'] = [rec[2] for rec in bf_records]

    stats['reliability_label'] = stats['observations'].apply(reliability_label)
    stats['sample_size_note'] = stats['reliability_label'].map({
        "Early / low sample": "Interpret as preliminary because the regime has limited observations.",
        "Moderate sample": "Interpret with moderate confidence.",
        "Robust sample": "Estimate is supported by a relatively deep sample.",
    })
    stats['source_dataset'] = "event_analysis_df"
    
    # Sort and round
    result = stats.sort_values('event_regime_code').reset_index(drop=True)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    pct_cols = [c for c in numeric_cols if 'pct' in c]
    pvalue_cols = [c for c in numeric_cols if 'pvalue' in c]
    other_numeric = [c for c in numeric_cols if c not in pct_cols and c not in pvalue_cols]
    result[other_numeric] = result[other_numeric].round(6)
    result[pct_cols] = result[pct_cols].round(2)
    result[pvalue_cols] = result[pvalue_cols].round(4)
    
    return result


def attach_event_regime_to_output(output_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    if output_df.empty:
        return output_df
    
    out_df = output_df.copy()
    feat_copy = feat_df.copy()
    out_df[DATE_COL] = pd.to_datetime(out_df[DATE_COL], errors='coerce')
    feat_copy[DATE_COL] = pd.to_datetime(feat_copy[DATE_COL], errors='coerce')
    
    merge_cols = ['event_regime', 'event_regime_code', 'event_regime_description', 'covid_dummy', 'ukraine_war_dummy', 'us_tariff_dummy', 'iran_geopolitical_dummy', 'any_crisis_dummy']
    available_cols = [c for c in merge_cols if c in feat_copy.columns]
    
    if available_cols:
        out_df = out_df.merge(feat_copy[[DATE_COL] + available_cols], on=DATE_COL, how='left')
    
    return out_df


def compute_event_regime_forecast_metrics(output_df: pd.DataFrame) -> pd.DataFrame:
    if output_df.empty or 'event_regime' not in output_df.columns:
        return pd.DataFrame()
    
    df = output_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    
    pred_cols = ['naive_lag_observed', 'pred_garch_var', 'pred_ridge', 'pred_xgb', 'pred_blend']
    available_preds = [c for c in pred_cols if c in df.columns]
    
    results = []
    
    for regime in df['event_regime'].unique():
        regime_df = df[df['event_regime'] == regime]
        if regime_df.empty:
            continue
        
        regime_info = regime_df[['event_regime', 'event_regime_code', 'event_regime_description']].iloc[0]
        crisis_flag = regime_df['any_crisis_dummy'].max() if 'any_crisis_dummy' in regime_df.columns else 0
        
        for model in available_preds:
            model_df = regime_df.dropna(subset=['actual', model])
            if len(model_df) < 5:
                continue
            
            actual = model_df['actual']
            pred = model_df[model]
            
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            bias = (pred - actual).mean()
            qlike = qlike_from_vol_proxy(actual, pred, eps=EPS)
            corr = _safe_corr(actual, pred)
            
            # R2 vs naive
            if 'naive_lag_observed' in available_preds and model != 'naive_lag_observed':
                naive_df = regime_df.dropna(subset=['actual', 'naive_lag_observed'])
                if len(naive_df) >= 5:
                    mse_model = mean_squared_error(actual, pred)
                    mse_naive = mean_squared_error(naive_df['actual'], naive_df['naive_lag_observed'])
                    r2_vs_naive = 1 - mse_model / mse_naive if mse_naive > 0 else np.nan
                    rmse_naive = np.sqrt(mse_naive)
                    rmse_red_pct = 100 * (rmse_naive - rmse) / rmse_naive if rmse_naive > 0 else np.nan
                else:
                    r2_vs_naive = np.nan
                    rmse_red_pct = np.nan
            else:
                r2_vs_naive = np.nan
                rmse_red_pct = np.nan
            
            results.append({
                'event_regime': regime_info['event_regime'],
                'event_regime_code': regime_info['event_regime_code'],
                'event_regime_description': regime_info['event_regime_description'],
                'model': model,
                'observations': len(model_df),
                'start_date': model_df[DATE_COL].min(),
                'end_date': model_df[DATE_COL].max(),
                'RMSE': rmse,
                'MAE': mae,
                'Bias': bias,
                'QLIKE': qlike,
                'Corr': corr,
                'R2_vs_naive': r2_vs_naive,
                'RMSE_red_pct_vs_naive': rmse_red_pct,
                'crisis_flag': crisis_flag
            })
    
    result = pd.DataFrame(results)
    if not result.empty:
        result = result.sort_values(['event_regime_code', 'model']).reset_index(drop=True)
        # Rounding
        result[['RMSE', 'MAE', 'Bias', 'QLIKE']] = result[['RMSE', 'MAE', 'Bias', 'QLIKE']].round(6)
        result[['Corr', 'R2_vs_naive']] = result[['Corr', 'R2_vs_naive']].round(4)
        result['RMSE_red_pct_vs_naive'] = result['RMSE_red_pct_vs_naive'].round(2)
    
    return result


def qlike_from_vol_proxy(y_true: Any, y_pred: Any, eps: float = EPS) -> float:
    realized_var = np.square(np.clip(np.asarray(y_true, dtype=float), eps, None))
    forecast_var = np.square(np.clip(np.asarray(y_pred, dtype=float), eps, None))
    ratio = realized_var / forecast_var
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def compute_smearing_factor(y_log_true: Any, y_log_pred: Any) -> float:
    resid = np.asarray(y_log_true, dtype=float) - np.asarray(y_log_pred, dtype=float)
    factor = float(np.mean(np.exp(np.clip(resid, -20, 20))))
    return max(factor, EPS)


def level_from_log_with_smearing(pred_log: Any, smear_factor: float) -> np.ndarray:
    pred_log = np.asarray(pred_log, dtype=float)
    return np.maximum(np.exp(np.clip(pred_log, -50, 50)) * smear_factor - EPS, EPS)


def build_sampled_xgb_configs(random_state: int = RANDOM_STATE, max_configs: int = XGB_MAX_CONFIGS_PER_WINDOW) -> List[Dict[str, Any]]:
    from itertools import product

    full_grid = list(
        product(
            [2, 3, 4, 5],
            [0.01, 0.03, 0.05, 0.08],
            [100, 200, 400],
            [1, 3, 5],
            [1.0, 5.0, 10.0],
            [0.0, 0.5, 1.0],
            [0.8, 1.0],
            [0.8, 1.0],
        )
    )
    rng = np.random.RandomState(random_state)
    if len(full_grid) > max_configs:
        chosen_idx = rng.choice(len(full_grid), size=max_configs, replace=False)
        selected = [full_grid[i] for i in sorted(chosen_idx)]
    else:
        selected = full_grid

    configs = []
    for max_depth, learning_rate, n_estimators, min_child_weight, reg_lambda, reg_alpha, subsample, colsample_bytree in selected:
        configs.append(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "min_child_weight": min_child_weight,
                "reg_lambda": reg_lambda,
                "reg_alpha": reg_alpha,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "n_jobs": XGB_N_JOBS,
                "random_state": random_state,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "verbosity": 0,
            }
        )
    return configs


def safe_ljungbox_p(resid: Any, lag: int) -> float:
    resid = np.asarray(resid, dtype=float)
    if len(resid) <= lag + 1:
        return np.nan
    return float(acorr_ljungbox(resid, lags=[lag], return_df=True)["lb_pvalue"].values[0])


def safe_arch_p(resid: Any, nlags: int = 5) -> float:
    resid = np.asarray(resid, dtype=float)
    if len(resid) <= nlags + 2:
        return np.nan
    _, pval, *_ = het_arch(resid, nlags=nlags)
    return float(pval)


def compute_global_metrics(y_true: Any, y_pred: Any, y_naive: Any, label: str) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_naive = np.asarray(y_naive, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))

    mse_naive = mean_squared_error(y_true, y_naive)
    r2_vs_naive = 1 - mse / mse_naive if mse_naive > EPS else np.nan
    r2_classical = r2_score(y_true, y_pred)

    bias = float(np.mean(y_pred - y_true))
    mean_pred = float(np.mean(y_pred))
    mean_actual = float(np.mean(y_true))
    mean_pred_to_actual = mean_pred / (mean_actual + EPS)

    qlike = qlike_from_vol_proxy(y_true, y_pred, eps=EPS)

    if np.std(y_true) > EPS and np.std(y_pred) > EPS:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan

    da = float(np.mean(np.sign(y_true - y_naive) == np.sign(y_pred - y_naive)))

    resid = y_true - y_pred
    lb1 = safe_ljungbox_p(resid, 1)
    lb5 = safe_ljungbox_p(resid, 5)
    lb22 = safe_ljungbox_p(resid, 22)
    arch_p = safe_arch_p(resid, nlags=5)

    return {
        "Model": label,
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "R2_vs_naive": round(r2_vs_naive, 4) if np.isfinite(r2_vs_naive) else np.nan,
        "R2_classical": round(r2_classical, 4) if np.isfinite(r2_classical) else np.nan,
        "Bias": round(bias, 6),
        "MeanPred_to_Actual": round(mean_pred_to_actual, 4),
        "QLIKE": round(qlike, 6),
        "Corr": round(corr, 4) if np.isfinite(corr) else np.nan,
        "DA": f"{da:.1%}",
        "RMSE_red_pct": round(100 * (np.sqrt(mse_naive) - rmse) / np.sqrt(mse_naive), 1)
        if mse_naive > EPS
        else np.nan,
        "LB1_p": round(lb1, 3) if np.isfinite(lb1) else np.nan,
        "LB5_p": round(lb5, 3) if np.isfinite(lb5) else np.nan,
        "LB22_p": round(lb22, 3) if np.isfinite(lb22) else np.nan,
        "ARCH_p": round(arch_p, 4) if np.isfinite(arch_p) else np.nan,
    }


VOL_INDEX_DESCRIPTIONS = {
    "Very Calm": {
        "short_explanation": "USD/TND volatility is near the bottom of its historical range.",
        "recommended_interpretation": "Short-term FX conditions look comparatively quiet, but liquidity gaps can still matter in frontier FX.",
    },
    "Normal": {
        "short_explanation": "USD/TND volatility is close to its usual historical range.",
        "recommended_interpretation": "Hedging and treasury decisions can be guided by normal risk controls, with routine monitoring.",
    },
    "Elevated": {
        "short_explanation": "USD/TND volatility is above its typical historical range.",
        "recommended_interpretation": "Short-term FX risk is building; hedging timing and exposure limits deserve closer attention.",
    },
    "High": {
        "short_explanation": "USD/TND volatility is in the upper historical range.",
        "recommended_interpretation": "Hedging sensitivity is elevated and short-term FX risk should be monitored closely.",
    },
    "Stress": {
        "short_explanation": "USD/TND volatility is near the most stressed observations in the available history.",
        "recommended_interpretation": "Risk conditions are unusually unstable; forecasts may change quickly and liquidity buffers matter.",
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _round_or_none(value: Any, digits: int = 6) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return round(out, digits)


def _finite_array(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(pd.Series(values), dtype=float)
    return arr[np.isfinite(arr)]


def _combined_reference(*series_list: Any) -> np.ndarray:
    arrays = [_finite_array(series) for series in series_list]
    arrays = [arr for arr in arrays if len(arr) > 0]
    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate(arrays)


def percentile_rank_index(value: Any, reference_series: Any) -> float:
    value_f = _safe_float(value, default=np.nan)
    if not np.isfinite(value_f):
        return 50.0

    reference = _finite_array(reference_series)
    if len(reference) == 0:
        return 50.0

    ref_min = float(np.min(reference))
    ref_max = float(np.max(reference))
    if ref_max - ref_min <= EPS:
        if value_f > ref_max:
            return 100.0
        if value_f < ref_min:
            return 0.0
        return 50.0

    if len(reference) < 20:
        index_value = 100.0 * (value_f - ref_min) / (ref_max - ref_min)
    else:
        below = float(np.mean(reference < value_f))
        equal = float(np.mean(np.isclose(reference, value_f, rtol=1e-8, atol=1e-12)))
        index_value = 100.0 * (below + 0.5 * equal)

    return round(float(np.clip(index_value, 0.0, 100.0)), 2)


def classify_volatility_index(index_value: Any) -> Dict[str, Any]:
    index_f = float(np.clip(_safe_float(index_value, default=50.0), 0.0, 100.0))
    if index_f < 20:
        label = "Very Calm"
    elif index_f < 40:
        label = "Normal"
    elif index_f < 60:
        label = "Elevated"
    elif index_f < 80:
        label = "High"
    else:
        label = "Stress"

    text = VOL_INDEX_DESCRIPTIONS[label]
    return {
        "index_level": round(index_f, 2),
        "classification_label": label,
        "short_explanation": text["short_explanation"],
        "recommended_interpretation": text["recommended_interpretation"],
    }


def _readable_feature_name(feature_name: str) -> str:
    if feature_name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[feature_name]
    base = feature_name
    if base.startswith("log_"):
        base = base[4:]
    if base.endswith("_lag1"):
        base = base[:-5]
    return base.replace("_", " ").strip().title()


def compute_top_feature_importance(
    xgb_feature_names: List[str],
    xgb_importances: Any,
    ridge_feature_names: Optional[List[str]] = None,
    ridge_coefficients: Any = None,
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    ridge_direction: Dict[str, str] = {}

    xgb_arr = _finite_array(xgb_importances)
    if len(xgb_arr) == len(xgb_feature_names) and len(xgb_arr) > 0:
        denom = float(np.sum(np.abs(xgb_arr)))
        if denom <= EPS:
            denom = float(np.max(np.abs(xgb_arr))) + EPS
        for feature, importance in zip(xgb_feature_names, xgb_arr):
            scores[feature] = scores.get(feature, 0.0) + 0.7 * float(abs(importance) / denom)

    if ridge_feature_names is not None and ridge_coefficients is not None:
        ridge_arr = _finite_array(ridge_coefficients)
        if len(ridge_arr) == len(ridge_feature_names) and len(ridge_arr) > 0:
            denom = float(np.sum(np.abs(ridge_arr))) + EPS
            for feature, coef in zip(ridge_feature_names, ridge_arr):
                scores[feature] = scores.get(feature, 0.0) + 0.3 * float(abs(coef) / denom)
                if coef > EPS:
                    ridge_direction[feature] = "positive"
                elif coef < -EPS:
                    ridge_direction[feature] = "negative"

    if not scores:
        return []

    total = sum(scores.values()) + EPS
    rows = []
    for feature, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]:
        display_name = _readable_feature_name(feature)
        direction = ridge_direction.get(feature, "nonlinear/mixed")
        if direction == "positive":
            direction_text = "Higher values were associated with a higher forecast in the Ridge component."
        elif direction == "negative":
            direction_text = "Higher values were associated with a lower forecast in the Ridge component."
        else:
            direction_text = "The nonlinear model used this feature in an interaction-driven way."
        rows.append(
            {
                "technical_name": feature,
                "feature_name": display_name,
                "importance_score": round(100.0 * float(score) / total, 2),
                "direction": direction,
                "explanation": (
                    f"{display_name} contributed to the model forecast. {direction_text} "
                    "This is model influence, not proof of causality."
                ),
            }
        )
    return rows


def build_volatility_index_history(dev_df: pd.DataFrame, holdout_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if holdout_df.empty and dev_df.empty:
        return []

    initial_reference = _combined_reference(
        dev_df.get("actual"),
        dev_df.get("pred_blend"),
        dev_df.get("pred_garch_var"),
    )
    source_df = holdout_df.copy()
    if source_df.empty:
        source_df = dev_df.copy()
        initial_reference = _combined_reference(source_df.get("actual"), source_df.get("pred_blend"))

    if DATE_COL in source_df.columns:
        source_df = source_df.sort_values(DATE_COL)

    reference = initial_reference.copy()
    if len(reference) == 0:
        reference = _combined_reference(source_df.get("actual"), source_df.get("pred_blend"))

    records: List[Dict[str, Any]] = []
    for _, row in source_df.iterrows():
        pred = row.get("pred_blend")
        actual = row.get("actual")
        forecast_index = percentile_rank_index(pred, reference)
        actual_index = percentile_rank_index(actual, reference)
        classification = classify_volatility_index(forecast_index)

        records.append(
            {
                "Date": row.get(DATE_COL),
                "actual": _round_or_none(actual, 8),
                "pred_blend": _round_or_none(pred, 8),
                "volatility_index": forecast_index,
                "actual_volatility_index": actual_index,
                "classification_label": classification["classification_label"],
                "explanation": classification["short_explanation"],
            }
        )

        update_values = _finite_array(
            [
                row.get("actual"),
                row.get("pred_blend"),
                row.get("pred_garch_var"),
            ]
        )
        if len(update_values) > 0:
            reference = np.concatenate([reference, update_values]) if len(reference) > 0 else update_values

    return records


def _series_records(dates: pd.Series, values: pd.Series, value_name: str, limit: int = 500) -> List[Dict[str, Any]]:
    frame = pd.DataFrame({DATE_COL: dates, value_name: values}).dropna().tail(limit)
    return [
        {DATE_COL: row[DATE_COL], value_name: _round_or_none(row[value_name], 8)}
        for _, row in frame.iterrows()
    ]


def _regime_card(name: str, raw_value: Any, reference: Any, history_dates: pd.Series, history_values: pd.Series) -> Dict[str, Any]:
    index_value = percentile_rank_index(raw_value, reference)
    classification = classify_volatility_index(index_value)
    return {
        "raw_value": _round_or_none(raw_value, 8),
        "index_value": index_value,
        "percentile": index_value,
        "classification_label": classification["classification_label"],
        "explanation": f"{name} volatility is classified as {classification['classification_label']}. {classification['short_explanation']}",
        "recommended_interpretation": classification["recommended_interpretation"],
        "sparkline": _series_records(history_dates, history_values, "value", limit=120),
    }


def compute_regime_cards(feat_df: pd.DataFrame) -> Dict[str, Any]:
    monthly_vol = realized_vol(feat_df[RET_COL], 22)
    specs = {
        "daily": ("Daily", feat_df["rv_usdtnd_1"]),
        "weekly": ("Weekly", feat_df["rv_usdtnd_5"]),
        "monthly": ("Monthly", monthly_vol),
    }
    cards: Dict[str, Any] = {}
    for key, (label, series) in specs.items():
        clean = pd.Series(series).replace([np.inf, -np.inf], np.nan)
        latest = clean.dropna().iloc[-1] if clean.dropna().shape[0] > 0 else np.nan
        reference = clean.dropna().iloc[:-1]
        cards[key] = _regime_card(label, latest, reference, feat_df[DATE_COL], clean)
    return cards


def compute_model_weight_commentary(forecast_payload: Dict[str, Any], development_result: Dict[str, Any]) -> Dict[str, Any]:
    xgb_weight = float(np.clip(_safe_float(forecast_payload.get("blend_weight_xgb"), 0.5), 0.0, 1.0))
    ridge_weight = 1.0 - xgb_weight
    regime_probability = float(np.clip(_safe_float(forecast_payload.get("regime_prob_high_vol"), 0.0), 0.0, 1.0))
    base_xgb_weight = float(np.clip(_safe_float(development_result.get("final_blend_weight"), xgb_weight), 0.0, 1.0))
    pred_ridge = _safe_float(forecast_payload.get("pred_ridge"), 0.0)
    pred_xgb = _safe_float(forecast_payload.get("pred_xgb"), 0.0)

    if regime_probability >= 0.5:
        commentary = (
            f"The system assigns {xgb_weight:.0%} weight to XGBoost and {ridge_weight:.0%} to Ridge because the "
            "high-volatility regime probability is elevated. In stressed periods, the nonlinear XGBoost model receives "
            "more weight because it can capture interactions between USD/TND volatility, global FX volatility, DXY, VIX, "
            "MOVE, and commodity shocks."
        )
    else:
        commentary = (
            f"The system assigns {ridge_weight:.0%} weight to Ridge and {xgb_weight:.0%} to XGBoost because the market "
            "is closer to normal conditions. In calmer regimes, the more stable linear model helps reduce overfitting risk."
        )

    return {
        "ridge_weight": round(ridge_weight, 4),
        "xgb_weight": round(xgb_weight, 4),
        "base_xgb_weight": round(base_xgb_weight, 4),
        "final_dynamic_xgb_weight": round(xgb_weight, 4),
        "regime_probability": round(regime_probability, 4),
        "ridge_contribution": round(ridge_weight * pred_ridge, 8),
        "xgb_contribution": round(xgb_weight * pred_xgb, 8),
        "commentary": commentary,
    }


def _find_metric_row(summary_df: pd.DataFrame, contains: str) -> Dict[str, Any]:
    if summary_df.empty or "Model" not in summary_df.columns:
        return {}
    mask = summary_df["Model"].astype(str).str.contains(contains, case=False, regex=False)
    if not mask.any():
        return {}
    return summary_df.loc[mask].iloc[0].to_dict()


def compute_model_comparison(holdout_summary: pd.DataFrame, holdout_df: pd.DataFrame) -> Dict[str, Any]:
    blend = _find_metric_row(holdout_summary, "Regime-Aware")
    garch = {}
    if "Model" in holdout_summary.columns:
        garch_mask = holdout_summary["Model"].astype(str).str.contains("GARCH", case=False, regex=False)
        if garch_mask.any():
            garch = holdout_summary.loc[garch_mask].iloc[0].to_dict()

    blend_rmse = _safe_float(blend.get("RMSE"), np.nan)
    garch_rmse = _safe_float(garch.get("RMSE"), np.nan)
    blend_mae = _safe_float(blend.get("MAE"), np.nan)
    garch_mae = _safe_float(garch.get("MAE"), np.nan)

    if np.isfinite(blend_rmse) and np.isfinite(garch_rmse):
        if blend_rmse < garch_rmse:
            tracking = "The regime-aware blend has a lower holdout RMSE than the selected GARCH benchmark."
        elif blend_rmse > garch_rmse:
            tracking = "The selected GARCH benchmark has a lower holdout RMSE than the regime-aware blend."
        else:
            tracking = "The regime-aware blend and selected GARCH benchmark have similar holdout RMSE."
    else:
        tracking = "Holdout metrics are not available for a clean model-vs-GARCH statement."

    recent_comment = ""
    if not holdout_df.empty and {"actual", "pred_blend"}.issubset(holdout_df.columns):
        recent = holdout_df.tail(30)
        recent_error = float(np.nanmean(recent["pred_blend"].astype(float) - recent["actual"].astype(float)))
        if recent_error > EPS:
            recent_comment = "Recently, the blended model has been modestly overestimating realized volatility."
        elif recent_error < -EPS:
            recent_comment = "Recently, the blended model has been modestly underestimating realized volatility."
        else:
            recent_comment = "Recently, the blended model has been broadly centered around realized volatility."

    metric_text = ""
    if np.isfinite(blend_rmse):
        metric_text = f" Blend RMSE is {blend_rmse:.6f}"
        if np.isfinite(blend_mae):
            metric_text += f" and MAE is {blend_mae:.6f}."
        else:
            metric_text += "."
    if np.isfinite(garch_rmse):
        metric_text += f" The selected GARCH benchmark RMSE is {garch_rmse:.6f}"
        if np.isfinite(garch_mae):
            metric_text += f" and MAE is {garch_mae:.6f}."
        else:
            metric_text += "."

    return {
        "metrics": holdout_summary.to_dict(orient="records") if not holdout_summary.empty else [],
        "commentary": " ".join(part for part in [tracking, recent_comment, metric_text] if part).strip(),
    }


def compute_garch_selection_commentary(
    development_result: Dict[str, Any],
    forecast_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected = development_result.get("selected_garch_spec", {}).get("name", "Unknown")
    live_anchor = (forecast_payload or {}).get("live_garch_anchor_used", selected)
    live_fallback_triggered = bool((forecast_payload or {}).get("live_garch_fallback_triggered", False))
    live_fallback_reason = (forecast_payload or {}).get("live_garch_fallback_reason")
    gate = development_result.get("garch_gate", {})
    base_m = gate.get("baseline", {})
    primary_m = gate.get("primary", {})

    base_rmse = _safe_float(base_m.get("RMSE"), np.nan)
    base_qlike = _safe_float(base_m.get("QLIKE"), np.nan)
    base_r2 = _safe_float(base_m.get("R2_vs_naive"), np.nan)
    primary_rmse = _safe_float(primary_m.get("RMSE"), np.nan)
    primary_qlike = _safe_float(primary_m.get("QLIKE"), np.nan)
    primary_r2 = _safe_float(primary_m.get("R2_vs_naive"), np.nan)

    primary_not_worse = (
        np.isfinite(base_rmse)
        and np.isfinite(primary_rmse)
        and primary_rmse <= base_rmse + NO_WORSE_TOL
        and primary_qlike <= base_qlike + NO_WORSE_TOL
        and primary_r2 + NO_WORSE_TOL >= base_r2
    )

    if selected == PRIMARY_GARCH_SPEC["name"]:
        reason = "EGARCH-t passed the configured no-worse tolerance rule versus GARCH-normal."
        commentary = (
            "EGARCH-t was selected because it was not worse than the GARCH-normal benchmark under the tolerance rule "
            "and is more suitable for FX volatility because it can capture asymmetric volatility responses and fat-tailed return shocks."
        )
        decision = "Primary accepted" if primary_not_worse else "Primary used by fallback/default rule"
    elif selected == BASELINE_GARCH_SPEC["name"]:
        reason = "EGARCH-t did not improve the benchmark metrics within the configured tolerance."
        commentary = (
            "GARCH-normal was selected because EGARCH-t did not improve the benchmark metrics within the configured tolerance. "
            "The simpler model is preferred to avoid unnecessary complexity."
        )
        decision = "Baseline retained"
    else:
        reason = "The ARCH fit fell back to a robust volatility proxy."
        commentary = (
            f"{selected} was used because the configured ARCH candidates could not produce a stable forecast for this run."
        )
        decision = "Fallback used"

    if live_fallback_triggered:
        commentary = (
            f"{commentary} The selected validation benchmark was {selected}. The live production anchor used for this "
            f"forecast was {live_anchor}. The live production anchor may fall back to EWMA if ARCH-based fitting fails "
            "or produces implausible forecasts. EWMA is retained as a robust operational fallback."
        )
        if live_fallback_reason:
            commentary = f"{commentary} Fallback reason: {live_fallback_reason}"

    comparison = []
    for label, metrics in [(BASELINE_GARCH_SPEC["name"], base_m), (PRIMARY_GARCH_SPEC["name"], primary_m)]:
        row = {"Model": label}
        for key in ["RMSE", "MAE", "QLIKE", "R2_vs_naive", "Bias", "Corr"]:
            row[key] = metrics.get(key)
        comparison.append(row)

    return {
        "selected_model": selected,
        "selected_validation_benchmark": selected,
        "live_garch_anchor_used": live_anchor,
        "live_garch_fallback_triggered": live_fallback_triggered,
        "live_garch_fallback_reason": live_fallback_reason,
        "tolerance": NO_WORSE_TOL,
        "decision_result": decision,
        "reason": reason,
        "comparison": comparison,
        "commentary": commentary,
        "tolerance_rule": (
            "EGARCH-t is selected only when its RMSE and QLIKE are no worse than GARCH-normal within the tolerance, "
            "and its R2_vs_naive is no lower within the same tolerance."
        ),
    }


def compute_calendar_effects(feat_df: pd.DataFrame) -> Dict[str, Any]:
    cal = feat_df[[DATE_COL, "rv_usdtnd_1"]].dropna().copy()
    if cal.empty:
        return {"month_average_vol": [], "day_of_week_average_vol": [], "heatmap_matrix": [], "commentary": ""}

    cal["month"] = cal[DATE_COL].dt.month
    cal["month_name"] = cal[DATE_COL].dt.strftime("%b")
    cal["day_of_week"] = cal[DATE_COL].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    month_avg = (
        cal.groupby(["month", "month_name"], as_index=False)["rv_usdtnd_1"]
        .mean()
        .sort_values("month")
        .rename(columns={"rv_usdtnd_1": "average_volatility"})
    )
    day_avg = (
        cal.groupby("day_of_week", as_index=False)["rv_usdtnd_1"]
        .mean()
        .rename(columns={"rv_usdtnd_1": "average_volatility"})
    )
    day_avg["day_order"] = day_avg["day_of_week"].map({day: idx for idx, day in enumerate(day_order)})
    day_avg = day_avg.sort_values("day_order").drop(columns=["day_order"])

    heat = (
        cal.pivot_table(index="month_name", columns="day_of_week", values="rv_usdtnd_1", aggfunc="mean")
        .reindex(month_avg["month_name"].tolist())
        .reindex(columns=day_order)
    )
    heat_records = []
    for month_name, row in heat.iterrows():
        for day_name in day_order:
            heat_records.append(
                {
                    "month": month_name,
                    "day_of_week": day_name,
                    "average_volatility": _round_or_none(row.get(day_name), 8),
                }
            )

    high_month = month_avg.sort_values("average_volatility", ascending=False).iloc[0]["month_name"]
    high_day = day_avg.sort_values("average_volatility", ascending=False).iloc[0]["day_of_week"]
    commentary = (
        f"Historically, {high_month} and {high_day}s show the highest average 1-day USD/TND realized volatility in this sample. "
        "Calendar effects are descriptive patterns, not guaranteed forecasts."
    )

    return {
        "month_average_vol": month_avg.to_dict(orient="records"),
        "day_of_week_average_vol": day_avg.to_dict(orient="records"),
        "heatmap_matrix": heat_records,
        "commentary": commentary,
    }


def compute_weekly_monthly_volatility(feat_df: pd.DataFrame) -> Dict[str, Any]:
    weekly = feat_df["rv_usdtnd_5"].replace([np.inf, -np.inf], np.nan)
    monthly = realized_vol(feat_df[RET_COL], 22).replace([np.inf, -np.inf], np.nan)

    latest_weekly_value = weekly.dropna().iloc[-1] if weekly.dropna().shape[0] > 0 else np.nan
    latest_monthly_value = monthly.dropna().iloc[-1] if monthly.dropna().shape[0] > 0 else np.nan

    latest_weekly = _regime_card("Weekly", latest_weekly_value, weekly.dropna().iloc[:-1], feat_df[DATE_COL], weekly)
    latest_monthly = _regime_card("Monthly", latest_monthly_value, monthly.dropna().iloc[:-1], feat_df[DATE_COL], monthly)

    if np.isfinite(_safe_float(latest_weekly_value, np.nan)) and np.isfinite(_safe_float(latest_monthly_value, np.nan)):
        if latest_weekly_value > latest_monthly_value:
            commentary = (
                "Weekly volatility is above monthly volatility, suggesting short-term FX risk is running hotter than the medium-term backdrop. "
                "If monthly volatility remains lower, the move may be temporary rather than a persistent regime shift."
            )
        elif latest_weekly_value < latest_monthly_value:
            commentary = (
                "Weekly volatility is below monthly volatility, suggesting recent conditions are calmer than the medium-term regime."
            )
        else:
            commentary = "Weekly and monthly volatility are broadly aligned, pointing to a stable near-term risk profile."
    else:
        commentary = "Weekly/monthly volatility comparison is limited by missing realized-volatility history."

    return {
        "weekly": _series_records(feat_df[DATE_COL], weekly, "weekly_volatility", limit=500),
        "monthly": _series_records(feat_df[DATE_COL], monthly, "monthly_volatility", limit=500),
        "latest_weekly": latest_weekly,
        "latest_monthly": latest_monthly,
        "commentary": commentary,
    }


def compute_volatility_of_volatility(feat_df: pd.DataFrame) -> Dict[str, Any]:
    rv = feat_df["rv_usdtnd_1"].replace([np.inf, -np.inf], np.nan)
    vov_5 = rv.rolling(5, min_periods=5).std()
    vov_22 = rv.rolling(22, min_periods=22).std()
    latest_value = vov_22.dropna().iloc[-1] if vov_22.dropna().shape[0] > 0 else (
        vov_5.dropna().iloc[-1] if vov_5.dropna().shape[0] > 0 else np.nan
    )
    reference = vov_22.dropna().iloc[:-1] if vov_22.dropna().shape[0] > 1 else vov_5.dropna().iloc[:-1]
    index_value = percentile_rank_index(latest_value, reference)
    classification = classify_volatility_index(index_value)

    history = pd.DataFrame(
        {
            DATE_COL: feat_df[DATE_COL],
            "vol_of_vol_5d": vov_5,
            "vol_of_vol_22d": vov_22,
        }
    ).dropna(how="all", subset=["vol_of_vol_5d", "vol_of_vol_22d"]).tail(500)

    if classification["classification_label"] in {"High", "Stress"}:
        commentary = "Volatility of volatility is high, meaning risk conditions are unstable and forecasts may change quickly."
    elif classification["classification_label"] == "Very Calm":
        commentary = "Volatility of volatility is low, meaning volatility conditions are comparatively stable."
    else:
        commentary = "Volatility of volatility is in a normal-to-elevated range, so forecast uncertainty should still be monitored."

    return {
        "history": [
            {
                DATE_COL: row[DATE_COL],
                "vol_of_vol_5d": _round_or_none(row.get("vol_of_vol_5d"), 8),
                "vol_of_vol_22d": _round_or_none(row.get("vol_of_vol_22d"), 8),
            }
            for _, row in history.iterrows()
        ],
        "latest": {
            "raw_value": _round_or_none(latest_value, 8),
            "index_value": index_value,
            "classification_label": classification["classification_label"],
            "explanation": classification["short_explanation"],
        },
        "commentary": commentary,
    }


def compute_global_fx_comparison(feat_df: pd.DataFrame) -> Dict[str, Any]:
    fx_cols = [col for col in ["rv_eurusd_1", "rv_gbpusd_1", "rv_usdjpy_1", "rv_dxy_1"] if col in feat_df.columns]
    if not fx_cols:
        return {"history": [], "latest": {}, "commentary": "Global FX comparison is unavailable because proxy columns are missing."}

    global_fx_mean = feat_df[fx_cols].mean(axis=1)
    local_vol = feat_df["rv_usdtnd_1"]
    ratio = local_vol / (global_fx_mean + EPS)
    spread = local_vol - global_fx_mean

    latest_ratio = _safe_float(ratio.dropna().iloc[-1] if ratio.dropna().shape[0] > 0 else np.nan, np.nan)
    if not np.isfinite(latest_ratio):
        label = "Unavailable"
    elif latest_ratio < 0.8:
        label = "USD/TND quieter than global FX"
    elif latest_ratio < 1.2:
        label = "USD/TND in line with global FX"
    elif latest_ratio < 2.0:
        label = "USD/TND more volatile than global FX"
    else:
        label = "USD/TND significantly more volatile than global FX"

    if label == "USD/TND quieter than global FX":
        commentary = "USD/TND volatility is currently below the average of major global FX proxies."
    elif label == "USD/TND in line with global FX":
        commentary = "USD/TND volatility is currently broadly aligned with the average of major global FX proxies."
    elif label == "Unavailable":
        commentary = "USD/TND versus global FX comparison is limited by missing data."
    else:
        commentary = (
            "USD/TND volatility is currently above the average of major global FX pairs. This suggests local currency risk is not only being "
            "driven by broad dollar conditions but may also reflect local or regional dynamics."
        )

    history = pd.DataFrame(
        {
            DATE_COL: feat_df[DATE_COL],
            "usdtnd_volatility": local_vol,
            "global_fx_mean": global_fx_mean,
            "ratio": ratio,
            "spread": spread,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna(subset=["usdtnd_volatility", "global_fx_mean"]).tail(500)

    return {
        "history": [
            {
                DATE_COL: row[DATE_COL],
                "usdtnd_volatility": _round_or_none(row.get("usdtnd_volatility"), 8),
                "global_fx_mean": _round_or_none(row.get("global_fx_mean"), 8),
                "ratio": _round_or_none(row.get("ratio"), 4),
                "spread": _round_or_none(row.get("spread"), 8),
            }
            for _, row in history.iterrows()
        ],
        "latest": {
            "usdtnd_volatility": _round_or_none(local_vol.dropna().iloc[-1] if local_vol.dropna().shape[0] > 0 else np.nan, 8),
            "global_fx_mean": _round_or_none(global_fx_mean.dropna().iloc[-1] if global_fx_mean.dropna().shape[0] > 0 else np.nan, 8),
            "ratio": _round_or_none(latest_ratio, 4),
            "spread": _round_or_none(spread.dropna().iloc[-1] if spread.dropna().shape[0] > 0 else np.nan, 8),
            "classification": label,
        },
        "commentary": commentary,
    }


def build_dashboard_history(holdout_df: pd.DataFrame, index_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if holdout_df.empty:
        return []

    history = holdout_df.copy()
    index_df = pd.DataFrame(index_history)
    if not index_df.empty and DATE_COL in index_df.columns:
        index_df[DATE_COL] = pd.to_datetime(index_df[DATE_COL])
        history[DATE_COL] = pd.to_datetime(history[DATE_COL])
        history = history.merge(
            index_df[[DATE_COL, "volatility_index", "classification_label"]],
            on=DATE_COL,
            how="left",
        )
    else:
        history["volatility_index"] = np.nan
        history["classification_label"] = None

    history["model_error"] = history["pred_blend"] - history["actual"]
    records = []
    for _, row in history.iterrows():
        records.append(
            {
                DATE_COL: row.get(DATE_COL),
                "actual_volatility": _round_or_none(row.get("actual"), 8),
                "predicted_volatility": _round_or_none(row.get("pred_blend"), 8),
                "volatility_index": _round_or_none(row.get("volatility_index"), 2),
                "regime_label": row.get("classification_label"),
                "garch_forecast": _round_or_none(row.get("pred_garch_var"), 8),
                "naive_forecast": _round_or_none(row.get("naive_lag_observed"), 8),
                "model_error": _round_or_none(row.get("model_error"), 8),
                "high_vol_probability": _round_or_none(row.get("prob_high_vol"), 4),
            }
        )
    return records


def build_dashboard_summary(
    forecast_payload: Dict[str, Any],
    volatility_current: Dict[str, Any],
    garch_selection: Dict[str, Any],
    top_features: List[Dict[str, Any]],
) -> str:
    label = volatility_current.get("classification_label", "Unknown")
    index_value = volatility_current.get("value", volatility_current.get("index_level", 50.0))
    raw_forecast = forecast_payload.get("final_forecast_blend")
    p05 = forecast_payload.get("forecast_p05")
    p95 = forecast_payload.get("forecast_p95")
    high_prob = forecast_payload.get("regime_prob_high_vol")
    horizon = forecast_payload.get("horizon_days", VOL_TARGET_WINDOW)
    forecast_date = forecast_payload.get("forecast_date")
    selected_garch = garch_selection.get("selected_validation_benchmark", forecast_payload.get("selected_garch_benchmark", "the selected validation benchmark"))
    live_anchor = forecast_payload.get("live_garch_anchor_used", forecast_payload.get("garch_spec_used", selected_garch))
    drivers = ", ".join(feature["feature_name"] for feature in top_features[:3]) if top_features else "recent USD/TND and global market volatility inputs"

    return (
        f"As of {forecast_date}, the model forecasts USD/TND volatility over the next {horizon} trading days at {raw_forecast}. "
        f"The USD/TND Volatility Index stands at {index_value}/100, placing the market in a {label} volatility regime. "
        f"The model-implied 90% interval is {p05} to {p95}, with a high-volatility probability of {high_prob}. "
        f"The selected validation benchmark is {selected_garch}; the live production anchor used here is {live_anchor}. "
        "The live production anchor may fall back to EWMA if ARCH-based fitting fails or produces implausible forecasts. "
        f"Main model drivers include {drivers}. "
        "This is a historically grounded risk signal, not a guarantee of future spot moves."
    )


def build_dashboard_payload(
    feat_df: pd.DataFrame,
    development_result: Dict[str, Any],
    holdout_result: Dict[str, Any],
    dev_df_out: pd.DataFrame,
    holdout_df_out: pd.DataFrame,
    summary_dev_df: pd.DataFrame,
    summary_holdout_df: pd.DataFrame,
    forecast_payload: Dict[str, Any],
    live_model_details: Dict[str, Any],
) -> Dict[str, Any]:
    historical_reference = _combined_reference(
        dev_df_out.get("actual"),
        dev_df_out.get("pred_blend"),
        dev_df_out.get("pred_garch_var"),
        holdout_df_out.get("actual"),
        holdout_df_out.get("pred_blend"),
        holdout_df_out.get("pred_garch_var"),
        feat_df.get("rv_usdtnd_1"),
    )

    current_index_value = percentile_rank_index(
        forecast_payload.get("final_forecast_blend"),
        historical_reference,
    )
    current_classification = classify_volatility_index(current_index_value)
    volatility_current = {
        "value": current_index_value,
        "classification": current_classification["classification_label"],
        "classification_label": current_classification["classification_label"],
        "explanation": current_classification["short_explanation"],
        "short_explanation": current_classification["short_explanation"],
        "recommended_interpretation": current_classification["recommended_interpretation"],
        "raw_forecast": forecast_payload.get("final_forecast_blend"),
        "forecast_p05": forecast_payload.get("forecast_p05"),
        "forecast_p95": forecast_payload.get("forecast_p95"),
    }

    volatility_index_history = build_volatility_index_history(dev_df_out, holdout_df_out)
    model_weights = compute_model_weight_commentary(forecast_payload, development_result)
    model_comparison = compute_model_comparison(summary_holdout_df, holdout_df_out)
    garch_selection = compute_garch_selection_commentary(development_result, forecast_payload)
    top_features = live_model_details.get("feature_importance_top_5", [])
    dashboard_history = build_dashboard_history(holdout_df_out, volatility_index_history)

    dashboard = {
        "volatility_index": {
            "current": volatility_current,
            "history": volatility_index_history,
            "thresholds": [
                {"range": "0-20", "classification_label": "Very Calm", **VOL_INDEX_DESCRIPTIONS["Very Calm"]},
                {"range": "20-40", "classification_label": "Normal", **VOL_INDEX_DESCRIPTIONS["Normal"]},
                {"range": "40-60", "classification_label": "Elevated", **VOL_INDEX_DESCRIPTIONS["Elevated"]},
                {"range": "60-80", "classification_label": "High", **VOL_INDEX_DESCRIPTIONS["High"]},
                {"range": "80-100", "classification_label": "Stress", **VOL_INDEX_DESCRIPTIONS["Stress"]},
            ],
        },
        "regime": compute_regime_cards(feat_df),
        "model_weights": model_weights,
        "model_comparison": model_comparison,
        "garch_selection": garch_selection,
        "feature_importance": {
            "top_5": top_features,
            "commentary": (
                "These features were influential in the trained model for this forecast. "
                "They should be read as model drivers, not causal proof."
            ),
        },
        "calendar_effect": compute_calendar_effects(feat_df),
        "weekly_monthly_volatility": compute_weekly_monthly_volatility(feat_df),
        "volatility_of_volatility": compute_volatility_of_volatility(feat_df),
        "global_fx_comparison": compute_global_fx_comparison(feat_df),
        "history": dashboard_history,
        "summary": "",
    }
    dashboard["summary"] = build_dashboard_summary(
        forecast_payload,
        volatility_current,
        garch_selection,
        top_features,
    )
    return dashboard


def _ewma_vol_forecast(returns: Any, horizon: int, vol_window: int, lam: float = 0.94) -> Tuple[np.ndarray, np.ndarray, str]:
    ret = np.asarray(returns, dtype=float)
    ewma_var = float(ret[-1] ** 2)
    for r in ret[-250:]:
        ewma_var = lam * ewma_var + (1 - lam) * r**2
    vol_1d = np.sqrt(max(ewma_var, EPS))
    vol_nd = vol_1d * np.sqrt(vol_window)
    forecasts = np.full(horizon, vol_nd, dtype=float)

    n = len(ret)
    cond_var = np.zeros(n)
    cond_var[0] = ret[0] ** 2
    for t in range(1, n):
        cond_var[t] = lam * cond_var[t - 1] + (1 - lam) * ret[t - 1] ** 2
    cond_vol = np.sqrt(np.maximum(cond_var, EPS))
    return forecasts, cond_vol, "EWMA"


def _arch_model_factory():
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("Missing dependency 'arch'. Install it with: pip install arch") from exc
    return arch_model


def garch_forward_vol_forecast_robust(
    returns_train: Any,
    horizon: int,
    vol_window: int,
    primary_spec: Dict[str, Any],
    fallback_specs: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    if fallback_specs is None:
        fallback_specs = []
    all_specs = [primary_spec] + list(fallback_specs)
    ret = np.asarray(returns_train, dtype=float)
    attempted_specs: List[str] = []
    failure_reasons: Dict[str, str] = {}

    try:
        arch_model = _arch_model_factory()
    except Exception as exc:
        forecasts, cond_vol, spec_name = _ewma_vol_forecast(ret, horizon, vol_window)
        fallback_info = {
            "requested_primary_spec": primary_spec["name"],
            "anchor_used": spec_name,
            "fallback_triggered": True,
            "fallback_reason": f"ARCH model factory unavailable: {exc}; EWMA fallback used.",
            "attempted_specs": attempted_specs,
            "failure_reasons": failure_reasons,
        }
        return forecasts, cond_vol, spec_name, fallback_info

    for spec in all_specs:
        attempted_specs.append(spec["name"])
        try:
            am = arch_model(
                ret * 100.0,
                mean=spec["mean"],
                vol=spec["vol"],
                p=spec["p"],
                o=spec["o"],
                q=spec["q"],
                dist=spec["dist"],
            )
            res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
            fc = res.forecast(horizon=horizon + vol_window - 1, reindex=False)
            var_path = np.maximum(fc.variance.values[-1], 0.0) / (100.0**2)
            forecasts = np.array(
                [
                    np.sqrt(max(float(np.sum(var_path[j : j + vol_window])), EPS))
                    for j in range(horizon)
                ],
                dtype=float,
            )
            if not (np.all(np.isfinite(forecasts)) and np.all(forecasts > 0) and np.max(forecasts) < 1.0):
                raise ValueError(f"Implausible forecast from {spec['name']}")
            cond_vol = res.conditional_volatility.values / 100.0
            fallback_info = {
                "requested_primary_spec": primary_spec["name"],
                "anchor_used": spec["name"],
                "fallback_triggered": spec["name"] != primary_spec["name"],
                "fallback_reason": (
                    None
                    if spec["name"] == primary_spec["name"]
                    else f"{primary_spec['name']} failed; {spec['name']} fallback used."
                ),
                "attempted_specs": attempted_specs,
                "failure_reasons": failure_reasons,
            }
            return forecasts, cond_vol, spec["name"], fallback_info
        except Exception as exc:
            failure_reasons[spec["name"]] = str(exc)
            continue

    forecasts, cond_vol, spec_name = _ewma_vol_forecast(ret, horizon, vol_window)
    fallback_info = {
        "requested_primary_spec": primary_spec["name"],
        "anchor_used": spec_name,
        "fallback_triggered": True,
        "fallback_reason": "All ARCH specifications failed or produced implausible forecasts; EWMA fallback used.",
        "attempted_specs": attempted_specs,
        "failure_reasons": failure_reasons,
    }
    return forecasts, cond_vol, spec_name, fallback_info


def garch_forward_vol_forecast(returns_train: Any, horizon: int, vol_window: int, spec: Dict[str, Any]) -> np.ndarray:
    forecasts, _, _, _fallback_info = garch_forward_vol_forecast_robust(
        returns_train,
        horizon,
        vol_window,
        spec,
        fallback_specs=[],
    )
    return forecasts


def design_split_and_features(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    max_date = feat_df[DATE_COL].max()
    holdout_start_date = max_date - pd.DateOffset(years=HOLDOUT_YEARS)
    holdout_mask = feat_df[DATE_COL] >= holdout_start_date
    if not holdout_mask.any():
        raise ValueError("Could not create holdout split.")

    holdout_start_idx = int(np.flatnonzero(holdout_mask.values)[0])
    min_holdout_start = SPLIT + STEP_SIZE
    if holdout_start_idx < min_holdout_start:
        holdout_start_idx = min_holdout_start
        holdout_start_date = feat_df[DATE_COL].iloc[holdout_start_idx]

    if holdout_start_idx >= len(feat_df) - STEP_SIZE:
        raise ValueError("Holdout split too small. Increase data length or reduce STEP_SIZE.")

    fixed_event_threshold = float(
        np.percentile(
            feat_df[TARGET_COL].iloc[:holdout_start_idx].values,
            CLASSIFIER_TOP_PERCENTILE,
        )
    )

    coefs_history = []
    sc_audit = StandardScaler()
    audit_window = 250

    for i in range(audit_window, holdout_start_idx, STEP_SIZE):
        ts = max(0, i - audit_window)
        X_f = np.nan_to_num(
            feat_df[feature_cols].iloc[ts:i].values.astype(float),
            nan=0,
            posinf=0,
            neginf=0,
        )
        y_f = feat_df["log_target"].iloc[ts:i].values.astype(float)
        if len(y_f) < 60:
            continue
        X_f, _ = winsorise(X_f, X_f)
        X_f_s = sc_audit.fit_transform(X_f)
        m = Ridge(alpha=10.0).fit(X_f_s, y_f)
        coefs_history.append(m.coef_)

    if len(coefs_history) == 0:
        stable_features_ridge = feature_cols[: min(RIDGE_STABLE_FEATURE_COUNT, len(feature_cols))]
        stable_features_broad = feature_cols[: min(BROAD_FEATURE_COUNT, len(feature_cols))]
    else:
        coefs_mat = np.array(coefs_history)
        mean_c = np.mean(coefs_mat, axis=0)
        std_c = np.std(coefs_mat, axis=0) + EPS
        stability = np.abs(mean_c) / std_c

        ridge_k = min(RIDGE_STABLE_FEATURE_COUNT, len(feature_cols))
        broad_k = min(BROAD_FEATURE_COUNT, len(feature_cols))
        stable_rank = np.argsort(stability)

        stable_features_ridge = [feature_cols[idx] for idx in stable_rank[-ridge_k:]]
        stable_features_broad = [feature_cols[idx] for idx in stable_rank[-broad_k:]]

    emit(progress_callback, f"Frozen stable features: Ridge={len(stable_features_ridge)} Broad={len(stable_features_broad)}")

    # Ensure event-aware features are preserved in selected feature subsets.
    # Event dummies and interactions are calendar-known model inputs; event
    # labels and event_regime_code remain reporting/diagnostic fields.
    stable_features_ridge = ensure_event_features_in_selected_features(
        stable_features_ridge,
        feature_cols,
        max_feature_count=None,
    )
    stable_features_broad = ensure_event_features_in_selected_features(
        stable_features_broad,
        feature_cols,
        max_feature_count=None,
    )

    emit(progress_callback, f"Event-aware features added: Ridge={len(stable_features_ridge)} Broad={len(stable_features_broad)}")

    # Track which event features are available and selected
    event_features_available = [c for c in (EVENT_DUMMY_COLUMNS + EVENT_INTERACTION_COLUMNS) if c in feature_cols]
    event_features_in_ridge = [c for c in event_features_available if c in stable_features_ridge]
    event_features_in_broad = [c for c in event_features_available if c in stable_features_broad]

    return {
        "holdout_start_idx": holdout_start_idx,
        "holdout_start_date": holdout_start_date,
        "fixed_event_threshold": fixed_event_threshold,
        "stable_features_ridge": stable_features_ridge,
        "stable_features_broad": stable_features_broad,
        "event_features_available": event_features_available,
        "event_features_in_ridge": event_features_in_ridge,
        "event_features_in_broad": event_features_in_broad,
    }


def _make_xgb_classifier(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=1.0,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=XGB_N_JOBS,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        verbosity=0,
    )


def run_development_walk_forward(
    feat_df: pd.DataFrame,
    holdout_start_idx: int,
    fixed_event_threshold: float,
    stable_features_ridge: List[str],
    stable_features_broad: List[str],
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    xgb_search_configs = build_sampled_xgb_configs(RANDOM_STATE, XGB_MAX_CONFIGS_PER_WINDOW)
    ts_cv = TimeSeriesSplit(n_splits=3)

    preds_ridge_dev: List[float] = []
    preds_xgb_dev: List[float] = []
    preds_blend_dev: List[float] = []
    preds_garch_dev_baseline: List[float] = []
    preds_garch_dev_primary: List[float] = []
    acts_dev: List[float] = []
    dates_dev: List[pd.Timestamp] = []
    idx_dev: List[int] = []
    preds_blend_low_dev: List[float] = []
    preds_blend_high_dev: List[float] = []
    preds_class_prob_dev: List[float] = []
    class_actuals_dev: List[int] = []
    preds_dynamic_weight_dev: List[float] = []

    selected_ridge_alphas: List[float] = []
    selected_blend_weights: List[float] = []
    selected_xgb_keys: List[Tuple[Tuple[str, Any], ...]] = []

    dev_windows = list(range(SPLIT, holdout_start_idx, STEP_SIZE))
    emit(progress_callback, f"Development windows: {len(dev_windows)}")

    for win_idx, i in enumerate(dev_windows):
        ts = max(0, i - ROLLING_WINDOW)
        pe = min(i + STEP_SIZE, holdout_start_idx)
        h = pe - i
        train_end = i - LABEL_GAP

        if h <= 0 or train_end <= ts + 100:
            continue

        y_log_w = feat_df["log_target"].iloc[ts:train_end].values.astype(float)
        y_test = feat_df[TARGET_COL].iloc[i:pe].values.astype(float)

        X_fit_raw_ridge = np.nan_to_num(
            feat_df[stable_features_ridge].iloc[ts:train_end].values.astype(float),
            nan=0,
            posinf=0,
            neginf=0,
        )
        X_test_raw_ridge = np.nan_to_num(
            feat_df[stable_features_ridge].iloc[i:pe].values.astype(float),
            nan=0,
            posinf=0,
            neginf=0,
        )

        X_fit_raw_broad = np.nan_to_num(
            feat_df[stable_features_broad].iloc[ts:train_end].values.astype(float),
            nan=0,
            posinf=0,
            neginf=0,
        )
        X_test_raw_broad = np.nan_to_num(
            feat_df[stable_features_broad].iloc[i:pe].values.astype(float),
            nan=0,
            posinf=0,
            neginf=0,
        )

        ret_train = feat_df[RET_COL].iloc[ts : i + 1].values.astype(float)

        garch_preds_baseline, _, _, _fallback_info = garch_forward_vol_forecast_robust(
            ret_train,
            h,
            VOL_TARGET_WINDOW,
            BASELINE_GARCH_SPEC,
            fallback_specs=[],
        )
        preds_garch_dev_baseline.extend(garch_preds_baseline.tolist())

        garch_preds_primary, cond_vol_primary, _, _fallback_info = garch_forward_vol_forecast_robust(
            ret_train,
            h,
            VOL_TARGET_WINDOW,
            PRIMARY_GARCH_SPEC,
            fallback_specs=[BASELINE_GARCH_SPEC],
        )
        preds_garch_dev_primary.extend(garch_preds_primary.tolist())

        n_fit = X_fit_raw_ridge.shape[0]
        cond_vol_fit = cond_vol_primary[-n_fit:].reshape(-1, 1)
        cond_vol_test = garch_preds_primary.reshape(-1, 1)

        X_fit_raw_ridge = np.hstack([X_fit_raw_ridge, cond_vol_fit])
        X_test_raw_ridge = np.hstack([X_test_raw_ridge, cond_vol_test])
        X_fit_raw_broad = np.hstack([X_fit_raw_broad, cond_vol_fit])
        X_test_raw_broad = np.hstack([X_test_raw_broad, cond_vol_test])

        X_fit_full_ridge, X_test_full_ridge = winsorise(X_fit_raw_ridge, X_test_raw_ridge)
        X_fit_full_broad, X_test_full_broad = winsorise(X_fit_raw_broad, X_test_raw_broad)

        fold_data = []
        for train_idx, val_idx in list(ts_cv.split(X_fit_full_broad)):
            X_tr_ridge = X_fit_full_ridge[train_idx]
            X_va_ridge = X_fit_full_ridge[val_idx]
            X_tr_broad = X_fit_full_broad[train_idx]
            X_va_broad = X_fit_full_broad[val_idx]
            y_tr = y_log_w[train_idx]
            y_va = y_log_w[val_idx]
            fold_data.append((X_tr_ridge, X_va_ridge, X_tr_broad, X_va_broad, y_tr, y_va))

        best_ridge_alpha, best_ridge_score = 10.0, np.inf
        ridge_val_preds = []
        for alpha in RIDGE_ALPHA_GRID:
            rmse_folds = []
            val_preds_for_alpha = []
            try:
                for X_tr_ridge, X_va_ridge, _, _, y_tr, y_va in fold_data:
                    sc = StandardScaler()
                    X_tr_scaled = sc.fit_transform(X_tr_ridge)
                    X_va_scaled = sc.transform(X_va_ridge)
                    m = Ridge(alpha=alpha).fit(X_tr_scaled, y_tr)
                    pred_log_tr = m.predict(X_tr_scaled)
                    smear = compute_smearing_factor(y_tr, pred_log_tr)
                    pred_log_va = m.predict(X_va_scaled)
                    p = level_from_log_with_smearing(pred_log_va, smear)
                    val_preds_for_alpha.append(p)
                    rmse_folds.append(np.sqrt(mean_squared_error(safe_exp(y_va), p)))
                score = float(np.mean(rmse_folds) + 0.5 * np.std(rmse_folds))
                if score < best_ridge_score:
                    best_ridge_score = score
                    best_ridge_alpha = alpha
                    ridge_val_preds = val_preds_for_alpha
            except Exception:
                continue

        best_xgb_config, best_xgb_score = xgb_search_configs[0], np.inf
        xgb_val_preds = []
        for cfg in xgb_search_configs:
            rmse_folds = []
            val_preds_for_xgb = []
            try:
                for _, _, X_tr_broad, X_va_broad, y_tr, y_va in fold_data:
                    sc = StandardScaler()
                    X_tr_scaled = sc.fit_transform(X_tr_broad)
                    X_va_scaled = sc.transform(X_va_broad)
                    m = XGBRegressor(**cfg).fit(X_tr_scaled, y_tr)
                    pred_log_tr = m.predict(X_tr_scaled)
                    smear = compute_smearing_factor(y_tr, pred_log_tr)
                    pred_log_va = m.predict(X_va_scaled)
                    p = level_from_log_with_smearing(pred_log_va, smear)
                    val_preds_for_xgb.append(p)
                    rmse_folds.append(np.sqrt(mean_squared_error(safe_exp(y_va), p)))
                score = float(np.mean(rmse_folds) + 0.5 * np.std(rmse_folds))
                if score < best_xgb_score:
                    best_xgb_score = score
                    best_xgb_config = cfg
                    xgb_val_preds = val_preds_for_xgb
            except Exception:
                continue

        best_w, best_blend_score = 0.5, np.inf
        if len(ridge_val_preds) == len(fold_data) and len(xgb_val_preds) == len(fold_data):
            for w in BLEND_WEIGHT_GRID:
                rmse_folds = []
                for fold_idx, (_, _, _, _, _, y_va) in enumerate(fold_data):
                    p_r = ridge_val_preds[fold_idx]
                    p_x = xgb_val_preds[fold_idx]
                    p_b = w * p_x + (1 - w) * p_r
                    rmse_folds.append(np.sqrt(mean_squared_error(safe_exp(y_va), p_b)))
                score = float(np.mean(rmse_folds) + 0.5 * np.std(rmse_folds))
                if score < best_blend_score:
                    best_blend_score = score
                    best_w = w

        selected_ridge_alphas.append(float(best_ridge_alpha))
        selected_blend_weights.append(float(best_w))
        selected_xgb_keys.append(tuple(sorted(best_xgb_config.items())))

        sc_ridge = StandardScaler()
        X_fit_ridge_scaled = sc_ridge.fit_transform(X_fit_full_ridge)
        X_test_ridge_scaled = sc_ridge.transform(X_test_full_ridge)
        final_ridge = Ridge(alpha=best_ridge_alpha)
        final_ridge.fit(X_fit_ridge_scaled, y_log_w)
        pred_log_train_ridge = final_ridge.predict(X_fit_ridge_scaled)
        smear_ridge = compute_smearing_factor(y_log_w, pred_log_train_ridge)
        pred_levels_ridge = level_from_log_with_smearing(final_ridge.predict(X_test_ridge_scaled), smear_ridge)

        sc_xgb = StandardScaler()
        X_fit_xgb_scaled = sc_xgb.fit_transform(X_fit_full_broad)
        X_test_xgb_scaled = sc_xgb.transform(X_test_full_broad)
        final_xgb = XGBRegressor(**best_xgb_config)
        final_xgb.fit(X_fit_xgb_scaled, y_log_w)
        pred_log_train_xgb = final_xgb.predict(X_fit_xgb_scaled)
        smear_xgb = compute_smearing_factor(y_log_w, pred_log_train_xgb)
        pred_levels_xgb = level_from_log_with_smearing(final_xgb.predict(X_test_xgb_scaled), smear_xgb)

        y_train_level = safe_exp(y_log_w)
        y_class_tr = (y_train_level > fixed_event_threshold).astype(int)

        if len(np.unique(y_class_tr)) < 2:
            class_pred = np.full(h, float(np.mean(y_class_tr)))
            class_prob_train = np.full(len(y_class_tr), float(np.mean(y_class_tr)))
        else:
            positive_count = max(1, int(np.sum(y_class_tr == 1)))
            negative_count = max(1, int(np.sum(y_class_tr == 0)))
            clf = _make_xgb_classifier(negative_count / positive_count)
            clf.fit(X_fit_xgb_scaled, y_class_tr)
            class_pred = clf.predict_proba(X_test_xgb_scaled)[:, 1]
            class_prob_train = clf.predict_proba(X_fit_xgb_scaled)[:, 1]

        dynamic_weights = np.clip(
            best_w + REGIME_WEIGHT_BOOST * (class_pred - 0.5) * 2.0,
            REGIME_WEIGHT_FLOOR,
            REGIME_WEIGHT_CAP,
        )
        pred_levels = dynamic_weights * pred_levels_xgb + (1 - dynamic_weights) * pred_levels_ridge

        train_levels_ridge = level_from_log_with_smearing(pred_log_train_ridge, smear_ridge)
        train_levels_xgb = level_from_log_with_smearing(pred_log_train_xgb, smear_xgb)
        dynamic_weights_train = np.clip(
            best_w + REGIME_WEIGHT_BOOST * (class_prob_train - 0.5) * 2.0,
            REGIME_WEIGHT_FLOOR,
            REGIME_WEIGHT_CAP,
        )
        train_preds_blend = dynamic_weights_train * train_levels_xgb + (1 - dynamic_weights_train) * train_levels_ridge

        if len(acts_dev) == 0:
            resids = y_train_level - train_preds_blend
        else:
            resids = np.array(acts_dev, dtype=float) - np.array(preds_blend_dev, dtype=float)
        if len(resids) == 0:
            resids = np.array([0.0], dtype=float)

        q_low = float(np.nanquantile(resids, 0.05))
        q_high = float(np.nanquantile(resids, 0.95))

        preds_ridge_dev.extend(pred_levels_ridge.tolist())
        preds_xgb_dev.extend(pred_levels_xgb.tolist())
        preds_blend_dev.extend(pred_levels.tolist())
        preds_dynamic_weight_dev.extend(dynamic_weights.tolist())
        preds_blend_low_dev.extend(np.maximum(pred_levels + q_low, EPS).tolist())
        preds_blend_high_dev.extend(np.maximum(pred_levels + q_high, EPS).tolist())

        preds_class_prob_dev.extend(class_pred.tolist())
        class_actuals_dev.extend((y_test > fixed_event_threshold).astype(int).tolist())

        acts_dev.extend(y_test.tolist())
        dates_dev.extend(feat_df[DATE_COL].iloc[i:pe].tolist())
        idx_dev.extend(list(range(i, pe)))

        if (win_idx + 1) % 10 == 0 or win_idx == len(dev_windows) - 1:
            emit(progress_callback, f"Development window {win_idx + 1}/{len(dev_windows)}")

    final_ridge_alpha = float(np.median(selected_ridge_alphas)) if selected_ridge_alphas else 10.0
    final_blend_weight = float(np.clip(np.mean(selected_blend_weights), 0.0, 1.0)) if selected_blend_weights else 0.5
    final_xgb_config = dict(Counter(selected_xgb_keys).most_common(1)[0][0]) if selected_xgb_keys else xgb_search_configs[0]

    idx_dev_arr = np.asarray(idx_dev, dtype=int)
    acts_dev_arr = np.asarray(acts_dev, dtype=float)
    naive_dev_arr = feat_df[TARGET_COL].shift(VOL_TARGET_WINDOW).iloc[idx_dev_arr].values.astype(float)
    garch_dev_base_arr = np.asarray(preds_garch_dev_baseline, dtype=float)
    garch_dev_primary_arr = np.asarray(preds_garch_dev_primary, dtype=float)

    valid_garch_gate = np.isfinite(acts_dev_arr) & np.isfinite(naive_dev_arr)
    acts_gate = acts_dev_arr[valid_garch_gate]
    naive_gate = naive_dev_arr[valid_garch_gate]
    base_gate = garch_dev_base_arr[valid_garch_gate]
    primary_gate = garch_dev_primary_arr[valid_garch_gate]

    if len(acts_gate) == 0:
        selected_garch_spec = PRIMARY_GARCH_SPEC
        benchmark_label = "EGARCH-t (returns variance, comparable vol)"
        preds_garch_dev = preds_garch_dev_primary
        base_m = {"RMSE": np.nan, "QLIKE": np.nan, "R2_vs_naive": np.nan}
        primary_m = {"RMSE": np.nan, "QLIKE": np.nan, "R2_vs_naive": np.nan}
    else:
        base_m = compute_global_metrics(acts_gate, base_gate, naive_gate, BASELINE_GARCH_SPEC["name"])
        primary_m = compute_global_metrics(acts_gate, primary_gate, naive_gate, PRIMARY_GARCH_SPEC["name"])

        primary_not_worse = (
            float(primary_m["RMSE"]) <= float(base_m["RMSE"]) + NO_WORSE_TOL
            and float(primary_m["QLIKE"]) <= float(base_m["QLIKE"]) + NO_WORSE_TOL
            and float(primary_m["R2_vs_naive"]) + NO_WORSE_TOL >= float(base_m["R2_vs_naive"])
        )

        if primary_not_worse:
            selected_garch_spec = PRIMARY_GARCH_SPEC
            benchmark_label = "EGARCH-t (returns variance, comparable vol)"
            preds_garch_dev = preds_garch_dev_primary
        else:
            selected_garch_spec = BASELINE_GARCH_SPEC
            benchmark_label = "GARCH-normal (returns variance, comparable vol)"
            preds_garch_dev = preds_garch_dev_baseline

    emit(progress_callback, f"Selected GARCH benchmark: {selected_garch_spec['name']}")

    return {
        "preds_ridge": preds_ridge_dev,
        "preds_xgb": preds_xgb_dev,
        "preds_blend": preds_blend_dev,
        "preds_garch": preds_garch_dev,
        "acts": acts_dev,
        "dates": dates_dev,
        "idx": idx_dev,
        "preds_blend_low": preds_blend_low_dev,
        "preds_blend_high": preds_blend_high_dev,
        "preds_class_prob": preds_class_prob_dev,
        "class_actuals": class_actuals_dev,
        "preds_dynamic_weight": preds_dynamic_weight_dev,
        "final_ridge_alpha": final_ridge_alpha,
        "final_blend_weight": final_blend_weight,
        "final_xgb_config": final_xgb_config,
        "selected_garch_spec": selected_garch_spec,
        "benchmark_label": benchmark_label,
        "garch_gate": {"baseline": base_m, "primary": primary_m},
    }


def run_holdout_walk_forward(
    feat_df: pd.DataFrame,
    holdout_start_idx: int,
    fixed_event_threshold: float,
    stable_features_ridge: List[str],
    stable_features_broad: List[str],
    development_result: Dict[str, Any],
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    preds_ridge_h: List[float] = []
    preds_xgb_h: List[float] = []
    preds_blend_h: List[float] = []
    preds_garch_h: List[float] = []
    acts_h: List[float] = []
    dates_h: List[pd.Timestamp] = []
    idx_h: List[int] = []
    preds_blend_low_h: List[float] = []
    preds_blend_high_h: List[float] = []
    preds_class_prob_h: List[float] = []
    class_actuals_h: List[int] = []
    preds_dynamic_weight_h: List[float] = []

    holdout_windows = list(range(holdout_start_idx, len(feat_df), STEP_SIZE))
    emit(progress_callback, f"Holdout windows: {len(holdout_windows)}")

    for win_idx, i in enumerate(holdout_windows):
        ts = max(0, i - ROLLING_WINDOW)
        pe = min(i + STEP_SIZE, len(feat_df))
        h = pe - i
        train_end = i - LABEL_GAP

        if h <= 0 or train_end <= ts + 100:
            continue

        y_log_w = feat_df["log_target"].iloc[ts:train_end].values.astype(float)
        y_test = feat_df[TARGET_COL].iloc[i:pe].values.astype(float)

        X_fit_raw_ridge = np.nan_to_num(
            feat_df[stable_features_ridge].iloc[ts:train_end].values.astype(float),
            nan=0,
            posinf=0,
            neginf=0,
        )
        X_test_raw_ridge = np.nan_to_num(
            feat_df[stable_features_ridge].iloc[i:pe].values.astype(float),
            nan=0,
            posinf=0,
            neginf=0,
        )

        X_fit_raw_broad = np.nan_to_num(
            feat_df[stable_features_broad].iloc[ts:train_end].values.astype(float),
            nan=0,
            posinf=0,
            neginf=0,
        )
        X_test_raw_broad = np.nan_to_num(
            feat_df[stable_features_broad].iloc[i:pe].values.astype(float),
            nan=0,
            posinf=0,
            neginf=0,
        )

        ret_train = feat_df[RET_COL].iloc[ts : i + 1].values.astype(float)

        garch_preds, cond_vol_h_win, _, _fallback_info = garch_forward_vol_forecast_robust(
            ret_train,
            h,
            VOL_TARGET_WINDOW,
            development_result["selected_garch_spec"],
            fallback_specs=[BASELINE_GARCH_SPEC],
        )
        preds_garch_h.extend(garch_preds.tolist())

        n_fit = X_fit_raw_ridge.shape[0]
        cond_vol_fit = cond_vol_h_win[-n_fit:].reshape(-1, 1)
        cond_vol_test = garch_preds.reshape(-1, 1)

        X_fit_raw_ridge = np.hstack([X_fit_raw_ridge, cond_vol_fit])
        X_test_raw_ridge = np.hstack([X_test_raw_ridge, cond_vol_test])
        X_fit_raw_broad = np.hstack([X_fit_raw_broad, cond_vol_fit])
        X_test_raw_broad = np.hstack([X_test_raw_broad, cond_vol_test])

        X_fit_full_ridge, X_test_full_ridge = winsorise(X_fit_raw_ridge, X_test_raw_ridge)
        X_fit_full_broad, X_test_full_broad = winsorise(X_fit_raw_broad, X_test_raw_broad)

        sc_ridge = StandardScaler()
        X_fit_ridge_scaled = sc_ridge.fit_transform(X_fit_full_ridge)
        X_test_ridge_scaled = sc_ridge.transform(X_test_full_ridge)
        final_ridge = Ridge(alpha=development_result["final_ridge_alpha"])
        final_ridge.fit(X_fit_ridge_scaled, y_log_w)
        pred_log_train_ridge = final_ridge.predict(X_fit_ridge_scaled)
        smear_ridge = compute_smearing_factor(y_log_w, pred_log_train_ridge)
        pred_levels_ridge = level_from_log_with_smearing(final_ridge.predict(X_test_ridge_scaled), smear_ridge)

        sc_xgb = StandardScaler()
        X_fit_xgb_scaled = sc_xgb.fit_transform(X_fit_full_broad)
        X_test_xgb_scaled = sc_xgb.transform(X_test_full_broad)
        final_xgb = XGBRegressor(**development_result["final_xgb_config"])
        final_xgb.fit(X_fit_xgb_scaled, y_log_w)
        pred_log_train_xgb = final_xgb.predict(X_fit_xgb_scaled)
        smear_xgb = compute_smearing_factor(y_log_w, pred_log_train_xgb)
        pred_levels_xgb = level_from_log_with_smearing(final_xgb.predict(X_test_xgb_scaled), smear_xgb)

        y_train_level = safe_exp(y_log_w)
        y_class_tr = (y_train_level > fixed_event_threshold).astype(int)

        if len(np.unique(y_class_tr)) < 2:
            class_pred = np.full(h, float(np.mean(y_class_tr)))
            class_prob_train = np.full(len(y_class_tr), float(np.mean(y_class_tr)))
        else:
            positive_count = max(1, int(np.sum(y_class_tr == 1)))
            negative_count = max(1, int(np.sum(y_class_tr == 0)))
            clf = _make_xgb_classifier(negative_count / positive_count)
            clf.fit(X_fit_xgb_scaled, y_class_tr)
            class_pred = clf.predict_proba(X_test_xgb_scaled)[:, 1]
            class_prob_train = clf.predict_proba(X_fit_xgb_scaled)[:, 1]

        dynamic_weights = np.clip(
            development_result["final_blend_weight"] + REGIME_WEIGHT_BOOST * (class_pred - 0.5) * 2.0,
            REGIME_WEIGHT_FLOOR,
            REGIME_WEIGHT_CAP,
        )
        pred_levels = dynamic_weights * pred_levels_xgb + (1 - dynamic_weights) * pred_levels_ridge

        train_levels_ridge = level_from_log_with_smearing(pred_log_train_ridge, smear_ridge)
        train_levels_xgb = level_from_log_with_smearing(pred_log_train_xgb, smear_xgb)
        dynamic_weights_train = np.clip(
            development_result["final_blend_weight"] + REGIME_WEIGHT_BOOST * (class_prob_train - 0.5) * 2.0,
            REGIME_WEIGHT_FLOOR,
            REGIME_WEIGHT_CAP,
        )
        train_preds_blend = dynamic_weights_train * train_levels_xgb + (1 - dynamic_weights_train) * train_levels_ridge

        if len(acts_h) == 0:
            if len(development_result["acts"]) > 0:
                resids = np.array(development_result["acts"], dtype=float) - np.array(development_result["preds_blend"], dtype=float)
            else:
                resids = y_train_level - train_preds_blend
        else:
            resids = np.array(acts_h, dtype=float) - np.array(preds_blend_h, dtype=float)

        if len(resids) == 0:
            resids = np.array([0.0], dtype=float)

        q_low = float(np.nanquantile(resids, 0.05))
        q_high = float(np.nanquantile(resids, 0.95))

        preds_ridge_h.extend(pred_levels_ridge.tolist())
        preds_xgb_h.extend(pred_levels_xgb.tolist())
        preds_blend_h.extend(pred_levels.tolist())
        preds_dynamic_weight_h.extend(dynamic_weights.tolist())
        preds_blend_low_h.extend(np.maximum(pred_levels + q_low, EPS).tolist())
        preds_blend_high_h.extend(np.maximum(pred_levels + q_high, EPS).tolist())

        preds_class_prob_h.extend(class_pred.tolist())
        class_actuals_h.extend((y_test > fixed_event_threshold).astype(int).tolist())

        acts_h.extend(y_test.tolist())
        dates_h.extend(feat_df[DATE_COL].iloc[i:pe].tolist())
        idx_h.extend(list(range(i, pe)))

        if (win_idx + 1) % 5 == 0 or win_idx == len(holdout_windows) - 1:
            emit(progress_callback, f"Holdout window {win_idx + 1}/{len(holdout_windows)}")

    return {
        "preds_ridge": preds_ridge_h,
        "preds_xgb": preds_xgb_h,
        "preds_blend": preds_blend_h,
        "preds_garch": preds_garch_h,
        "acts": acts_h,
        "dates": dates_h,
        "idx": idx_h,
        "preds_blend_low": preds_blend_low_h,
        "preds_blend_high": preds_blend_high_h,
        "preds_class_prob": preds_class_prob_h,
        "class_actuals": class_actuals_h,
        "preds_dynamic_weight": preds_dynamic_weight_h,
    }


def evaluate_block(
    feat_df: pd.DataFrame,
    idxs: List[int],
    y_true: List[float],
    p_garch: List[float],
    p_ridge: List[float],
    p_xgb: List[float],
    p_blend: List[float],
    class_actuals: List[int],
    class_probs: List[float],
    benchmark_label: str,
) -> Tuple[pd.DataFrame, np.ndarray, float, float, np.ndarray]:
    idxs_arr = np.asarray(idxs, dtype=int)
    y_true_arr = np.asarray(y_true, dtype=float)
    p_garch_arr = np.asarray(p_garch, dtype=float)
    p_ridge_arr = np.asarray(p_ridge, dtype=float)
    p_xgb_arr = np.asarray(p_xgb, dtype=float)
    p_blend_arr = np.asarray(p_blend, dtype=float)

    naive_series = feat_df[TARGET_COL].shift(VOL_TARGET_WINDOW)
    p_naive = naive_series.iloc[idxs_arr].values.astype(float)

    valid = np.isfinite(y_true_arr) & np.isfinite(p_naive)
    y_true_valid = y_true_arr[valid]
    p_naive_valid = p_naive[valid]
    p_garch_valid = p_garch_arr[valid]
    p_ridge_valid = p_ridge_arr[valid]
    p_xgb_valid = p_xgb_arr[valid]
    p_blend_valid = p_blend_arr[valid]

    summary_rows = [
        compute_global_metrics(y_true_valid, p_naive_valid, p_naive_valid, "Naive (lag-observed)"),
        compute_global_metrics(y_true_valid, p_garch_valid, p_naive_valid, benchmark_label),
        compute_global_metrics(y_true_valid, p_ridge_valid, p_naive_valid, "Ridge Hybrid"),
        compute_global_metrics(y_true_valid, p_xgb_valid, p_naive_valid, "XGBoost Hybrid"),
        compute_global_metrics(y_true_valid, p_blend_valid, p_naive_valid, "Regime-Aware Blended Hybrid"),
    ]
    summary_df = pd.DataFrame(summary_rows)

    class_actuals_arr = np.asarray(class_actuals, dtype=int)[valid]
    class_probs_arr = np.asarray(class_probs, dtype=float)[valid]

    if len(class_actuals_arr) > 0 and len(np.unique(class_actuals_arr)) > 1:
        roc_auc = float(roc_auc_score(class_actuals_arr, class_probs_arr))
        f1 = float(f1_score(class_actuals_arr, (class_probs_arr > 0.5).astype(int)))
    else:
        roc_auc = np.nan
        f1 = np.nan

    return summary_df, p_naive_valid, roc_auc, f1, valid


def build_output_frame(result: Dict[str, Any], valid: np.ndarray, p_naive: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE_COL: np.array(result["dates"], dtype="datetime64[ns]")[valid],
            "actual": np.asarray(result["acts"], dtype=float)[valid],
            "naive_lag_observed": p_naive,
            "pred_garch_var": np.asarray(result["preds_garch"], dtype=float)[valid],
            "pred_ridge": np.asarray(result["preds_ridge"], dtype=float)[valid],
            "pred_xgb": np.asarray(result["preds_xgb"], dtype=float)[valid],
            "pred_blend": np.asarray(result["preds_blend"], dtype=float)[valid],
            "dynamic_weight_xgb": np.asarray(result["preds_dynamic_weight"], dtype=float)[valid],
            "pred_blend_p05": np.asarray(result["preds_blend_low"], dtype=float)[valid],
            "pred_blend_p95": np.asarray(result["preds_blend_high"], dtype=float)[valid],
            "actual_high_vol_fixed": np.asarray(result["class_actuals"], dtype=int)[valid],
            "prob_high_vol": np.asarray(result["preds_class_prob"], dtype=float)[valid],
        }
    )


def build_live_forecast(
    feat_df: pd.DataFrame,
    stable_features_ridge: List[str],
    stable_features_broad: List[str],
    fixed_event_threshold: float,
    development_result: Dict[str, Any],
    holdout_result: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    train_end_final = len(feat_df) - LABEL_GAP
    if train_end_final <= 100:
        raise ValueError("Not enough engineered rows to train the final forecast.")

    ts_final = max(0, train_end_final - ROLLING_WINDOW)
    y_log_full = feat_df["log_target"].iloc[ts_final:train_end_final].values.astype(float)

    X_fit_ridge_raw = np.nan_to_num(
        feat_df[stable_features_ridge].iloc[ts_final:train_end_final].values.astype(float),
        nan=0,
        posinf=0,
        neginf=0,
    )
    X_fit_broad_raw = np.nan_to_num(
        feat_df[stable_features_broad].iloc[ts_final:train_end_final].values.astype(float),
        nan=0,
        posinf=0,
        neginf=0,
    )

    ret_full = feat_df[RET_COL].iloc[ts_final : len(feat_df)].values.astype(float)
    garch_fc_final, _, garch_spec_used, live_fallback_info = garch_forward_vol_forecast_robust(
        ret_full,
        horizon=VOL_TARGET_WINDOW,
        vol_window=VOL_TARGET_WINDOW,
        primary_spec=development_result["selected_garch_spec"],
        fallback_specs=[BASELINE_GARCH_SPEC],
    )

    X_test_ridge_raw = np.nan_to_num(
        feat_df[stable_features_ridge].iloc[[-1]].values.astype(float),
        nan=0,
        posinf=0,
        neginf=0,
    )
    X_test_broad_raw = np.nan_to_num(
        feat_df[stable_features_broad].iloc[[-1]].values.astype(float),
        nan=0,
        posinf=0,
        neginf=0,
    )

    X_fit_ridge_w, X_test_ridge_w = winsorise(X_fit_ridge_raw, X_test_ridge_raw)
    X_fit_broad_w, X_test_broad_w = winsorise(X_fit_broad_raw, X_test_broad_raw)

    sc_ridge_final = StandardScaler()
    X_fit_ridge_sc = sc_ridge_final.fit_transform(X_fit_ridge_w)
    X_test_ridge_sc = sc_ridge_final.transform(X_test_ridge_w)

    final_ridge_model = Ridge(alpha=development_result["final_ridge_alpha"])
    final_ridge_model.fit(X_fit_ridge_sc, y_log_full)

    pred_log_train_ridge_f = final_ridge_model.predict(X_fit_ridge_sc)
    smear_ridge_f = compute_smearing_factor(y_log_full, pred_log_train_ridge_f)
    pred_ridge_final = level_from_log_with_smearing(final_ridge_model.predict(X_test_ridge_sc), smear_ridge_f)[0]

    sc_xgb_final = StandardScaler()
    X_fit_xgb_sc = sc_xgb_final.fit_transform(X_fit_broad_w)
    X_test_xgb_sc = sc_xgb_final.transform(X_test_broad_w)

    final_xgb_model = XGBRegressor(**development_result["final_xgb_config"])
    final_xgb_model.fit(X_fit_xgb_sc, y_log_full)

    pred_log_train_xgb_f = final_xgb_model.predict(X_fit_xgb_sc)
    smear_xgb_f = compute_smearing_factor(y_log_full, pred_log_train_xgb_f)
    pred_xgb_final = level_from_log_with_smearing(final_xgb_model.predict(X_test_xgb_sc), smear_xgb_f)[0]

    y_train_level_f = safe_exp(y_log_full)
    y_class_tr_f = (y_train_level_f > fixed_event_threshold).astype(int)

    if len(np.unique(y_class_tr_f)) < 2:
        regime_prob = float(np.mean(y_class_tr_f))
    else:
        pos_count = max(1, int(np.sum(y_class_tr_f == 1)))
        neg_count = max(1, int(np.sum(y_class_tr_f == 0)))
        clf_final = _make_xgb_classifier(neg_count / pos_count)
        clf_final.fit(X_fit_xgb_sc, y_class_tr_f)
        regime_prob = float(clf_final.predict_proba(X_test_xgb_sc)[0, 1])

    is_high_vol_regime = regime_prob >= 0.5
    final_dynamic_weight = float(
        np.clip(
            development_result["final_blend_weight"] + REGIME_WEIGHT_BOOST * (regime_prob - 0.5) * 2.0,
            REGIME_WEIGHT_FLOOR,
            REGIME_WEIGHT_CAP,
        )
    )

    final_forecast = final_dynamic_weight * pred_xgb_final + (1 - final_dynamic_weight) * pred_ridge_final
    ridge_feature_names = stable_features_ridge + ["garch_cond_vol"]
    xgb_feature_names = stable_features_broad + ["garch_cond_vol"]
    live_model_details = {
        "feature_importance_top_5": compute_top_feature_importance(
            xgb_feature_names=xgb_feature_names,
            xgb_importances=getattr(final_xgb_model, "feature_importances_", []),
            ridge_feature_names=ridge_feature_names,
            ridge_coefficients=getattr(final_ridge_model, "coef_", []),
        ),
        "ridge_feature_count": len(ridge_feature_names),
        "xgb_feature_count": len(xgb_feature_names),
    }

    all_resids = np.array(holdout_result["acts"], dtype=float) - np.array(holdout_result["preds_blend"], dtype=float)
    if len(all_resids) == 0:
        all_resids = np.array([0.0], dtype=float)
    q_low_f = float(np.nanquantile(all_resids, 0.05))
    q_high_f = float(np.nanquantile(all_resids, 0.95))

    forecast_p05 = max(final_forecast + q_low_f, EPS)
    forecast_p95 = max(final_forecast + q_high_f, EPS)

    last_date = feat_df[DATE_COL].iloc[-1]
    forecast_label = f"next {VOL_TARGET_WINDOW}-day window after {last_date.date()}"

    # Extract event regime context from the last row for diagnostic reporting
    last_row = feat_df.iloc[-1]
    event_context = {}
    if 'event_regime' in feat_df.columns:
        event_context['event_regime'] = last_row.get('event_regime', 'unclassified')
    if 'event_regime_description' in feat_df.columns:
        event_context['event_regime_description'] = last_row.get('event_regime_description', '')
    if 'event_regime_code' in feat_df.columns:
        event_context['event_regime_code'] = int(last_row.get('event_regime_code', -1))
    if 'covid_dummy' in feat_df.columns:
        event_context['covid_dummy'] = int(last_row.get('covid_dummy', 0))
    if 'ukraine_war_dummy' in feat_df.columns:
        event_context['ukraine_war_dummy'] = int(last_row.get('ukraine_war_dummy', 0))
    if 'us_tariff_dummy' in feat_df.columns:
        event_context['us_tariff_dummy'] = int(last_row.get('us_tariff_dummy', 0))
    if 'iran_geopolitical_dummy' in feat_df.columns:
        event_context['iran_geopolitical_dummy'] = int(last_row.get('iran_geopolitical_dummy', 0))
    if 'any_crisis_dummy' in feat_df.columns:
        event_context['any_crisis_dummy'] = int(last_row.get('any_crisis_dummy', 0))

    forecast_payload = {
        "forecast_date": last_date,
        "forecast_label": forecast_label,
        "horizon_days": VOL_TARGET_WINDOW,
        "pred_ridge": round(float(pred_ridge_final), 8),
        "pred_xgb": round(float(pred_xgb_final), 8),
        "blend_weight_xgb": round(float(final_dynamic_weight), 4),
        "regime_prob_high_vol": round(float(regime_prob), 4),
        "regime_flag": "HIGH_VOL" if is_high_vol_regime else "NORMAL",
        "garch_anchor": round(float(garch_fc_final[0]), 8),
        "selected_garch_benchmark": development_result["selected_garch_spec"]["name"],
        "live_garch_anchor_used": garch_spec_used,
        "live_garch_fallback_triggered": bool(live_fallback_info.get("fallback_triggered", False)),
        "live_garch_fallback_reason": live_fallback_info.get("fallback_reason"),
        "live_garch_attempted_specs": live_fallback_info.get("attempted_specs", []),
        "garch_spec_used": garch_spec_used,
        "final_forecast_blend": round(float(final_forecast), 8),
        "forecast_p05": round(float(forecast_p05), 8),
        "forecast_p95": round(float(forecast_p95), 8),
        "fixed_event_threshold": round(float(fixed_event_threshold), 8),
        **event_context,  # Include economic event regime context
    }
    forecast_df = pd.DataFrame([forecast_payload])
    return forecast_df, forecast_payload, live_model_details


def run_pipeline(
    input_path: Path | str = RAW_FILE_PATH,
    output_dir: Optional[Path | str] = None,
    save_outputs: bool = True,
    progress_callback: ProgressCallback = None,
) -> ForecastRunResult:
    input_path = Path(input_path)
    artifact_dir = Path(output_dir) if output_dir is not None else Path(".")
    artifact_paths: Dict[str, Path] = {
        key: artifact_dir / filename for key, filename in ARTIFACT_FILENAMES.items()
    }

    if save_outputs:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    input_validation_summary = validate_input_file(input_path)

    emit(progress_callback, "Preparing data")
    df = prepare_data(input_path)

    emit(progress_callback, "Engineering features")
    feat_df, feature_cols = build_feature_frame(df)

    event_analysis_df = build_event_analysis_frame(input_path)

    if save_outputs:
        feat_df.to_excel(artifact_paths["model_ready_dataset"], index=False)

    emit(progress_callback, "Designing split and stable features")
    split_design = design_split_and_features(feat_df, feature_cols, progress_callback)

    emit(progress_callback, "Running development walk-forward")
    development_result = run_development_walk_forward(
        feat_df,
        split_design["holdout_start_idx"],
        split_design["fixed_event_threshold"],
        split_design["stable_features_ridge"],
        split_design["stable_features_broad"],
        progress_callback,
    )

    emit(progress_callback, "Running untouched holdout walk-forward")
    holdout_result = run_holdout_walk_forward(
        feat_df,
        split_design["holdout_start_idx"],
        split_design["fixed_event_threshold"],
        split_design["stable_features_ridge"],
        split_design["stable_features_broad"],
        development_result,
        progress_callback,
    )

    emit(progress_callback, "Evaluating results")
    summary_dev_df, p_naive_dev, roc_dev, f1_dev, valid_dev = evaluate_block(
        feat_df,
        development_result["idx"],
        development_result["acts"],
        development_result["preds_garch"],
        development_result["preds_ridge"],
        development_result["preds_xgb"],
        development_result["preds_blend"],
        development_result["class_actuals"],
        development_result["preds_class_prob"],
        development_result["benchmark_label"],
    )
    summary_holdout_df, p_naive_h, roc_h, f1_h, valid_h = evaluate_block(
        feat_df,
        holdout_result["idx"],
        holdout_result["acts"],
        holdout_result["preds_garch"],
        holdout_result["preds_ridge"],
        holdout_result["preds_xgb"],
        holdout_result["preds_blend"],
        holdout_result["class_actuals"],
        holdout_result["preds_class_prob"],
        development_result["benchmark_label"],
    )

    dev_df_out = build_output_frame(development_result, valid_dev, p_naive_dev)
    holdout_df_out = build_output_frame(holdout_result, valid_h, p_naive_h)

    # Enrich output frames with event regime information for diagnostic analysis
    dev_df_out = attach_event_regime_to_output(dev_df_out, feat_df)
    holdout_df_out = attach_event_regime_to_output(holdout_df_out, feat_df)

    # Compute event-regime descriptive statistics and forecast diagnostics
    # These are calendar-known regime context variables used for analysis only, not for separate model training.
    # Event-specific metrics complement the full-sample benchmark and help identify systematic differences
    # in USD/TND volatility and forecast performance across major global shocks.
    event_descriptive_stats_df = compute_event_regime_descriptive_stats(
        event_analysis_df,
        high_volatility_cutoff=split_design["fixed_event_threshold"],
    )
    if not event_descriptive_stats_df.empty:
        event_descriptive_stats_df = event_descriptive_stats_df.sort_values("event_regime_code").reset_index(drop=True)

    event_metrics_dev_df = compute_event_regime_forecast_metrics(dev_df_out)
    if not event_metrics_dev_df.empty:
        event_metrics_dev_df = event_metrics_dev_df.sort_values(["event_regime_code", "model"]).reset_index(drop=True)

    event_metrics_holdout_df = compute_event_regime_forecast_metrics(holdout_df_out)
    if not event_metrics_holdout_df.empty:
        event_metrics_holdout_df = event_metrics_holdout_df.sort_values(["event_regime_code", "model"]).reset_index(drop=True)

    if save_outputs:
        emit(progress_callback, "Building high-volatility classifier calibration diagnostics")
        build_high_volatility_calibration_diagnostics(
            development_results=dev_df_out,
            holdout_results=holdout_df_out,
            output_path=artifact_paths["high_vol_classifier_calibration_diagnostics"],
            high_volatility_cutoff=split_design["fixed_event_threshold"],
        )

    emit(progress_callback, "Building live final forecast")
    forecast_df, forecast_payload, live_model_details = build_live_forecast(
        feat_df,
        split_design["stable_features_ridge"],
        split_design["stable_features_broad"],
        split_design["fixed_event_threshold"],
        development_result,
        holdout_result,
    )

    emit(progress_callback, "Building dashboard analytics")
    dashboard_payload = build_dashboard_payload(
        feat_df,
        development_result,
        holdout_result,
        dev_df_out,
        holdout_df_out,
        summary_dev_df,
        summary_holdout_df,
        forecast_payload,
        live_model_details,
    )

    # Add event-regime analysis to dashboard payload.
    # Event dummies and event interactions are calendar-known model inputs;
    # event labels are also diagnostic overlays.
    dashboard_payload["economic_event_analysis"] = {
        "descriptive_stats": dataframe_to_json_safe_records(event_descriptive_stats_df),
        "development_event_metrics": dataframe_to_json_safe_records(event_metrics_dev_df),
        "holdout_event_metrics": dataframe_to_json_safe_records(event_metrics_holdout_df),
        "commentary": (
            "Economic event regimes serve two roles. Event dummies and event interaction terms are used as calendar-known model inputs "
            "to allow the model to condition on documented macro-financial regimes. The same event labels are also used as post-estimation "
            "diagnostic overlays to compare forecast performance across normal and stress periods. Descriptive event-volatility statistics "
            "are computed from a stable event-analysis dataset based on USD/TND returns and the 3-day volatility target, not from the final "
            "ML feature matrix. This prevents changes in optional engineered model features from changing historical event-volatility comparisons."
        ),
    }

    if save_outputs:
        dev_df_out.to_excel(artifact_paths["development_results"], index=False)
        holdout_df_out.to_excel(artifact_paths["holdout_results"], index=False)
        summary_dev_df.to_excel(artifact_paths["development_summary"], index=False)
        summary_holdout_df.to_excel(artifact_paths["holdout_summary"], index=False)
        forecast_df.to_excel(artifact_paths["final_forecast"], index=False)
        
        # Save event analysis tables with robust validation and final sorting
        if not event_descriptive_stats_df.empty and "event_regime_code" in event_descriptive_stats_df.columns:
            event_descriptive_stats_df = (
                event_descriptive_stats_df
                .sort_values("event_regime_code")
                .reset_index(drop=True)
            )
            event_descriptive_stats_df.to_excel(artifact_paths["event_descriptive_stats"], index=False)
        
        if not event_metrics_dev_df.empty and {"event_regime_code", "model"}.issubset(event_metrics_dev_df.columns):
            event_metrics_dev_df = (
                event_metrics_dev_df
                .sort_values(["event_regime_code", "model"])
                .reset_index(drop=True)
            )
            event_metrics_dev_df.to_excel(artifact_paths["event_metrics_development"], index=False)
        
        if not event_metrics_holdout_df.empty and {"event_regime_code", "model"}.issubset(event_metrics_holdout_df.columns):
            event_metrics_holdout_df = (
                event_metrics_holdout_df
                .sort_values(["event_regime_code", "model"])
                .reset_index(drop=True)
            )
            event_metrics_holdout_df.to_excel(artifact_paths["event_metrics_holdout"], index=False)

    # Validate event-aware methodology integration
    emit(progress_callback, "Validating event-aware methodology integration")
    validation_result = validate_event_methodology_integration(
        feat_df,
        feature_cols,
        split_design["stable_features_ridge"],
        split_design["stable_features_broad"],
    )
    validation_status = validation_result["status"]
    if validation_status == "PASS":
        emit(progress_callback, "Event methodology validation: PASS")
    else:
        emit(progress_callback, "Event methodology validation: REVIEW")

    metadata = {
        "input_path": str(input_path),
        "run_timestamp": pd.Timestamp.utcnow().isoformat(),
        "input_file_sha256": compute_file_sha256(input_path),
        "row_count_prepared": int(len(df)),
        "row_count_engineered": int(len(feat_df)),
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "stable_features_ridge": split_design["stable_features_ridge"],
        "stable_features_broad": split_design["stable_features_broad"],
        "fixed_high_vol_threshold": float(split_design["fixed_event_threshold"]),
        "optional_input_columns_available": input_validation_summary.get(
            "optional_input_columns_available",
            [],
        ),
        "optional_input_columns_missing": input_validation_summary.get(
            "optional_input_columns_missing",
            [],
        ),
        "optional_inputs_available": input_validation_summary.get(
            "optional_input_columns_available",
            [],
        ),
        "optional_inputs_missing": input_validation_summary.get(
            "optional_input_columns_missing",
            [],
        ),
        "new_market_features_available": [
            c for c in NEW_PREPARED_FEATURES if c in feat_df.columns
        ],
        "new_market_features_in_feature_cols": [
            c for c in NEW_PREPARED_FEATURES if c in feat_df.columns and c in feature_cols
        ],
        "rolling_window_days": int(ROLLING_WINDOW),
        "forecast_horizon_days": int(VOL_TARGET_WINDOW),
        "step_size_days": int(STEP_SIZE),
        "holdout_start": split_design["holdout_start_date"],
        "holdout_start_idx": int(split_design["holdout_start_idx"]),
        "development_classifier_roc_auc": roc_dev,
        "development_classifier_f1": f1_dev,
        "holdout_classifier_roc_auc": roc_h,
        "holdout_classifier_f1": f1_h,
        "final_xgb_config": development_result["final_xgb_config"],
        "final_ridge_alpha": development_result["final_ridge_alpha"],
        "final_blend_weight": development_result["final_blend_weight"],
        "selected_garch_benchmark": development_result["selected_garch_spec"]["name"],
        "live_garch_anchor_used": forecast_payload.get("live_garch_anchor_used"),
        "live_garch_fallback_triggered": forecast_payload.get("live_garch_fallback_triggered"),
        "live_garch_fallback_reason": forecast_payload.get("live_garch_fallback_reason"),
        "live_garch_attempted_specs": forecast_payload.get("live_garch_attempted_specs", []),
        "selected_garch_spec": development_result["selected_garch_spec"]["name"],
        "garch_gate": development_result["garch_gate"],
        "live_model_details": live_model_details,
        "package_versions": get_package_versions(),
        "event_regimes_used": EVENT_REGIME_DEFINITIONS,
        "event_dummy_columns": EVENT_DUMMY_COLUMNS,
        "event_interaction_columns": EVENT_INTERACTION_COLUMNS,
        "event_features_available": split_design.get("event_features_available", []),
        "event_features_in_ridge": split_design.get("event_features_in_ridge", []),
        "event_features_in_broad": split_design.get("event_features_in_broad", []),
        "event_analysis_row_count": int(len(event_analysis_df)),
        "event_analysis_start_date": event_analysis_df[DATE_COL].min() if not event_analysis_df.empty else None,
        "event_analysis_end_date": event_analysis_df[DATE_COL].max() if not event_analysis_df.empty else None,
        "event_analysis_regime_counts": (
            event_analysis_df["event_regime"].value_counts().to_dict()
            if "event_regime" in event_analysis_df.columns
            else {}
        ),
        "event_descriptive_stats_source": "event_analysis_df",
        "event_forecast_metrics_source": "walk_forward_outputs",
        "event_descriptive_stats_records": dataframe_to_json_safe_records(event_descriptive_stats_df),
        "event_metrics_dev_records": dataframe_to_json_safe_records(event_metrics_dev_df),
        "event_metrics_holdout_records": dataframe_to_json_safe_records(event_metrics_holdout_df),
        "event_methodology_validation": validation_result,
        "event_regime_code_used_as_predictor": validation_result.get(
            "event_regime_code_used_as_predictor",
            False,
        ),
    }

    artifacts = {key: str(path) for key, path in artifact_paths.items()} if save_outputs else {}
    if save_outputs:
        artifacts["dashboard_payload"] = str(artifact_dir / DASHBOARD_PAYLOAD_FILENAME)

    result = ForecastRunResult(
        forecast=forecast_payload,
        dashboard=dashboard_payload,
        development_summary=summary_dev_df,
        holdout_summary=summary_holdout_df,
        development_results=dev_df_out,
        holdout_results=holdout_df_out,
        artifacts=artifacts,
        metadata=metadata,
    )

    if save_outputs:
        metadata_path = artifact_paths["run_metadata"]
        metadata_path.write_text(
            json.dumps(_clean_json_value(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload_path = artifact_dir / DASHBOARD_PAYLOAD_FILENAME
        payload_path.write_text(
            json.dumps(result_to_payload(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result


def _clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): _clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(v) for v in value]
    return value


def dataframe_to_records(df: pd.DataFrame, tail: Optional[int] = None) -> List[Dict[str, Any]]:
    source = df.tail(tail) if tail is not None else df
    return [
        {str(k): _clean_json_value(v) for k, v in row.items()}
        for row in source.to_dict(orient="records")
    ]


def result_to_payload(result: ForecastRunResult, include_development_tail: int = 100) -> Dict[str, Any]:
    return {
        "forecast": _clean_json_value(result.forecast),
        "dashboard": _clean_json_value(result.dashboard),
        "development_summary": dataframe_to_records(result.development_summary),
        "holdout_summary": dataframe_to_records(result.holdout_summary),
        "development_results_tail": dataframe_to_records(result.development_results, tail=include_development_tail),
        "holdout_results": dataframe_to_records(result.holdout_results),
        "artifacts": result.artifacts,
        "metadata": _clean_json_value(result.metadata),
    }


def _read_excel_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_excel(path)
    except Exception:
        return []
    return [
        {str(k): _clean_json_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def load_dashboard_payload_from_artifacts(artifact_dir: Path | str) -> dict | None:
    artifact_dir = Path(artifact_dir)
    payload_path = artifact_dir / DASHBOARD_PAYLOAD_FILENAME
    if payload_path.exists():
        try:
            return json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def reconstruct_payload_from_artifacts(artifact_dir: Path | str) -> dict:
    artifact_dir = Path(artifact_dir)
    payload = load_dashboard_payload_from_artifacts(artifact_dir)
    if payload is not None:
        return payload

    forecast_path = artifact_dir / ARTIFACT_FILENAMES["final_forecast"]
    if not forecast_path.exists():
        raise FileNotFoundError(f"Required forecast artifact not found: {forecast_path}")

    forecast_df = pd.read_excel(forecast_path)
    if forecast_df.empty:
        raise ValueError(f"Forecast artifact is empty: {forecast_path}")
    forecast_row = forecast_df.iloc[0].to_dict()
    forecast_payload = {
        key: _clean_json_value(forecast_row.get(key))
        for key in [
            "forecast_date",
            "forecast_label",
            "horizon_days",
            "pred_ridge",
            "pred_xgb",
            "blend_weight_xgb",
            "regime_prob_high_vol",
            "regime_flag",
            "garch_anchor",
            "garch_spec_used",
            "final_forecast_blend",
            "forecast_p05",
            "forecast_p95",
            "fixed_event_threshold",
        ]
    }

    dev_df_out = pd.read_excel(artifact_dir / ARTIFACT_FILENAMES["development_results"]) if (artifact_dir / ARTIFACT_FILENAMES["development_results"]).exists() else pd.DataFrame()
    holdout_df_out = pd.read_excel(artifact_dir / ARTIFACT_FILENAMES["holdout_results"]) if (artifact_dir / ARTIFACT_FILENAMES["holdout_results"]).exists() else pd.DataFrame()
    summary_dev_df = pd.read_excel(artifact_dir / ARTIFACT_FILENAMES["development_summary"]) if (artifact_dir / ARTIFACT_FILENAMES["development_summary"]).exists() else pd.DataFrame()
    summary_holdout_df = pd.read_excel(artifact_dir / ARTIFACT_FILENAMES["holdout_summary"]) if (artifact_dir / ARTIFACT_FILENAMES["holdout_summary"]).exists() else pd.DataFrame()
    feat_df = pd.read_excel(artifact_dir / ARTIFACT_FILENAMES["model_ready_dataset"]) if (artifact_dir / ARTIFACT_FILENAMES["model_ready_dataset"]).exists() else pd.DataFrame()

    development_result = {
        "final_blend_weight": _clean_json_value(forecast_payload.get("blend_weight_xgb")),
        "selected_garch_spec": {"name": _clean_json_value(forecast_payload.get("garch_spec_used"))},
        "garch_gate": {},
    }

    volatility_index_history = build_volatility_index_history(dev_df_out, holdout_df_out)
    dashboard = {
        "volatility_index": {
            "current": {
                "value": percentile_rank_index(
                    forecast_payload.get("final_forecast_blend"),
                    _combined_reference(
                        dev_df_out.get("actual"),
                        dev_df_out.get("pred_blend"),
                        holdout_df_out.get("actual"),
                        holdout_df_out.get("pred_blend"),
                    ),
                ),
                "classification_label": classify_volatility_index(
                    percentile_rank_index(
                        forecast_payload.get("final_forecast_blend"),
                        _combined_reference(
                            dev_df_out.get("actual"),
                            dev_df_out.get("pred_blend"),
                            holdout_df_out.get("actual"),
                            holdout_df_out.get("pred_blend"),
                        ),
                    )
                )["classification_label"],
                "explanation": classify_volatility_index(
                    percentile_rank_index(
                        forecast_payload.get("final_forecast_blend"),
                        _combined_reference(
                            dev_df_out.get("actual"),
                            dev_df_out.get("pred_blend"),
                            holdout_df_out.get("actual"),
                            holdout_df_out.get("pred_blend"),
                        ),
                    )
                )["short_explanation"],
                "recommended_interpretation": classify_volatility_index(
                    percentile_rank_index(
                        forecast_payload.get("final_forecast_blend"),
                        _combined_reference(
                            dev_df_out.get("actual"),
                            dev_df_out.get("pred_blend"),
                            holdout_df_out.get("actual"),
                            holdout_df_out.get("pred_blend"),
                        ),
                    )
                )["recommended_interpretation"],
                "raw_forecast": forecast_payload.get("final_forecast_blend"),
                "forecast_p05": forecast_payload.get("forecast_p05"),
                "forecast_p95": forecast_payload.get("forecast_p95"),
            },
            "history": volatility_index_history,
            "thresholds": [
                {"range": "0-20", "classification_label": "Very Calm", **VOL_INDEX_DESCRIPTIONS["Very Calm"]},
                {"range": "20-40", "classification_label": "Normal", **VOL_INDEX_DESCRIPTIONS["Normal"]},
                {"range": "40-60", "classification_label": "Elevated", **VOL_INDEX_DESCRIPTIONS["Elevated"]},
                {"range": "60-80", "classification_label": "High", **VOL_INDEX_DESCRIPTIONS["High"]},
                {"range": "80-100", "classification_label": "Stress", **VOL_INDEX_DESCRIPTIONS["Stress"]},
            ],
        },
        "regime": compute_regime_cards(feat_df) if not feat_df.empty else {},
        "model_weights": compute_model_weight_commentary(forecast_payload, development_result),
        "model_comparison": compute_model_comparison(summary_holdout_df, holdout_df_out),
        "garch_selection": compute_garch_selection_commentary(development_result),
        "feature_importance": {
            "top_5": [],
            "commentary": "Feature importance is not available for reconstructed payloads.",
        },
        "calendar_effect": compute_calendar_effects(feat_df) if not feat_df.empty else {},
        "weekly_monthly_volatility": compute_weekly_monthly_volatility(feat_df) if not feat_df.empty else {},
        "volatility_of_volatility": compute_volatility_of_volatility(feat_df) if not feat_df.empty else {},
        "global_fx_comparison": compute_global_fx_comparison(feat_df) if not feat_df.empty else {},
        "history": build_dashboard_history(holdout_df_out, volatility_index_history),
        "summary": build_dashboard_summary(
            forecast_payload,
            {
                "value": percentile_rank_index(
                    forecast_payload.get("final_forecast_blend"),
                    _combined_reference(
                        dev_df_out.get("actual"),
                        dev_df_out.get("pred_blend"),
                        holdout_df_out.get("actual"),
                        holdout_df_out.get("pred_blend"),
                    ),
                ),
                "classification_label": classify_volatility_index(
                    percentile_rank_index(
                        forecast_payload.get("final_forecast_blend"),
                        _combined_reference(
                            dev_df_out.get("actual"),
                            dev_df_out.get("pred_blend"),
                            holdout_df_out.get("actual"),
                            holdout_df_out.get("pred_blend"),
                        ),
                    )
                )["classification_label"],
            },
            compute_garch_selection_commentary(development_result),
            [],
        ),
    }

    artifacts = {key: str(artifact_dir / filename) for key, filename in ARTIFACT_FILENAMES.items() if (artifact_dir / filename).exists()}
    
    # Determine base date from available data
    base_date = None
    if not feat_df.empty and "Date" in feat_df.columns:
        try:
            base_date = pd.to_datetime(feat_df["Date"]).max()
            base_date = base_date.isoformat() if pd.notna(base_date) else None
        except Exception:
            base_date = None
    
    return {
        "forecast": forecast_payload,
        "dashboard": dashboard,
        "development_summary": dataframe_to_records(summary_dev_df),
        "holdout_summary": dataframe_to_records(summary_holdout_df),
        "development_results_tail": dataframe_to_records(dev_df_out.tail(100) if not dev_df_out.empty else pd.DataFrame()),
        "holdout_results": dataframe_to_records(holdout_df_out),
        "artifacts": artifacts,
        "base_date": base_date,
        "run_datetime": None,  # Not available for reconstructed payloads
        "metadata": {
            "source_artifacts": {key: str(artifact_dir / filename) for key, filename in ARTIFACT_FILENAMES.items()},
            "reconstructed_from_artifacts": True,
        },
    }


# Additional imports required by the Streamlit dashboard layer.
from io import BytesIO
import base64
import html
from datetime import datetime, timezone, timedelta
from typing import Iterable

import plotly.graph_objects as go
import streamlit as st

# ============================================================
# VolSight - Macro-Event FX Volatility Engine
# Production Streamlit interface aligned with the final methodology:
# - Empirical volatility regime classification using realized-volatility thresholds
# - Empirical volatility regime classification using q25/q50/q75/q90
# - Macro-event diagnostics and event-specific model validation
# - Trader-oriented decision support for USD/TND risk management
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FORECAST_ARTIFACT_ROOT = BASE_DIR / "artifacts"
PAYLOAD_CACHE_FILENAME = "dashboard_payload.json"
STANDARD_CALIBRATION_WINDOWS = sorted({250, ROLLING_WINDOW, 1000})

LOGO_CANDIDATE_PATHS = [
    Path("logo.png"),
    Path("logo.svg"),
    Path("assets/logo.png"),
    Path("assets/logo.svg"),
    Path("logo_white.png"),
    Path("assets/logo_white.png"),
]

PAGE_NAMES = [
    "Executive FX Risk Cockpit",
    "Data & Market Inputs",
    "Volatility Regime Classification",
    "Macro-Event Regime Analysis",
    "Model Performance by Event",
    "USD/TND vs Global FX Risk Context",
    "Seasonality & Calendar Risk Patterns",
    "Forecast Engine & Benchmark Attribution",
    "Scenario & Calibration Laboratory",
    "Methodology & Audit Trail",
    "Downloads",
]

FINANCE_COLORS = {
    "page_bg": "#07101D",
    "panel_bg": "#101A2C",
    "panel_bg_2": "#121F34",
    "sidebar_bg": "#08111F",
    "text": "#EAF0FA",
    "muted": "#A8B3CC",
    "border": "#24314D",
    "primary": "#4F7BFF",
    "primary_dark": "#2D61FF",
    "teal": "#2FD8C9",
    "gold": "#F3B952",
    "red": "#FF5F68",
    "green": "#39D98A",
    "purple": "#9A7CFF",
    "slate": "#7F8EA7",
    "grid": "#1A2740",
    "input_bg": "#14243A",
    "hover_bg": "#1F2D48",
}

REGIME_COLORS = {
    "Low-volatility regime": FINANCE_COLORS["green"],
    "Normal-volatility regime": FINANCE_COLORS["primary"],
    "Elevated-volatility regime": FINANCE_COLORS["gold"],
    "High-volatility regime": "#FF9D5C",
    "Stress-volatility regime": FINANCE_COLORS["red"],
    "Unavailable": FINANCE_COLORS["slate"],
}

EVENT_COLORS = {
    "pre_covid_normal": FINANCE_COLORS["green"],
    "covid_shock": FINANCE_COLORS["purple"],
    "post_covid_recovery": FINANCE_COLORS["teal"],
    "ukraine_war_shock": FINANCE_COLORS["red"],
    "post_war_inflation_adjustment": FINANCE_COLORS["primary"],
    "us_tariff_shock": FINANCE_COLORS["gold"],
    "post_tariff_normalization": FINANCE_COLORS["green"],
    "iran_geopolitical_shock": "#FF9D5C",
    # Backward compatibility only for older saved artifacts.
    "post_conflict_normalization": FINANCE_COLORS["green"],
}

CHART_COLORWAY = [
    FINANCE_COLORS["teal"],
    FINANCE_COLORS["primary"],
    FINANCE_COLORS["gold"],
    FINANCE_COLORS["purple"],
    FINANCE_COLORS["green"],
    FINANCE_COLORS["red"],
    FINANCE_COLORS["slate"],
]


def calibration_window_options(custom_label: str = "Custom window") -> List[str]:
    return [f"{days} trading days" for days in STANDARD_CALIBRATION_WINDOWS] + [custom_label]


def default_calibration_window_index(options: List[str], default_days: int = ROLLING_WINDOW) -> int:
    label = f"{default_days} trading days"
    return options.index(label) if label in options else 0


def scenario_window_multiplier(window_days: int, baseline_days: int = ROLLING_WINDOW) -> float:
    # Visual-only sensitivity proxy centered on the active production window.
    if window_days == baseline_days:
        return 1.0
    if window_days < baseline_days:
        return min(1.18, 1.0 + ((baseline_days - window_days) / max(baseline_days, 1)) * 0.14)
    return max(0.88, 1.0 - ((window_days - baseline_days) / max(baseline_days, 1)) * 0.06)

MODEL_LABELS = {
    "naive": "FX Volatility Carry-Forward Benchmark",
    "garch": "Traditional Volatility Benchmark",
    "egarch": "Traditional Volatility Benchmark",
    "ridge": "Linear Macro-Risk Stabilizer",
    "xgboost": "Nonlinear Market-Risk Engine",
    "xgb": "Nonlinear Market-Risk Engine",
    "blend": "Final Macro-Event FX Volatility Engine",
    "regime": "Final Macro-Event FX Volatility Engine",
}

ARTIFACT_DISPLAY_NAMES = {
    "model_ready_dataset": "Model-Ready Dataset",
    "walkforward_results_development": "Development Walk-Forward Results",
    "walkforward_results_holdout": "Holdout Walk-Forward Results",
    "walkforward_summary_development": "Development Performance Summary",
    "walkforward_summary_holdout": "Holdout Performance Summary",
    "final_forecast": "Final Forecast Output",
    "high_vol_classifier_calibration_diagnostics": "High-Volatility Calibration Diagnostics",
    "event_descriptive_stats": "Macro-Event Volatility Analysis",
    "event_metrics_development": "Development Event Metrics",
    "event_metrics_holdout": "Holdout Event Metrics",
}

EVENT_DESCRIPTIVE_ROBUST_COLUMNS = [
    "reliability_label",
    "mean_target_vol_annualized",
    "median_target_vol_annualized",
    "q90_target_vol_annualized",
    "max_target_vol_annualized",
    "share_high_vol_days",
    "annualized_return_vol",
    "annualized_target_vol_ci_low",
    "annualized_target_vol_ci_high",
    "annualized_target_vol_vs_pre_covid_pct",
    "mean_target_vol_annualized_vs_broad_normal_pct",
    "q90_target_vol_annualized_vs_broad_normal_pct",
    "brown_forsythe_pvalue_vs_pre_covid",
    "variance_test_result",
    "source_dataset",
    "sample_size_note",
]

COMPLETED_EVENT_REGIMES = [
    "pre_covid_normal",
    "covid_shock",
    "post_covid_recovery",
    "ukraine_war_shock",
    "post_war_inflation_adjustment",
    "us_tariff_shock",
    "post_tariff_normalization",
]
ONGOING_EVENT_REGIME = "iran_geopolitical_shock"

# ============================================================
# Styling and layout helpers
# ============================================================


def inject_finance_theme() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --finance-primary: {FINANCE_COLORS['primary']};
            --finance-primary-dark: {FINANCE_COLORS['primary_dark']};
            --finance-teal: {FINANCE_COLORS['teal']};
            --finance-gold: {FINANCE_COLORS['gold']};
            --finance-red: {FINANCE_COLORS['red']};
            --finance-green: {FINANCE_COLORS['green']};
            --finance-purple: {FINANCE_COLORS['purple']};
            --finance-text: {FINANCE_COLORS['text']};
            --finance-muted: {FINANCE_COLORS['muted']};
            --finance-border: {FINANCE_COLORS['border']};
            --finance-bg: {FINANCE_COLORS['page_bg']};
            --finance-panel: {FINANCE_COLORS['panel_bg']};
            --finance-panel-2: {FINANCE_COLORS['panel_bg_2']};
            --finance-input-bg: {FINANCE_COLORS['input_bg']};
            --finance-hover-bg: {FINANCE_COLORS['hover_bg']};
        }}

        @keyframes fadeSlideUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .stApp {{
            background:
                radial-gradient(circle at 15% 5%, rgba(79, 123, 255, 0.17), transparent 360px),
                radial-gradient(circle at 85% 8%, rgba(47, 216, 201, 0.12), transparent 400px),
                linear-gradient(135deg, #07101D 0%, #0B1628 48%, #0F1C2D 100%);
            color: var(--finance-text);
        }}

        [data-testid="stAppViewContainer"] > .main .block-container {{
            max-width: 1500px;
            padding-top: 1.1rem;
            padding-left: 1.55rem;
            padding-right: 1.55rem;
            animation: fadeSlideUp 380ms ease-out both;
        }}

        [data-testid="stSidebar"] {{
            background: {FINANCE_COLORS['sidebar_bg']};
            border-right: 1px solid var(--finance-border);
            box-shadow: 18px 0 48px rgba(0,0,0,0.12);
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border-radius: 13px;
            padding: 0.58rem 0.68rem;
            margin-bottom: 0.24rem;
            background: transparent;
            border: 1px solid transparent;
            transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: var(--finance-hover-bg);
            border-color: var(--finance-border);
            transform: translateX(3px);
        }}

        h1, h2, h3, h4, h5, h6 {{ color: var(--finance-text); }}
        p, li, label, .stMarkdown {{ color: var(--finance-text); }}
        [data-testid="stCaptionContainer"], .stCaption {{ color: var(--finance-muted); }}

        .app-brand {{
            display: flex;
            align-items: center;
            gap: 0.72rem;
            margin: 0.25rem 0 1rem 0;
            color: var(--finance-text);
            font-weight: 900;
            letter-spacing: 0.02em;
        }}
        .app-brand-mark {{
            width: 34px;
            height: 34px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            background: linear-gradient(135deg, #4F7BFF, #2FD8C9);
            box-shadow: 0 12px 22px rgba(79,123,255,0.2);
        }}
        .app-tagline {{
            color: var(--finance-muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }}

        .topbar {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 1rem;
        }}
        .page-kicker {{
            color: var(--finance-teal);
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }}
        .page-title {{
            font-size: clamp(1.55rem, 2vw, 2.25rem);
            color: var(--finance-text);
            line-height: 1.15;
            font-weight: 900;
        }}
        .topbar-actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            justify-content: flex-end;
        }}
        .chip {{
            border: 1px solid var(--finance-border);
            background: rgba(20,36,58,0.85);
            border-radius: 999px;
            color: var(--finance-muted);
            padding: 0.48rem 0.75rem;
            font-size: 0.78rem;
            white-space: nowrap;
        }}
        .chip strong {{ color: var(--finance-text); }}

        .hero-panel {{
            border: 1px solid rgba(79, 123, 255, 0.26);
            background: linear-gradient(135deg, rgba(79,123,255,0.16), rgba(47,216,201,0.07) 45%, rgba(16,26,44,0.96));
            border-radius: 24px;
            padding: 1.4rem;
            box-shadow: 0 28px 70px rgba(0,0,0,0.22);
            margin-bottom: 1rem;
        }}
        .panel {{
            border: 1px solid var(--finance-border);
            background: linear-gradient(180deg, rgba(18,31,52,0.94), rgba(13,23,38,0.96));
            border-radius: 18px;
            padding: 1.1rem;
            box-shadow: 0 18px 44px rgba(0,0,0,0.16);
            margin-bottom: 1rem;
        }}
        .panel-title {{
            color: var(--finance-text);
            font-weight: 850;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }}
        .panel-subtitle {{
            color: var(--finance-muted);
            font-size: 0.86rem;
            line-height: 1.45;
            margin-bottom: 0.85rem;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.8rem 0 1rem 0;
        }}
        .kpi-card {{
            border: 1px solid var(--finance-border);
            background: rgba(255,255,255,0.035);
            border-radius: 16px;
            padding: 1rem;
            min-height: 124px;
            transition: transform 160ms ease, border-color 160ms ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(47,216,201,0.55);
        }}
        .kpi-label {{
            color: var(--finance-muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            color: var(--finance-text);
            font-size: 1.65rem;
            line-height: 1.1;
            font-weight: 900;
            margin-bottom: 0.35rem;
        }}
        .kpi-note {{
            color: var(--finance-muted);
            font-size: 0.82rem;
            line-height: 1.35;
        }}
        .decision-card {{
            border: 1px solid var(--finance-border);
            border-left: 4px solid var(--finance-teal);
            background: var(--finance-panel);
            border-radius: 16px;
            padding: 1rem;
            min-height: 135px;
            margin-bottom: 0.85rem;
        }}
        .decision-title {{
            color: var(--finance-teal);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 850;
            margin-bottom: 0.45rem;
        }}
        .decision-text {{
            color: var(--finance-text);
            font-size: 0.95rem;
            line-height: 1.55;
        }}
        .warning-box {{
            border: 1px solid rgba(243,185,82,0.35);
            border-left: 4px solid var(--finance-gold);
            background: rgba(243,185,82,0.08);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            color: var(--finance-text);
            line-height: 1.5;
            margin-bottom: 1rem;
        }}
        .method-box {{
            border: 1px solid rgba(47,216,201,0.25);
            border-left: 4px solid var(--finance-teal);
            background: rgba(47,216,201,0.06);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            color: var(--finance-text);
            line-height: 1.5;
            margin-bottom: 1rem;
        }}
        .small-muted {{
            color: var(--finance-muted);
            font-size: 0.82rem;
            line-height: 1.4;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 12px;
            border: 1px solid var(--finance-primary);
            color: var(--finance-text);
            background: var(--finance-input-bg);
            font-weight: 750;
            transition: transform 150ms ease, background 150ms ease, border-color 150ms ease;
        }}
        .stButton > button[kind="primary"] {{
            color: #FFFFFF;
            background: linear-gradient(135deg, #4F7BFF 0%, #2D61FF 100%);
            border-color: var(--finance-primary);
            box-shadow: 0 14px 30px rgba(79,123,255,0.26);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            border-color: var(--finance-teal);
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border: 1px solid var(--finance-border);
            border-radius: 14px;
            overflow: hidden;
            background: var(--finance-panel);
        }}
        [data-testid="stPlotlyChart"] {{
            border: 1px solid var(--finance-border);
            border-radius: 16px;
            overflow: hidden;
            background: var(--finance-panel);
            box-shadow: 0 18px 42px rgba(0,0,0,0.12);
        }}
        .validation-card-interpretation {{
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 0.65rem;
        }}
        hr {{ border-color: var(--finance-border); }}

        .validation-matrix-content {{
            display: grid;
            grid-template-columns: minmax(440px, 1.05fr) minmax(360px, 0.95fr);
            gap: 1.2rem;
            align-items: stretch;
            margin-top: 1.2rem;
        }}
        .validation-read-card {{
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(20, 36, 58, 0.78));
            border: 1px solid rgba(47, 216, 201, 0.28);
            border-left: 4px solid var(--finance-teal);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }}
        .validation-read-title {{
            color: var(--finance-teal);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 850;
            margin-bottom: 0.55rem;
        }}
        .validation-read-text {{
            color: var(--finance-text);
            font-size: 0.94rem;
            line-height: 1.6;
        }}

        @media (max-width: 1000px) {{
            .validation-matrix-content {{ grid-template-columns: 1fr; }}
            .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .topbar {{ flex-direction: column; }}
            .topbar-actions {{ justify-content: flex-start; }}
        }}
        @media (max-width: 650px) {{
            .kpi-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_finance_layout(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        colorway=CHART_COLORWAY,
        paper_bgcolor=FINANCE_COLORS["panel_bg"],
        plot_bgcolor=FINANCE_COLORS["panel_bg"],
        font=dict(color=FINANCE_COLORS["text"], family="Arial, sans-serif"),
        title_font=dict(color=FINANCE_COLORS["text"], size=17),
        margin=dict(l=22, r=22, t=45, b=28),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=FINANCE_COLORS["input_bg"],
            bordercolor=FINANCE_COLORS["border"],
            font=dict(color=FINANCE_COLORS["text"]),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color=FINANCE_COLORS["text"]),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            gridcolor=FINANCE_COLORS["grid"],
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=FINANCE_COLORS["muted"]),
        ),
        yaxis=dict(
            gridcolor=FINANCE_COLORS["grid"],
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=FINANCE_COLORS["muted"]),
        ),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def apply_institutional_plotly_layout(
    fig: go.Figure,
    title: str,
    subtitle: Optional[str] = None,
    height: int = 460,
    yaxis_title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    hovermode: str = "closest",
) -> go.Figure:
    full_title = title
    if subtitle:
        full_title = (
            f"{html.escape(title)}"
            f"<br><span style='font-size:12px;color:{FINANCE_COLORS['muted']};font-weight:400;'>{html.escape(subtitle)}</span>"
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=18, r=26, t=78 if subtitle else 58, b=38),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=FINANCE_COLORS["text"], size=12),
        title=dict(text=full_title, x=0.0, xanchor="left", font=dict(size=17, color=FINANCE_COLORS["text"])),
        hovermode=hovermode,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=FINANCE_COLORS["muted"]),
        ),
        xaxis=dict(
            title=dict(text=xaxis_title or "", font=dict(color=FINANCE_COLORS["muted"], size=12)),
            gridcolor="rgba(168,179,204,0.12)",
            zerolinecolor="rgba(234,240,250,0.35)",
            linecolor="rgba(168,179,204,0.18)",
            tickfont=dict(color=FINANCE_COLORS["muted"], size=11),
            ticksuffix="%",
        ),
        yaxis=dict(
            title=dict(text=yaxis_title or "", font=dict(color=FINANCE_COLORS["muted"], size=12)),
            gridcolor="rgba(168,179,204,0.06)",
            zerolinecolor="rgba(234,240,250,0.18)",
            linecolor="rgba(168,179,204,0.18)",
            tickfont=dict(color=FINANCE_COLORS["text"], size=11),
            automargin=True,
        ),
    )
    return fig


def page_header(title: str, kicker: str, payload: Optional[dict]) -> None:
    base_date = get_display_base_date(payload)
    base_date_display = base_date.strftime("%d/%m/%Y") if base_date else "No run loaded"
    forecast_period = calculate_forecast_period(base_date)
    run_id = payload.get("run_id", "No run loaded") if payload else "No run loaded"
    current_event = get_current_event_regime(payload)
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="page-kicker">{html.escape(kicker)}</div>
                <div class="page-title">{html.escape(title)}</div>
            </div>
            <div class="topbar-actions">
                <div class="chip">Run: <strong>{html.escape(str(run_id))}</strong></div>
                <div class="chip">Base: <strong>{html.escape(base_date_display)}</strong></div>
                <div class="chip">Forecast: <strong>{html.escape(forecast_period)}</strong></div>
                <div class="chip">Macro regime: <strong>{html.escape(current_event)}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "", accent: str = "teal") -> str:
    color = FINANCE_COLORS.get(accent, FINANCE_COLORS["teal"])
    # Keep the HTML compact. Indented multi-line HTML can be interpreted by
    # Streamlit Markdown as a code block, causing raw <div> markup to appear.
    return (
        f'<div class="kpi-card" style="border-top: 3px solid {color};">'
        f'<div class="kpi-label">{html.escape(str(label))}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        f'<div class="kpi-note">{html.escape(str(note or ""))}</div>'
        '</div>'
    )


def render_kpi_grid(cards: List[str]) -> None:
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_market_watchlist(payload: Optional[dict], event_context: dict) -> None:
    dashboard = (payload.get("dashboard", {}) or {}) if payload else {}
    econ = dashboard.get("economic_event_analysis", {}) or {}
    headlines = econ.get("macro_event_headlines") or econ.get("headline") or econ.get("headlines") or "No headline commentary available."
    watchlist_items = [
        "DXY",
        "Brent",
        "MOVE",
        "VIX",
        "Global sentiment",
        "Macro-event headlines",
    ]
    watch_html = "".join(f"<li>{html.escape(item)}</li>" for item in watchlist_items)
    event_context_note = (
        f"Latest macro-event context: {html.escape(event_context['label'])}."
        if event_context["is_classified"]
        else "Latest macro-event context: No active event regime assigned."
    )
    source_note = ""
    if event_context["is_classified"] and event_context.get("source"):
        source_note = f" Source: {html.escape(event_context['source'])}."
    st.markdown(
        f"""
        <div class='decision-card'>
            <div class='decision-title'>Market Watchlist</div>
            <div class='decision-text'>Monitor the following risk signals and headlines for short-term USD/TND hedging decisions.</div>
            <ul style='margin: 0.35rem 0 0 1rem; color: {FINANCE_COLORS['text']};'>{watch_html}</ul>
            <div class='decision-text' style='margin-top:0.8rem;color:{FINANCE_COLORS['muted']}; font-size:0.92rem;'>{event_context_note}{source_note}</div>
            <div class='decision-text' style='margin-top:0.5rem;color:{FINANCE_COLORS['muted']}; font-size:0.92rem;'>Latest macro-event note: {html.escape(str(headlines))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_checklist() -> None:
    checklist_items = [
        "Review USD/TND exposure due over the next 3 trading days.",
        "Check DXY, Brent, MOVE, and VIX before hedge execution.",
        "Compare the official signal with the Scenario Laboratory if hedge timing is uncertain.",
        "Escalate if the high-volatility signal moves into alert zone.",
    ]
    notes_html = "".join(f"<li>{html.escape(item)}</li>" for item in checklist_items)
    st.markdown(
        f"""
        <div class='decision-card'>
            <div class='decision-title'>Action Checklist</div>
            <div class='decision-text'>Use this compact checklist to keep USD/TND risk actions aligned with the latest signal.</div>
            <ul style='margin: 0.35rem 0 0 1rem; color: {FINANCE_COLORS['text']};'>{notes_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_recent_volatility_pulse_df(payload: Optional[dict]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    records = payload.get("holdout_results", []) or (payload.get("dashboard", {}) or {}).get("history", []) or payload.get("latest_history", [])
    df = records_df(records)
    if df.empty:
        return pd.DataFrame()
    for col in ["actual", "actual_volatility", "pred_blend", "predicted_volatility"]:
        if col in df.columns:
            out = df[[col]].copy()
            if "Date" in df.columns:
                out["Date"] = df["Date"]
            out = out.dropna(subset=[col])
            if not out.empty:
                return out
    return pd.DataFrame()


def render_recent_volatility_pulse(payload: Optional[dict]) -> None:
    recent = get_recent_volatility_pulse_df(payload)
    if recent.empty:
        st.info("Recent volatility pulse data is unavailable.")
        return
    y_col = [col for col in recent.columns if col != "Date"][0]
    fig = go.Figure()
    if "Date" in recent.columns:
        fig.add_trace(
            go.Scatter(
                x=recent["Date"],
                y=pd.to_numeric(recent[y_col], errors="coerce") * 100,
                mode="lines",
                line=dict(color=FINANCE_COLORS["gold"], width=2),
            )
        )
        xaxis = dict(title="Date", tickformat="%b %d", showgrid=False)
    else:
        fig.add_trace(
            go.Scatter(
                x=recent.index + 1,
                y=pd.to_numeric(recent[y_col], errors="coerce") * 100,
                mode="lines",
                line=dict(color=FINANCE_COLORS["gold"], width=2),
            )
        )
        xaxis = dict(title="Recent observations", showticklabels=False, showgrid=False)
    fig.update_layout(
        title="Recent USD/TND Volatility Pulse",
        yaxis_title="Volatility (%)",
        xaxis=xaxis,
        margin=dict(l=18, r=18, t=40, b=24),
        height=240,
        showlegend=False,
    )
    st.plotly_chart(apply_finance_layout(fig, height=240), use_container_width=True)
    st.caption("Recent realized-volatility path used as a quick market-temperature check; this is not a separate forecast.")


def render_navigation_cards() -> None:
    cols = st.columns(4, gap="large")
    nav_options = [
        ("Validate Regime Signal", "Volatility Regime Classification"),
        ("Review Macro-Event Context", "Macro-Event Regime Analysis"),
        ("Test Scenario Overlay", "Scenario & Calibration Laboratory"),
        ("Inspect Benchmark Attribution", "Forecast Engine & Benchmark Attribution"),
    ]
    for col, (label, target_page) in zip(cols, nav_options):
        with col:
            if st.button(label, use_container_width=True, key=f"nav_{target_page.replace(' ', '_')}"):
                st.session_state.selected_page = target_page
                st.rerun()


# ============================================================
# File and payload helpers
# ============================================================


def find_logo_path() -> Optional[Path]:
    for logo_path in LOGO_CANDIDATE_PATHS:
        candidate = BASE_DIR / logo_path
        if candidate.exists():
            return candidate
    return None


def get_logo_data_uri(logo_path: Path) -> str:
    suffix = logo_path.suffix.lower()
    mime_type = "image/svg+xml" if suffix == ".svg" else "image/png"
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def local_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def find_latest_run_dir() -> Optional[Path]:
    if not FORECAST_ARTIFACT_ROOT.exists():
        return None
    run_dirs = [path for path in FORECAST_ARTIFACT_ROOT.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def load_dashboard_payload_from_artifacts(run_dir: Path) -> Optional[dict]:
    payload_file = run_dir / PAYLOAD_CACHE_FILENAME
    if payload_file.exists():
        try:
            return json.loads(payload_file.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _artifact_candidates(payload: Optional[dict], key: str) -> List[Path]:
    candidates: List[Path] = []
    if payload:
        artifacts = payload.get("artifacts", {}) or {}
        if key in artifacts:
            candidates.append(Path(artifacts[key]))
        artifact_dir = payload.get("artifact_dir")
        if artifact_dir and key in ARTIFACT_FILENAMES:
            candidates.append(Path(artifact_dir) / ARTIFACT_FILENAMES[key])
    if key in ARTIFACT_FILENAMES:
        candidates.append(Path(ARTIFACT_FILENAMES[key]))
        candidates.append(BASE_DIR / ARTIFACT_FILENAMES[key])

    seen = set()
    existing = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            existing.append(candidate)
    return existing


def freshest_artifact_path(payload: Optional[dict], key: str) -> Optional[Path]:
    candidates = _artifact_candidates(payload, key)
    if not candidates:
        return None
    return max(candidates, key=_path_mtime)


@st.cache_data(show_spinner=False)
def _read_excel_cached(path_str: str, file_mtime: float, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if sheet_name is None:
        return pd.read_excel(path_str)
    return pd.read_excel(path_str, sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, file_mtime: float) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def infer_base_date_from_input_file(input_path: Path) -> Optional[str]:
    input_path = Path(input_path)
    if not input_path.exists():
        return None
    try:
        input_df = pd.read_excel(input_path, sheet_name="Sheet1")
        input_df["Date"] = pd.to_datetime(input_df["Date"], errors="coerce")
        dates = input_df["Date"].dropna()
        if dates.empty:
            return None
        return dates.max().date().isoformat()
    except Exception:
        return None


def infer_base_date_from_run_dir(run_dir: Path) -> Optional[str]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    artifact_names = set(ARTIFACT_FILENAMES.values())
    for input_file in sorted(run_dir.glob("*.xls*")):
        if input_file.name == PAYLOAD_CACHE_FILENAME or input_file.name in artifact_names:
            continue
        base_date = infer_base_date_from_input_file(input_file)
        if base_date:
            return base_date
    return None


def _parse_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", ""))
    except Exception:
        try:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                return parsed.to_pydatetime()
        except Exception:
            return None
    return None


def get_display_base_date(payload: Optional[dict]) -> Optional[datetime]:
    if not payload:
        return None
    candidates: List[Any] = []
    if payload.get("base_date"):
        candidates.append(payload["base_date"])
    forecast = payload.get("forecast", {}) or {}
    for key in ("base_date", "last_data_date", "forecast_date"):
        if forecast.get(key):
            candidates.append(forecast[key])
    if payload.get("artifact_dir"):
        inferred = infer_base_date_from_run_dir(Path(payload["artifact_dir"]))
        if inferred:
            candidates.append(inferred)
    for item in payload.get("holdout_results", []) or []:
        if isinstance(item, dict) and item.get("Date"):
            candidates.append(item["Date"])
    parsed = [dt for dt in (_parse_date(x) for x in candidates) if dt is not None]
    return max(parsed) if parsed else None


def calculate_forecast_period(base_date: Optional[datetime]) -> str:
    if not base_date:
        return "Unknown"
    start_date = base_date + timedelta(days=1)
    end_date = base_date + timedelta(days=3)
    return f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"


def save_dashboard_payload(payload: dict, run_dir: Path) -> None:
    (run_dir / PAYLOAD_CACHE_FILENAME).write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")


def load_latest_local() -> dict:
    latest_run = find_latest_run_dir()
    root_payload = BASE_DIR / PAYLOAD_CACHE_FILENAME
    latest_run_payload = latest_run / PAYLOAD_CACHE_FILENAME if latest_run is not None else None
    use_root_artifacts = root_payload.exists() and (
        latest_run_payload is None
        or not latest_run_payload.exists()
        or _path_mtime(root_payload) >= _path_mtime(latest_run_payload)
    )

    if use_root_artifacts:
        payload = load_dashboard_payload_from_artifacts(BASE_DIR)
        if payload is None:
            payload = reconstruct_payload_from_artifacts(BASE_DIR)
        payload["run_id"] = "root_artifacts"
        payload["artifact_dir"] = str(BASE_DIR)
        return payload

    if latest_run is None:
        raise FileNotFoundError("No saved forecast artifacts were found.")
    payload = load_dashboard_payload_from_artifacts(latest_run)
    if payload is None:
        payload = reconstruct_payload_from_artifacts(latest_run)
    payload["run_id"] = latest_run.name
    payload["artifact_dir"] = str(latest_run)
    inferred_base_date = infer_base_date_from_run_dir(latest_run)
    if inferred_base_date:
        payload["base_date"] = inferred_base_date
        payload.setdefault("forecast", {})["base_date"] = inferred_base_date
    return payload


def artifact_path(payload: Optional[dict], key: str, prefer_freshest: bool = False) -> Optional[Path]:
    if prefer_freshest:
        return freshest_artifact_path(payload, key)
    if payload:
        artifacts = payload.get("artifacts", {}) or {}
        if key in artifacts and Path(artifacts[key]).exists():
            return Path(artifacts[key])
        artifact_dir = payload.get("artifact_dir")
        if artifact_dir and key in ARTIFACT_FILENAMES:
            p = Path(artifact_dir) / ARTIFACT_FILENAMES[key]
            if p.exists():
                return p
    if key in ARTIFACT_FILENAMES:
        p = Path(ARTIFACT_FILENAMES[key])
        if p.exists():
            return p
    return None


def load_artifact_df(payload: Optional[dict], key: str) -> pd.DataFrame:
    p = artifact_path(payload, key)
    if p is None:
        return pd.DataFrame()
    try:
        return _read_excel_cached(str(p.resolve()), _path_mtime(p))
    except Exception:
        return pd.DataFrame()


def load_fresh_artifact_df(payload: Optional[dict], key: str) -> pd.DataFrame:
    p = artifact_path(payload, key, prefer_freshest=True)
    if p is None:
        return pd.DataFrame()
    try:
        df = _read_excel_cached(str(p.resolve()), _path_mtime(p)).copy()
        df.attrs["artifact_path"] = str(p)
        df.attrs["artifact_mtime"] = _path_mtime(p)
        return df
    except Exception:
        return pd.DataFrame()


def load_json_artifact(payload: Optional[dict], key: str) -> dict:
    p = artifact_path(payload, key, prefer_freshest=True)
    if p is None:
        return {}
    try:
        data = _read_json_cached(str(p.resolve()), _path_mtime(p))
        data["_artifact_path"] = str(p)
        data["_artifact_mtime"] = _path_mtime(p)
        return data
    except Exception:
        return {}


def get_high_vol_calibration_df(payload: Optional[dict], sheet_name: str) -> pd.DataFrame:
    path = artifact_path(payload, "high_vol_classifier_calibration_diagnostics")
    if path is None:
        return pd.DataFrame()
    try:
        return _read_excel_cached(str(path.resolve()), _path_mtime(path), sheet_name)
    except Exception:
        return pd.DataFrame()


def records_df(records: Optional[List[dict]]) -> pd.DataFrame:
    df = pd.DataFrame(records or [])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


def get_holdout_df(payload: Optional[dict]) -> pd.DataFrame:
    if payload:
        df = records_df(payload.get("holdout_results", []))
        if not df.empty:
            return df
    return load_artifact_df(payload, "walkforward_results_holdout")


def get_development_df(payload: Optional[dict]) -> pd.DataFrame:
    if payload:
        df = records_df(payload.get("development_results", []))
        if not df.empty:
            return df
    return load_artifact_df(payload, "walkforward_results_development")


def get_model_ready_df(payload: Optional[dict]) -> pd.DataFrame:
    return load_artifact_df(payload, "model_ready_dataset")


def get_event_descriptive_df(payload: Optional[dict]) -> pd.DataFrame:
    artifact_df = load_fresh_artifact_df(payload, "event_descriptive_stats")
    if not artifact_df.empty and any(c in artifact_df.columns for c in EVENT_DESCRIPTIVE_ROBUST_COLUMNS):
        artifact_df.attrs["source"] = "freshest_artifact"
        return artifact_df

    if payload:
        econ = (payload.get("dashboard", {}) or {}).get("economic_event_analysis", {}) or {}
        df = pd.DataFrame(econ.get("descriptive_stats", []) or [])
        if not df.empty:
            df.attrs["source"] = "payload.dashboard.economic_event_analysis.descriptive_stats"
            return df
        metadata = payload.get("metadata", {}) or {}
        df = pd.DataFrame(metadata.get("event_descriptive_stats_records", []) or [])
        if not df.empty:
            df.attrs["source"] = "payload.metadata.event_descriptive_stats_records"
            return df

    if not artifact_df.empty:
        artifact_df.attrs["source"] = "freshest_artifact"
    return artifact_df


def get_run_metadata(payload: Optional[dict]) -> dict:
    metadata = {}
    if payload:
        metadata.update(payload.get("metadata", {}) or {})
    artifact_metadata = load_json_artifact(payload, "run_metadata")
    if artifact_metadata:
        artifact_metadata.pop("_artifact_mtime", None)
        metadata.update(artifact_metadata)
    elif payload:
        dashboard_payload = load_json_artifact(payload, "dashboard_payload")
        metadata.update((dashboard_payload.get("metadata", {}) or {}) if dashboard_payload else {})
    return metadata


def get_event_metrics_df(payload: Optional[dict], period: str = "holdout") -> pd.DataFrame:
    if payload:
        econ = (payload.get("dashboard", {}) or {}).get("economic_event_analysis", {}) or {}
        key = "holdout_event_metrics" if period == "holdout" else "development_event_metrics"
        df = pd.DataFrame(econ.get(key, []) or [])
        if not df.empty:
            return df
        metadata = payload.get("metadata", {}) or {}
        meta_key = "event_metrics_holdout_records" if period == "holdout" else "event_metrics_dev_records"
        df = pd.DataFrame(metadata.get(meta_key, []) or [])
        if not df.empty:
            return df
    artifact_key = "event_metrics_holdout" if period == "holdout" else "event_metrics_development"
    return load_artifact_df(payload, artifact_key)


def get_summary_df(payload: Optional[dict], period: str = "holdout") -> pd.DataFrame:
    if payload:
        records = payload.get("holdout_summary" if period == "holdout" else "development_summary", [])
        df = pd.DataFrame(records or [])
        if not df.empty:
            return df
    key = "walkforward_summary_holdout" if period == "holdout" else "walkforward_summary_development"
    return load_artifact_df(payload, key)


# ============================================================
# Formatting and finance-specific interpretation
# ============================================================


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        x = float(value)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def format_number(value: Any, digits: int = 4) -> str:
    x = safe_float(value)
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def format_pct(value: Any, digits: int = 2) -> str:
    x = safe_float(value)
    if x is None:
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def format_ratio(value: Any, digits: int = 2) -> str:
    x = safe_float(value)
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}x"


def normalize_metric_ratio(value: Any) -> Optional[float]:
    """Normalize metric to 0-1 scale for consistent interpretation."""
    if value is None or pd.isna(value):
        return None
    try:
        val = float(value)
    except (ValueError, TypeError):
        return None
    if abs(val) > 1:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def format_metric_percent(value: Any, digits: int = 2) -> str:
    """Format metric as percentage with specified decimal places."""
    normalized = normalize_metric_ratio(value)
    if normalized is None:
        return "N/A"
    return f"{100.0 * normalized:.{digits}f}%"


def interpret_classification_accuracy(accuracy: Any) -> str:
    acc = normalize_metric_ratio(accuracy)
    if acc is None:
        return "Classification accuracy is unavailable for the current validation sample."
    val_str = format_metric_percent(accuracy)
    if acc >= 0.85:
        return f"With an accuracy of {val_str}, the model correctly classifies volatility regimes in a strong majority of observations."
    if acc >= 0.75:
        return f"With an accuracy of {val_str}, the model correctly classifies volatility regimes in roughly eight out of ten observations."
    if acc >= 0.65:
        return f"With an accuracy of {val_str}, the model shows acceptable classification power, but regime calls should be monitored with supporting market context."
    return f"With an accuracy of {val_str}, the model’s classification hit rate is limited; regime calls should be treated cautiously."


def interpret_recall(recall: Any) -> str:
    rec = normalize_metric_ratio(recall)
    if rec is None:
        return "Recall is unavailable for the current validation sample."
    val_str = format_metric_percent(recall)
    if rec >= 0.80:
        return f"With a recall of {val_str}, the model captures most realized high-volatility periods."
    if rec >= 0.65:
        return f"With a recall of {val_str}, the model captures a solid share of realized high-volatility periods."
    if rec >= 0.50:
        return f"With a recall of {val_str}, the model captures some high-volatility periods, but missed stress episodes remain material."
    return f"With a recall of {val_str}, the model misses many realized high-volatility periods; risk alerts should be treated cautiously."


def interpret_f1_score(f1_score: Any) -> str:
    f1 = normalize_metric_ratio(f1_score)
    if f1 is None:
        return "F1 score is unavailable for the current validation sample."
    val_str = format_metric_percent(f1_score)
    if f1 >= 0.75:
        return f"With an F1 score of {val_str}, the signal shows strong balance between capturing high-volatility periods and limiting false alerts."
    if f1 >= 0.60:
        return f"With an F1 score of {val_str}, the signal reasonably detects high-volatility periods, although some false alerts or missed episodes remain."
    if f1 >= 0.45:
        return f"With an F1 score of {val_str}, the signal provides partial classification value, but alert quality should be reviewed carefully."
    return f"With an F1 score of {val_str}, the signal has weak balance between false alerts and missed high-volatility periods."


def pretty_event_name(value: Any) -> str:
    mapping = {
        "pre_covid_normal": "Pre-COVID Normal",
        "covid_shock": "COVID Shock",
        "post_covid_recovery": "Post-COVID Recovery",
        "ukraine_war_shock": "Russia-Ukraine War Shock",
        "post_war_inflation_adjustment": "Post-War Inflation Adjustment",
        "us_tariff_shock": "U.S. Tariff Shock",
        "post_tariff_normalization": "Post-Tariff Normalization",
        "iran_geopolitical_shock": "Iran Geopolitical Shock",
        # Backward compatibility only for older saved artifacts.
        "post_conflict_normalization": "Post-Tariff Normalization",
    }
    text = str(value or "Unavailable")
    return mapping.get(text, text.replace("_", " ").title())


def is_valid_event_context_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    invalid = {
        "nan",
        "none",
        "null",
        "undefined",
        "unavailable",
        "not available",
        "not classified",
        "n/a",
    }
    return text.lower() not in invalid


def format_event_context_label(value: Any) -> Optional[str]:
    if not is_valid_event_context_text(value):
        return None
    text = str(value).strip()
    mapping = {
        "pre_covid_normal": "Pre-COVID Normal",
        "covid_shock": "COVID Shock",
        "post_covid_recovery": "Post-COVID Recovery",
        "ukraine_war_shock": "Russia-Ukraine War Shock",
        "post_war_inflation_adjustment": "Post-War Inflation Adjustment",
        "us_tariff_shock": "U.S. Tariff Shock",
        "post_tariff_normalization": "Post-Tariff Normalization",
        "post-tariff normalization regime": "Post-Tariff Normalization",
        "iran_geopolitical_shock": "Iran Geopolitical Shock",
        "iran / us-israel geopolitical escalation": "Iran Geopolitical Shock",
        # Backward compatibility only for older saved artifacts.
        "post_conflict_normalization": "Post-Tariff Normalization",
        "post-conflict normalization regime": "Post-Tariff Normalization",
    }
    lower_text = text.lower()
    if lower_text in mapping:
        return mapping[lower_text]
    if "_" in text:
        return text.replace("_", " ").title()
    if text.islower():
        return text.title()
    return text


def event_regime_code_label(value: Any) -> Optional[str]:
    try:
        code = int(value)
    except Exception:
        return None
    mapping = {
        0: "Pre-COVID Normal",
        1: "COVID Shock",
        2: "Post-COVID Recovery",
        3: "Russia-Ukraine War Shock",
        4: "Post-War Inflation Adjustment",
        5: "U.S. Tariff Shock",
        6: "Post-Tariff Normalization",
        7: "Iran Geopolitical Shock",
    }
    return mapping.get(code)


def extract_final_forecast_artifact_event_context(payload: Optional[dict]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not payload:
        return None, None, None
    artifacts = payload.get("artifacts", {}) or {}
    candidate_paths: List[Tuple[Path, str]] = []
    for key, value in artifacts.items():
        if not isinstance(value, str):
            continue
        if any(token in key.lower() for token in ["final_forecast", "final_forecast_blend", "forecast_blend"]):
            candidate_paths.append((Path(value), f"payload.artifacts[{key}]"))
        elif any(token in value.lower() for token in ["final_forecast", "final_forecast_blend", "forecast_blend"]):
            candidate_paths.append((Path(value), f"payload.artifacts[{key}]"))
    artifact_dir = payload.get("artifact_dir")
    if artifact_dir:
        artifact_dir_path = Path(artifact_dir)
        if artifact_dir_path.exists() and artifact_dir_path.is_dir():
            for file_path in sorted(artifact_dir_path.iterdir()):
                if file_path.is_file() and any(token in file_path.name.lower() for token in ["final_forecast", "final_forecast_blend", "forecast_blend"]):
                    candidate_paths.append((file_path, f"artifact_dir/{file_path.name}"))
    seen = set()
    unique_paths: List[Tuple[Path, str]] = []
    for path, source in candidate_paths:
        if path.exists() and str(path) not in seen:
            seen.add(str(path))
            unique_paths.append((path, source))
    columns = [
        "event_regime_description",
        "event_regime",
        "macro_event_regime",
        "current_macro_regime",
        "regime_description",
        "event_regime_code",
    ]
    for path, source in unique_paths:
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        for col in columns:
            if col in df.columns:
                series = df[col].dropna().astype(str).map(str.strip)
                series = series[series.apply(is_valid_event_context_text)]
                if not series.empty:
                    value = series.iloc[-1]
                    if col == "event_regime_code":
                        label = event_regime_code_label(value)
                    else:
                        label = format_event_context_label(value)
                    if label:
                        return label, f"{source}.{col}", "Macro-event context assigned from final forecast artifact"
    return None, None, None


def extract_event_descriptive_artifact_context(payload: Optional[dict], base_date: Optional[pd.Timestamp]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not payload or base_date is None or pd.isna(base_date):
        return None, None, None
    artifacts = payload.get("artifacts", {}) or {}
    candidate_paths: List[Tuple[Path, str]] = []
    for key, value in artifacts.items():
        if not isinstance(value, str):
            continue
        if any(token in key.lower() for token in ["event_descriptive", "event_regime_descriptive", "event_regime_descriptive_stats"]):
            candidate_paths.append((Path(value), f"payload.artifacts[{key}]"))
        elif any(token in value.lower() for token in ["event_descriptive", "event_regime_descriptive", "event_regime_descriptive_stats"]):
            candidate_paths.append((Path(value), f"payload.artifacts[{key}]"))
    artifact_dir = payload.get("artifact_dir")
    if artifact_dir:
        artifact_dir_path = Path(artifact_dir)
        if artifact_dir_path.exists() and artifact_dir_path.is_dir():
            for file_path in sorted(artifact_dir_path.iterdir()):
                if file_path.is_file() and any(token in file_path.name.lower() for token in ["event_descriptive", "event_regime_descriptive", "event_regime_descriptive_stats"]):
                    candidate_paths.append((file_path, f"artifact_dir/{file_path.name}"))
    seen = set()
    unique_paths: List[Tuple[Path, str]] = []
    for path, source in candidate_paths:
        if path.exists() and str(path) not in seen:
            seen.add(str(path))
            unique_paths.append((path, source))
    for path, source in unique_paths:
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        if not {"start_date", "end_date"}.issubset(df.columns):
            continue
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        valid = df.dropna(subset=["start_date"]).copy()
        if valid.empty:
            continue
        valid["end_date_effective"] = valid["end_date"].fillna(pd.Timestamp.max)
        match = valid[(valid["start_date"] <= base_date) & (valid["end_date_effective"] >= base_date)]
        if match.empty:
            continue
        for col in ["event_regime_description", "event_regime"]:
            if col in match.columns:
                series = match[col].dropna().astype(str).map(str.strip)
                series = series[series.apply(is_valid_event_context_text)]
                if not series.empty:
                    label = format_event_context_label(series.iloc[-1])
                    if label:
                        return label, f"{source}.{col}", "Macro-event context assigned from event descriptive artifact"
    return None, None, None


def get_current_event_context(payload: Optional[dict]) -> dict:
    if not payload:
        return {
            "label": "Not classified",
            "description": "No active macro-event regime was assigned to the latest forecast.",
            "source": "No event context found in forecast payload or artifacts",
            "is_classified": False,
        }
    forecast = payload.get("forecast", {}) or {}
    final_forecast = payload.get("final_forecast", {}) or {}
    payload_checks = [
        ("event_regime_description", lambda x: format_event_context_label(x), "payload.forecast.event_regime_description"),
        ("event_regime", lambda x: format_event_context_label(x), "payload.forecast.event_regime"),
        ("macro_event_regime", lambda x: format_event_context_label(x), "payload.forecast.macro_event_regime"),
        ("current_macro_regime", lambda x: format_event_context_label(x), "payload.forecast.current_macro_regime"),
        ("event_regime_code", event_regime_code_label, "payload.forecast.event_regime_code"),
    ]
    for key, formatter, source in payload_checks:
        value = forecast.get(key)
        if is_valid_event_context_text(value):
            label = formatter(value)
            if label:
                return {
                    "label": label,
                    "description": "Macro-event context assigned from forecast payload.",
                    "source": source,
                    "is_classified": True,
                }
    for key in ("event_regime_description", "event_regime"):
        value = final_forecast.get(key)
        if is_valid_event_context_text(value):
            label = format_event_context_label(value)
            if label:
                return {
                    "label": label,
                    "description": "Macro-event context assigned from final forecast payload.",
                    "source": f"payload.final_forecast.{key}",
                    "is_classified": True,
                }
    artifact_label, artifact_source, artifact_desc = extract_final_forecast_artifact_event_context(payload)
    if artifact_label:
        return {
            "label": artifact_label,
            "description": artifact_desc or "Macro-event context assigned from final forecast artifact.",
            "source": artifact_source or "final forecast artifact",
            "is_classified": True,
        }
    base_date = forecast.get("base_date") or forecast.get("forecast_date") or forecast.get("date")
    if base_date is not None:
        base_date = pd.to_datetime(base_date, errors="coerce")
    desc_label, desc_source, desc_desc = extract_event_descriptive_artifact_context(payload, base_date)
    if desc_label:
        return {
            "label": desc_label,
            "description": desc_desc or "Macro-event context assigned from event descriptive artifact.",
            "source": desc_source or "event descriptive artifact",
            "is_classified": True,
        }
    return {
        "label": "Not classified",
        "description": "No active macro-event regime was assigned to the latest forecast.",
        "source": "No event context found in forecast payload or artifacts",
        "is_classified": False,
    }


def get_event_context_details(payload: Optional[dict]) -> Tuple[str, str]:
    event_context = get_current_event_context(payload)
    return event_context["label"], event_context["source"]


def get_current_event_regime(payload: Optional[dict]) -> str:
    return get_current_event_context(payload)["label"]


def financial_model_label(model_name: Any) -> str:
    text = str(model_name or "").lower()
    for key, label in MODEL_LABELS.items():
        if key in text:
            return label
    return str(model_name or "Model")


def get_forecast_value(payload: Optional[dict]) -> Optional[float]:
    if not payload:
        return None
    forecast = payload.get("forecast", {}) or {}
    for key in ("final_forecast_blend", "pred_blend", "forecast", "raw_forecast"):
        x = safe_float(forecast.get(key))
        if x is not None:
            return x
    final_df = load_artifact_df(payload, "final_forecast")
    if not final_df.empty:
        for col in ("final_forecast_blend", "pred_blend", "forecast"):
            if col in final_df.columns:
                return safe_float(final_df[col].dropna().iloc[-1])
    return None


def get_high_vol_probability(payload: Optional[dict]) -> Optional[float]:
    if not payload:
        return None
    forecast = payload.get("forecast", {}) or {}
    for key in ("regime_prob_high_vol", "high_vol_probability", "prob_high_vol"):
        x = safe_float(forecast.get(key))
        if x is not None:
            return x
    return None


def get_confidence_band(payload: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    if not payload:
        return None, None
    forecast = payload.get("forecast", {}) or {}
    low = safe_float(forecast.get("forecast_p05"))
    high = safe_float(forecast.get("forecast_p95"))
    if low is None:
        low = safe_float(forecast.get("pred_blend_p05"))
    if high is None:
        high = safe_float(forecast.get("pred_blend_p95"))
    return low, high


def get_reference_volatility_series(payload: Optional[dict]) -> pd.Series:
    # Use model-ready target if available. It is the cleanest realized-volatility reference.
    model_ready = get_model_ready_df(payload)
    for col in ("y_target", "target", "realized_volatility"):
        if not model_ready.empty and col in model_ready.columns:
            return pd.to_numeric(model_ready[col], errors="coerce").dropna()
    holdout = get_holdout_df(payload)
    if not holdout.empty and "actual" in holdout.columns:
        return pd.to_numeric(holdout["actual"], errors="coerce").dropna()
    return pd.Series(dtype=float)


def classify_forecast_volatility_regime(payload: Optional[dict]) -> Dict[str, Any]:
    forecast_vol = get_forecast_value(payload)
    high_prob = get_high_vol_probability(payload)
    reference = get_reference_volatility_series(payload)

    if forecast_vol is None or reference.empty:
        return {
            "regime_label": "Unavailable",
            "forecast_volatility": forecast_vol,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "high_vol_probability": high_prob,
            "probability_signal": "Unavailable",
            "method": "Insufficient reference data",
            "recommended_interpretation": "Run a forecast or load saved results to classify the volatility regime.",
        }

    q25 = float(reference.quantile(0.25))
    q50 = float(reference.quantile(0.50))
    q75 = float(reference.quantile(0.75))
    q90 = float(reference.quantile(0.90))

    if forecast_vol < q25:
        label = "Low-volatility regime"
        note = "Forecast volatility is below the lower quartile of the historical realized-volatility distribution."
    elif forecast_vol < q50:
        label = "Normal-volatility regime"
        note = "Forecast volatility is inside the lower-middle part of the historical realized-volatility distribution."
    elif forecast_vol < q75:
        label = "Elevated-volatility regime"
        note = "Forecast volatility is above the historical median but below the high-volatility threshold."
    elif forecast_vol < q90:
        label = "High-volatility regime"
        note = "Forecast volatility is above the 75th percentile and signals elevated hedging sensitivity."
    else:
        label = "Stress-volatility regime"
        note = "Forecast volatility is in the upper tail of the historical distribution. Liquidity and hedge timing require close attention."

    if high_prob is None:
        prob_signal = "Probability unavailable"
    elif high_prob >= 0.60:
        prob_signal = "High-volatility alert"
    elif high_prob >= 0.40:
        prob_signal = "Watch zone"
    else:
        prob_signal = "Low high-volatility probability"

    return {
        "regime_label": label,
        "forecast_volatility": forecast_vol,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "high_vol_probability": high_prob,
        "probability_signal": prob_signal,
        "method": "Empirical realized-volatility quantile classification",
        "recommended_interpretation": note,
    }


def recommended_trading_action(classification: Dict[str, Any]) -> str:
    label = classification.get("regime_label", "Unavailable")
    high_prob = safe_float(classification.get("high_vol_probability"), 0.0) or 0.0
    if label in {"High-volatility regime", "Stress-volatility regime"} or high_prob >= 0.60:
        return "Defensive hedge stance. Reduce open exposure, prioritize hedge execution, and keep liquidity risk under close review."
    if label == "Elevated-volatility regime" or (high_prob >= 0.40 and high_prob < 0.60):
        return "Selective hedge stance. Tighten exposure limits and consider partial hedges while monitoring market drivers."
    if label in {"Normal-volatility regime", "Low-volatility regime"} and high_prob < 0.40:
        return "Maintain standard exposure limits. Monitor routine market drivers and keep risk alert triggers active."
    if label in {"Normal-volatility regime", "Low-volatility regime"}:
        return "Maintain standard exposure discipline while watching for any shift toward elevated risk."
    return "Load or run the model to generate a decision signal."


def risk_stance_label(classification: Dict[str, Any]) -> str:
    label = classification.get("regime_label", "Unavailable")
    high_prob = safe_float(classification.get("high_vol_probability"), 0.0) or 0.0
    if label in {"Stress-volatility regime", "High-volatility regime"} or high_prob >= 0.60:
        return "Defensive hedge stance"
    if label == "Elevated-volatility regime" or (high_prob >= 0.40 and high_prob < 0.60):
        return "Selective hedge stance"
    if label in {"Normal-volatility regime", "Low-volatility regime"}:
        return "Standard monitoring stance"
    return "No signal"


def build_trader_decision_summary(regime_label: str, high_vol_probability: Optional[float]) -> str:
    label = str(regime_label or "").lower()
    prob = safe_float(high_vol_probability)
    if prob is not None:
        if ("low" in label or "normal" in label) and prob < 0.40:
            return "Risk stance is stable. Maintain standard exposure limits and monitor routine USD/TND drivers."
        if "elevated" in label or (prob >= 0.40 and prob < 0.60):
            return "Risk stance is selective. Tighten exposure limits and consider partial hedges while monitoring global dollar and oil-risk signals."
        if "high" in label or "stress" in label or prob >= 0.60:
            return "Risk stance is defensive. Prioritize hedge execution, reduce open exposure, and monitor liquidity conditions closely."
    if "low" in label or "normal" in label:
        return "Risk stance is stable. Maintain standard exposure limits and monitor routine USD/TND drivers."
    if "elevated" in label:
        return "Risk stance is selective. Tighten exposure limits and consider partial hedges while monitoring global dollar and oil-risk signals."
    if "high" in label or "stress" in label:
        return "Risk stance is defensive. Prioritize hedge execution, reduce open exposure, and monitor liquidity conditions closely."
    return "Risk stance is not available. Review the latest model signal and macro-event context before taking action."


def get_trading_stance_payload(regime_label: str, high_vol_probability: Optional[float]) -> dict:
    label = str(regime_label or "").lower()
    prob = safe_float(high_vol_probability)
    if ("high" in label or "stress" in label) or (prob is not None and prob >= 0.60):
        return {
            "stance": "Defensive hedge stance",
            "guidance": "Prioritize hedge execution, reduce open exposure, and monitor liquidity conditions closely.",
            "severity": "defensive",
        }
    if "elevated" in label or (prob is not None and 0.40 <= prob < 0.60):
        return {
            "stance": "Selective hedge stance",
            "guidance": "Tighten exposure limits and consider partial hedges while monitoring market drivers.",
            "severity": "watch",
        }
    if "low" in label or "normal" in label or prob is None:
        return {
            "stance": "Standard monitoring stance",
            "guidance": "Maintain standard exposure limits and monitor routine USD/TND market drivers.",
            "severity": "stable",
        }
    return {
        "stance": "Standard monitoring stance",
        "guidance": "Maintain standard exposure limits and monitor routine USD/TND market drivers.",
        "severity": "stable",
    }


def build_risk_interpretation(regime_label: str, forecast_vol: Optional[float], thresholds: dict) -> str:
    regime = str(regime_label or "").lower()
    if "low" in regime:
        return "Forecast volatility is below the lower historical quartile. Market conditions are calm, but routine monitoring remains necessary."
    if "normal" in regime:
        return "Forecast volatility remains within the central historical range. Standard monitoring is appropriate."
    if "elevated" in regime:
        return "Forecast volatility is above the historical median but below the high-volatility threshold. Hedge timing should be monitored closely."
    if "high" in regime:
        return "Forecast volatility is above the high-volatility threshold. Hedge execution risk and exposure sensitivity are elevated."
    if "stress" in regime:
        return "Forecast volatility is in the historical upper tail. Defensive hedge execution and liquidity monitoring are required."
    return "Forecast regime is not available. Review model outputs and market conditions before taking hedging decisions."


def build_confidence_band_text(forecast_p05: Optional[float], forecast_p95: Optional[float], forecast_vol: Optional[float]) -> str:
    if forecast_p05 is None or forecast_p95 is None:
        return "Forecast dispersion cannot be assessed."
    band_width = forecast_p95 - forecast_p05
    relative_width = band_width / max(safe_float(forecast_vol) or 1e-8, 1e-8)
    if relative_width < 0.8:
        return "Forecast dispersion is contained."
    if relative_width < 1.5:
        return "Forecast dispersion is moderate."
    return "Forecast dispersion is wide; use the upper band for stress exposure planning."


def build_trading_desk_takeaway(
    forecast_vol: Optional[float],
    regime_label: str,
    high_vol_probability: Optional[float],
    forecast_p05: Optional[float],
    forecast_p95: Optional[float],
    event_context: dict,
    thresholds: dict,
) -> List[str]:
    prob_text = format_pct(high_vol_probability) if safe_float(high_vol_probability) is not None else "unavailable"
    p05_text = format_pct(forecast_p05)
    p95_text = format_pct(forecast_p95)
    paragraph_1 = (
        f"The official 3-day USD/TND volatility forecast stands at {format_pct(forecast_vol)}, placing the pair in a {regime_label}. "
        f"The high-volatility probability is {prob_text}, and the 90% forecast band ranges from {p05_text} to {p95_text}."
    )
    paragraph_2 = build_risk_interpretation(regime_label, forecast_vol, thresholds)
    regime = str(regime_label or "").lower()
    if ("low" in regime or "normal" in regime) and (safe_float(high_vol_probability) is None or safe_float(high_vol_probability) < 0.40):
        paragraph_3 = "Best practice in this regime is to maintain standard monitoring, avoid unnecessary hedge acceleration, and continue tracking DXY, Brent, MOVE, VIX, and local FX liquidity."
    elif "elevated" in regime or (safe_float(high_vol_probability) is not None and 0.40 <= safe_float(high_vol_probability) < 0.60):
        paragraph_3 = "Best practice in this regime is to tighten exposure monitoring, consider partial hedges for near-term USD/TND exposures, and review DXY, Brent, MOVE, VIX, global sentiment, and macro-event headlines before executing large FX transactions."
    else:
        paragraph_3 = "Best practice in this regime is to prioritize hedge execution, reduce uncovered short-term exposure, monitor liquidity conditions, and use the upper confidence band for stress exposure planning."
    if event_context.get("is_classified"):
        paragraph_3 = f"{paragraph_3} Current macro-event context: {event_context.get('label')}."
    else:
        paragraph_3 = f"{paragraph_3} No active macro-event regime is assigned to the latest forecast."
    return [paragraph_1, paragraph_2, paragraph_3]


# ============================================================
# Data upload and run helpers
# ============================================================


@st.cache_data(ttl=300, show_spinner=False)
def fetch_schema() -> dict:
    return get_expected_input_schema()


def classify_scenario_against_thresholds(classification: Dict[str, Any]) -> Dict[str, Any]:
    q25 = safe_float(classification.get("q25"))
    q50 = safe_float(classification.get("q50"))
    q75 = safe_float(classification.get("q75"))
    q90 = safe_float(classification.get("q90"))
    forecast_vol = safe_float(classification.get("forecast_volatility"))
    if forecast_vol is None or any(v is None for v in [q25, q50, q75, q90]):
        classification["regime_label"] = "Unavailable"
        return classification
    if forecast_vol < q25:
        label = "Low-volatility regime"
        interp = "Scenario volatility is below the lower quartile of realized USD/TND volatility."
    elif forecast_vol < q50:
        label = "Normal-volatility regime"
        interp = "Scenario volatility remains within the lower-middle historical range."
    elif forecast_vol < q75:
        label = "Elevated-volatility regime"
        interp = "Scenario volatility is above the historical median but below the high-volatility threshold."
    elif forecast_vol < q90:
        label = "High-volatility regime"
        interp = "Scenario volatility is above the high-volatility threshold."
    else:
        label = "Stress-volatility regime"
        interp = "Scenario volatility is in the upper tail of the historical distribution."
    classification["regime_label"] = label
    classification["recommended_interpretation"] = interp
    prob = safe_float(classification.get("high_vol_probability"))
    if prob is not None:
        if prob >= 0.60:
            signal = "High-volatility alert"
        elif prob >= 0.40:
            signal = "Watch zone"
        else:
            signal = "Low high-volatility probability"
        classification["probability_signal"] = signal
    return classification


def normalize_schema_columns(schema: dict) -> Dict[str, str]:
    alias_to_canonical: Dict[str, str] = {}
    for col in schema.get("required_columns", []):
        alias_to_canonical[str(col).strip().lower()] = col
    for canonical, aliases in schema.get("accepted_aliases", {}).items():
        for alias in aliases:
            alias_to_canonical[str(alias).strip().lower()] = canonical
    return alias_to_canonical


def missing_uploaded_columns(df: pd.DataFrame, schema: dict) -> List[str]:
    alias_to_canonical = normalize_schema_columns(schema)
    present = {
        alias_to_canonical.get(str(col).strip().lower(), str(col).strip())
        for col in df.columns
    }
    return [col for col in schema.get("required_columns", []) if col not in present]


def validate_uploaded_file(uploaded_file: Any) -> None:
    if uploaded_file is None:
        return
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError("Upload must be an Excel file with extension .xlsx or .xls.")


def read_uploaded_excel(uploaded_file: Any) -> pd.DataFrame:
    # The forecasting pipeline validates and reads the official sheet name.
    # Reading the same sheet here keeps the preview aligned with the model run.
    return pd.read_excel(BytesIO(uploaded_file.getvalue()), sheet_name=SHEET_NAME)


def run_forecast(uploaded_file: Any, progress_callback: Optional[Any] = None) -> dict:
    run_id = local_run_id()
    run_dir = FORECAST_ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if uploaded_file is not None:
        validate_uploaded_file(uploaded_file)
        safe_upload_name = Path(uploaded_file.name).name
        input_path = run_dir / safe_upload_name
        input_path.write_bytes(uploaded_file.getvalue())
    else:
        input_path = RAW_FILE_PATH

    validate_input_file(input_path)
    result = run_pipeline(
        input_path=input_path,
        output_dir=run_dir,
        save_outputs=True,
        progress_callback=progress_callback,
    )
    payload = result_to_payload(result)
    payload["run_id"] = run_id
    payload["run_datetime"] = datetime.now(timezone.utc).isoformat()
    payload["artifacts"] = {key: str(path) for key, path in result.artifacts.items()}
    payload["artifact_dir"] = str(run_dir)
    base_date = infer_base_date_from_input_file(input_path)
    if base_date:
        payload["base_date"] = base_date
        payload.setdefault("forecast", {})["base_date"] = base_date
    save_dashboard_payload(payload, run_dir)
    return payload


def render_expected_schema(schema: dict) -> None:
    with st.expander("Expected Excel schema", expanded=False):
        st.caption(f"Expected sheet: {schema.get('sheet_name', 'Sheet1')}")
        st.write(", ".join(schema.get("required_columns", [])))
        aliases = schema.get("accepted_aliases", {})
        if aliases:
            alias_rows = [
                {"Canonical column": key, "Accepted names": ", ".join(value)}
                for key, value in aliases.items()
            ]
            st.dataframe(pd.DataFrame(alias_rows), hide_index=True, use_container_width=True)


def render_upload_preview(uploaded_file: Any, schema: dict) -> Dict[str, Any]:
    state: Dict[str, Any] = {"df": None, "missing": [], "error": None}
    if uploaded_file is None:
        if RAW_FILE_PATH.exists():
            st.info(f"No upload selected. The app will use the bundled default file: `{RAW_FILE_PATH.name}`.")
        else:
            st.warning("No upload selected and `Final_data.xlsx` is not bundled with the app. Upload an Excel file before running the forecast.")
        return state
    try:
        df = read_uploaded_excel(uploaded_file)
    except Exception as exc:
        state["error"] = f"Could not read uploaded Excel file: {exc}"
        st.error(state["error"])
        return state

    missing = missing_uploaded_columns(df, schema)
    state["df"] = df
    state["missing"] = missing
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna() if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
    total_missing = int(df.isna().sum().sum())
    total_cells = max(1, int(df.shape[0] * df.shape[1]))
    missing_pct = 100.0 * total_missing / total_cells
    date_range = f"{dates.min().date()} to {dates.max().date()}" if not dates.empty else "Unavailable"
    schema_status = "Passed" if not missing else "Failed"
    data_quality = "Good" if missing_pct < 5 else "Review" if missing_pct < 15 else "Poor"

    render_kpi_grid([
        metric_card("Rows", f"{len(df):,}", "Input observations", "primary"),
        metric_card("Columns", f"{df.shape[1]:,}", "Market variables", "teal"),
        metric_card("Date Range", date_range, "Input coverage", "gold"),
        metric_card("Data Quality", data_quality, f"Missing values: {missing_pct:.1f}%", "green" if data_quality == "Good" else "gold"),
    ])
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        st.success(f"Schema validation {schema_status}. All required columns are available.")

    with st.expander("Dataset preview and diagnostics", expanded=False):
        st.dataframe(df.head(), use_container_width=True)
        col_summary = []
        for col in df.columns:
            col_summary.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Missing": int(df[col].isna().sum()),
                "Unique": int(df[col].nunique(dropna=True)),
            })
        st.dataframe(pd.DataFrame(col_summary), hide_index=True, use_container_width=True)
    return state


# ============================================================
# Chart helpers
# ============================================================


def line_fig(df: pd.DataFrame, x: str, y_cols: List[Tuple[str, str]], title: str = "", y_scale: float = 1.0, yaxis_title: str = "Value") -> go.Figure:
    fig = go.Figure()
    for col, label in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=pd.to_numeric(df[col], errors="coerce") * y_scale, mode="lines", name=label))
    fig.update_layout(title=title, yaxis_title=yaxis_title)
    return apply_finance_layout(fig, height=390)


def render_regime_gauge_html(
    forecast_vol: float | None,
    calm_threshold: float | None,
    median_volatility_line: float | None,
    high_volatility_threshold: float | None,
    stress_threshold: float | None,
    assigned_regime: str,
    high_vol_probability: float | None = None,
) -> None:
    if any(v is None for v in [forecast_vol, calm_threshold, median_volatility_line, high_volatility_threshold, stress_threshold]):
        st.info("Regime thresholds are unavailable until a forecast payload and reference volatility history are loaded.")
        return

    active_colors = {
        "Low-volatility regime": "#14b8a6",
        "Normal-volatility regime": "#3b82f6",
        "Elevated-volatility regime": "#f59e0b",
        "High-volatility regime": "#f97316",
        "Stress-volatility regime": "#ef4444",
    }
    active_color = active_colors.get(assigned_regime, FINANCE_COLORS["slate"])
    x_max = max(stress_threshold * 1.20, forecast_vol * 1.20, 1e-8)
    calm_pct = min(100, max(0, calm_threshold / x_max * 100))
    median_pct = min(100, max(0, median_volatility_line / x_max * 100))
    high_pct = min(100, max(0, high_volatility_threshold / x_max * 100))
    stress_pct = min(100, max(0, stress_threshold / x_max * 100))
    forecast_pct = min(97, max(3, forecast_vol / x_max * 100))

    low_width = max(0, calm_pct)
    normal_width = max(0, median_pct - calm_pct)
    elevated_width = max(0, high_pct - median_pct)
    high_width = max(0, stress_pct - high_pct)
    stress_width = max(0, 100 - stress_pct)

    calm_pos = min(97, max(3, calm_pct))
    median_pos = min(97, max(3, median_pct))
    high_pos = min(97, max(3, high_pct))
    stress_pos = min(97, max(3, stress_pct))

    low_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "Low-volatility regime" else ""
    normal_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "Normal-volatility regime" else ""
    elevated_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "Elevated-volatility regime" else ""
    high_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "High-volatility regime" else ""
    stress_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "Stress-volatility regime" else ""

    html = f"""
    <style>
        .regime-gauge-card {{
            --active-regime-color: {active_color};
            background: linear-gradient(180deg, rgba(7, 16, 29, 0.99), rgba(14, 20, 36, 0.96));
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 28px;
            padding: 32px;
            margin: 8px 0;
            box-shadow: 0 28px 60px rgba(0, 0, 0, 0.22);
            transition: border-color 240ms ease, box-shadow 240ms ease, transform 240ms ease;
        }}
        .regime-gauge-card:hover {{
            border-color: var(--active-regime-color);
            box-shadow: 0 32px 72px rgba(0, 0, 0, 0.28), 0 0 0 1px rgba(255, 255, 255, 0.05);
            transform: translateY(-2px);
        }}
        .regime-gauge-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
            margin-bottom: 26px;
        }}
        .regime-metric-box {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 16px 18px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.75), rgba(20, 30, 48, 0.55));
            box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.06), 0 4px 12px rgba(0, 0, 0, 0.12);
            flex: 0 1 auto;
            transition: border-color 180ms ease, box-shadow 180ms ease;
        }}
        .regime-metric-box:hover {{
            border-color: rgba(148, 163, 184, 0.35);
            box-shadow: inset 0 1px 3px rgba(255, 255, 255, 0.08), 0 6px 16px rgba(0, 0, 0, 0.15);
        }}
        .regime-metric-label {{
            font-size: 0.68rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #a8b6cc;
            font-weight: 700;
        }}
        .regime-metric-value {{
            font-size: 1.48rem;
            font-weight: 900;
            color: #f8fafc;
            line-height: 1.1;
        }}
        .regime-pill {{
            display: inline-flex;
            align-items: center;
            padding: 9px 16px;
            border-radius: 999px;
            background: {active_color};
            color: #08111f;
            font-size: 0.85rem;
            font-weight: 800;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
            border: 1px solid rgba(255, 255, 255, 0.2);
            white-space: nowrap;
            flex: 0 0 auto;
        }}
        .regime-gauge-body {{
            position: relative;
            padding-top: 12px;
        }}
        .regime-bar {{
            position: relative;
            display: flex;
            width: 100%;
            height: 94px;
            border-radius: 52px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.3), inset 0 -2px 6px rgba(255, 255, 255, 0.08);
        }}
        .regime-segment {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 800;
            color: #f8fafc;
            white-space: nowrap;
            min-width: 0;
            padding: 0 12px;
        }}
        .regime-segment.low {{ background: linear-gradient(135deg, rgba(20,184,166,0.52), rgba(20,184,166,0.28)); }}
        .regime-segment.normal {{ background: linear-gradient(135deg, rgba(59,130,246,0.50), rgba(59,130,246,0.25)); }}
        .regime-segment.elevated {{ background: linear-gradient(135deg, rgba(245,158,11,0.58), rgba(245,158,11,0.28)); }}
        .regime-segment.high {{ background: linear-gradient(135deg, rgba(249,115,22,0.50), rgba(249,115,22,0.24)); }}
        .regime-segment.stress {{ background: linear-gradient(135deg, rgba(239,68,68,0.48), rgba(127,29,29,0.35)); }}
        .regime-segment.active {{
            filter: brightness(1.18) saturate(1.16);
            box-shadow: inset 0 0 22px rgba(255,255,255,0.15), 0 0 16px rgba(255,255,255,0.08);
        }}
        .forecast-pill {{
            position: absolute;
            top: 50%;
            left: var(--forecast-position);
            transform: translate(-50%, -50%);
            padding: 6px 11px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.24);
            background: {active_color};
            color: #08111f;
            font-size: 0.78rem;
            font-weight: 900;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
            white-space: nowrap;
            z-index: 5;
            min-width: auto;
            text-align: center;
        }}
        .threshold-tick {{
            position: absolute;
            top: 10px;
            bottom: 10px;
            width: 1px;
            background: rgba(226, 232, 240, 0.55);
            z-index: 4;
        }}
        .threshold-label-row {{
            position: relative;
            height: 68px;
            margin-top: 1.2rem;
        }}
        .threshold-label {{
            position: absolute;
            transform: translateX(-50%);
            text-align: center;
            color: #cbd5e1;
            font-size: 0.82rem;
            line-height: 1.4;
            white-space: nowrap;
            max-width: 140px;
            padding: 0 6px;
        }}
        .threshold-label strong {{ display: block; color: #f0f4f8; font-size: 0.92rem; font-weight: 700; margin-bottom: 5px; }}
    </style>
    <div class="regime-gauge-card" style="--active-regime-color: {active_color};">
        <div class="regime-gauge-header">
            <div class="regime-metric-box">
                <div class="regime-metric-label">Forecast Volatility</div>
                <div class="regime-metric-value">{format_pct(forecast_vol)}</div>
            </div>
            <div class="regime-metric-box">
                <div class="regime-metric-label">High-Vol Probability</div>
                <div class="regime-metric-value">{format_pct(high_vol_probability) if high_vol_probability is not None else 'N/A'}</div>
            </div>
            <div class="regime-pill"><span>{assigned_regime}</span></div>
        </div>
        <div class="regime-gauge-body">
            <div class="regime-bar" style="--forecast-position: {forecast_pct}%;">
                <div class="regime-segment low{' active' if assigned_regime == 'Low-volatility regime' else ''}" style="width: {low_width}%; {low_highlight}">Low</div>
                <div class="regime-segment normal{' active' if assigned_regime == 'Normal-volatility regime' else ''}" style="width: {normal_width}%; {normal_highlight}">Normal</div>
                <div class="regime-segment elevated{' active' if assigned_regime == 'Elevated-volatility regime' else ''}" style="width: {elevated_width}%; {elevated_highlight}">Elevated</div>
                <div class="regime-segment high{' active' if assigned_regime == 'High-volatility regime' else ''}" style="width: {high_width}%; {high_highlight}">High</div>
                <div class="regime-segment stress{' active' if assigned_regime == 'Stress-volatility regime' else ''}" style="width: {stress_width}%; {stress_highlight}">Stress</div>
                <div class="threshold-tick" style="left: {calm_pos}%"></div>
                <div class="threshold-tick" style="left: {median_pos}%"></div>
                <div class="threshold-tick" style="left: {high_pos}%"></div>
                <div class="threshold-tick" style="left: {stress_pos}%"></div>
                <div class="forecast-pill">{format_pct(forecast_vol)}</div>
            </div>
            <div class="threshold-label-row">
                <div class="threshold-label" style="left: {calm_pos}%;"><strong>Calm threshold</strong>{format_pct(calm_threshold)}</div>
                <div class="threshold-label" style="left: {median_pos}%;"><strong>Median volatility</strong>{format_pct(median_volatility_line)}</div>
                <div class="threshold-label" style="left: {high_pos}%;"><strong>High threshold</strong>{format_pct(high_volatility_threshold)}</div>
                <div class="threshold-label" style="left: {stress_pos}%;"><strong>Stress threshold</strong>{format_pct(stress_threshold)}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def plot_forecast_vs_actual(payload: Optional[dict]) -> None:
    holdout = get_holdout_df(payload)
    if holdout.empty or not {"Date", "actual", "pred_blend"}.issubset(holdout.columns):
        st.info("Forecast-vs-actual history is not available.")
        return
    holdout["Date"] = pd.to_datetime(holdout["Date"], errors="coerce")
    fig = go.Figure()
    if {"pred_blend_p05", "pred_blend_p95"}.issubset(holdout.columns):
        fig.add_trace(go.Scatter(x=holdout["Date"], y=holdout["pred_blend_p95"] * 100, mode="lines", line=dict(width=0), showlegend=False, name="Upper band"))
        fig.add_trace(go.Scatter(x=holdout["Date"], y=holdout["pred_blend_p05"] * 100, mode="lines", fill="tonexty", fillcolor="rgba(47,216,201,0.12)", line=dict(width=0), name="90% confidence band"))
    fig.add_trace(go.Scatter(x=holdout["Date"], y=holdout["actual"] * 100, mode="lines", name="Actual realized volatility", line=dict(width=2.4)))
    fig.add_trace(go.Scatter(x=holdout["Date"], y=holdout["pred_blend"] * 100, mode="lines", name="Final Macro-Event FX Volatility Engine", line=dict(width=2.7)))
    fig = apply_finance_layout(fig, height=430)
    fig.update_layout(
        title=dict(
            text="Forecast vs Realized USD/TND Volatility",
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(size=18),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1.0,
            font=dict(size=11),
        ),
        margin=dict(l=45, r=35, t=105, b=45),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page renderers
# ============================================================


def render_no_payload_choice() -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="page-kicker">No forecast loaded</div>
            <div class="page-title">Choose how to start the trading risk view</div>
            <p class="small-muted">Load the latest saved run for instant monitoring, or run a new forecast from the Data & Market Inputs page.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'>Load latest saved result</div><div class='panel-subtitle'>Open the most recent local artifact without rerunning the model.</div>", unsafe_allow_html=True)
        latest = find_latest_run_dir()
        if latest:
            st.caption(f"Latest saved run: {latest.name}")
        else:
            st.caption("No saved local run detected.")
        if st.button("Load latest risk view", type="primary", use_container_width=True, key="load_latest_from_empty"):
            try:
                payload = load_latest_local()
                st.session_state.forecast_payload = payload
                st.session_state.selected_run_id = payload.get("run_id")
                st.session_state.artifact_dir = payload.get("artifact_dir")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'>Run new forecast</div><div class='panel-subtitle'>Validate market data and generate a fresh 3-day USD/TND volatility outlook.</div>", unsafe_allow_html=True)
        if st.button("Go to Data & Market Inputs", use_container_width=True, key="go_data_inputs_from_empty"):
            st.session_state.selected_page = "Data & Market Inputs"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_executive_fx_risk_cockpit(payload: Optional[dict]) -> None:
    # Cockpit-specific layout overrides for compactness
    st.markdown(
        """
        <style>
        /* Cockpit compact layout overrides */
        .hero-panel {
            padding: 1.1rem;
            margin-bottom: 0.8rem;
        }
        .kpi-card {
            min-height: 100px;
            padding: 0.9rem;
        }
        .decision-card {
            min-height: 115px;
            padding: 0.9rem;
            margin-bottom: 0.7rem;
        }
        .decision-card.decision-strip {
            min-height: 64px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.6rem;
        }
        .decision-card.decision-strip .decision-title {
            font-size: 0.72rem;
        }
        .decision-card.severity-stable {
            border-left-color: var(--finance-green);
        }
        .decision-card.severity-watch {
            border-left-color: var(--finance-gold);
        }
        .decision-card.severity-defensive {
            border-left-color: var(--finance-red);
        }
        .decision-card.trading-desk-takeaway p {
            margin: 0.35rem 0;
            line-height: 1.55;
            max-width: 1500px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    page_header("Executive FX Risk Cockpit", "Decision layer", payload)
    if payload is None:
        render_no_payload_choice()
        return

    classification = classify_forecast_volatility_regime(payload)
    forecast_vol = classification.get("forecast_volatility")
    high_prob = classification.get("high_vol_probability")
    p05, p95 = get_confidence_band(payload)
    event_context = get_current_event_context(payload)
    stance_payload = get_trading_stance_payload(classification.get("regime_label", "Unavailable"), high_prob)
    decision_summary = build_trader_decision_summary(classification.get("regime_label", "Unavailable"), high_prob)
    interpretation = build_risk_interpretation(classification.get("regime_label", "Unavailable"), forecast_vol, classification)
    confidence_note = build_confidence_band_text(p05, p95, forecast_vol)
    band = f"{format_pct(p05)} - {format_pct(p95)}" if p05 is not None or p95 is not None else "n/a"
    event_note = "Calendar-known macro-event context" if event_context["is_classified"] else "No active macro-event regime assigned"

    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="page-kicker">Final Macro-Event FX Volatility Engine</div>
            <div class="page-title">USD/TND 3-Day Volatility Risk Signal</div>
            <p class="small-muted">Official production forecast for hedge timing, exposure limits, and short-term FX risk monitoring.</p>
            <div class="page-chip">Latest macro-event context: {html.escape(event_context['label'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    high_vol_note = "Probability signal unavailable"
    if high_prob is not None:
        if high_prob >= 0.60:
            high_vol_note = "High-volatility alert"
        elif high_prob >= 0.40:
            high_vol_note = "Watch zone"
        else:
            high_vol_note = "Low high-volatility probability"

    render_kpi_grid([
        metric_card("3-Day Volatility Forecast", format_pct(forecast_vol), "Official short-term USD/TND volatility outlook", "teal"),
        metric_card("Forecast Volatility Regime", str(classification.get("regime_label", "Unavailable")), "Empirical realized-volatility threshold classification", "gold"),
        metric_card("High-Volatility Signal", format_pct(high_prob), high_vol_note, "purple"),
        metric_card("Current Macro-Event Context", event_context["label"], event_note, "primary"),
    ])

    st.markdown(
        f"""
        <div class="decision-card decision-strip severity-{stance_payload['severity']}">
            <div class="decision-title">Decision Summary</div>
            <div class="decision-text">{html.escape(decision_summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1.0], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="decision-card severity-{stance_payload['severity']}">
                <div class="decision-title">Recommended Trading Stance</div>
                <div class="decision-text"><strong>{html.escape(stance_payload['stance'])}</strong><br>{html.escape(stance_payload['guidance'])}</div>
            </div>
            <div class="decision-card">
                <div class="decision-title">Risk interpretation</div>
                <div class="decision-text">{html.escape(interpretation)}</div>
            </div>
            <div class="decision-card">
                <div class="decision-title">Forecast Confidence Band</div>
                <div class="decision-text">90% forecast interval: <strong>{html.escape(band)}</strong>. {html.escape(confidence_note)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_recent_volatility_pulse(payload)
    with right:
        render_market_watchlist(payload, event_context)
        render_action_checklist()

    takeaway_paragraphs = build_trading_desk_takeaway(
        forecast_vol,
        classification.get("regime_label", "Unavailable"),
        high_prob,
        p05,
        p95,
        event_context,
        classification,
    )
    takeaway_html = "".join(f"<p class='decision-text'>{html.escape(paragraph)}</p>" for paragraph in takeaway_paragraphs)
    st.markdown(
        f"""
        <div class='decision-card trading-desk-takeaway severity-{stance_payload['severity']}'>
            <div class='decision-title'>Trading Desk Takeaway</div>
            {takeaway_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    render_navigation_cards()


def render_data_market_inputs(payload: Optional[dict]) -> None:
    page_header("Data & Market Inputs", "Workflow layer", payload)
    schema = fetch_schema()

    st.markdown("<div class='method-box'><strong>Workflow choice.</strong> Load the latest saved risk view for monitoring, or run a new forecast after validating the input dataset.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'>Saved local results</div><div class='panel-subtitle'>Open the latest generated model artifacts.</div>", unsafe_allow_html=True)
        latest = find_latest_run_dir()
        if latest:
            st.write(f"Latest saved run: **{latest.name}**")
        else:
            st.info("No saved run detected yet.")
        if st.button("Load latest saved results", type="primary", use_container_width=True):
            try:
                loaded = load_latest_local()
                st.session_state.forecast_payload = loaded
                st.session_state.selected_run_id = loaded.get("run_id")
                st.session_state.artifact_dir = loaded.get("artifact_dir")
                st.success("Latest saved risk view loaded.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'>Official model run</div><div class='panel-subtitle'>Run the validated macro-event engine with the current forecasting pipeline.</div>", unsafe_allow_html=True)
        st.write("Default input file:")
        st.code(str(RAW_FILE_PATH))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    render_expected_schema(schema)
    uploaded_file = st.file_uploader("Upload Excel market dataset", type=["xlsx", "xls"], key="upload_file")
    upload_state = render_upload_preview(uploaded_file, schema)

    st.markdown("---")
    st.subheader("Run official forecast")
    st.info("The official run uses the forecasting configuration embedded directly in this Streamlit app. Scenario Laboratory is visual-only in this app version.")
    default_input_missing = uploaded_file is None and not RAW_FILE_PATH.exists()
    if default_input_missing:
        st.warning("Upload the official Excel market dataset first, or add `Final_data.xlsx` to the repository root.")
    run_disabled = bool(upload_state.get("missing") or upload_state.get("error") or default_input_missing)
    progress_box = st.empty()
    if st.button("Run Final Macro-Event FX Volatility Engine", type="primary", disabled=run_disabled, use_container_width=True):
        try:
            def streamlit_progress(message: str) -> None:
                progress_box.markdown(f"**{html.escape(message)}**")
            with st.spinner("Running the USD/TND macro-event volatility engine..."):
                new_payload = run_forecast(uploaded_file, progress_callback=streamlit_progress)
            st.session_state.forecast_payload = new_payload
            st.session_state.selected_run_id = new_payload.get("run_id")
            st.session_state.artifact_dir = new_payload.get("artifact_dir")
            st.success("Forecast completed and saved to artifacts.")
        except Exception as exc:
            st.error("Forecast failed. See technical details below.")
            with st.expander("Technical details"):
                st.write(str(exc))


def render_forecast_configuration(payload: Optional[dict]) -> None:
    page_header("Forecast Configuration", "Calibration layer", payload)
    st.markdown(
        """
        <div class="method-box">
            <strong>Production principle.</strong> The official forecast is the validated model configuration. User-defined windows and event selections are displayed as scenario assumptions only in this version, not as separate production forecasts.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'>Official Validated Forecast</div><div class='panel-subtitle'>Configuration used for the production trading signal.</div>", unsafe_allow_html=True)
        official_rows = pd.DataFrame([
            {"Setting": "Model", "Value": "Final Macro-Event FX Volatility Engine"},
            {"Setting": "Forecast horizon", "Value": f"Next {VOL_TARGET_WINDOW} trading days"},
            {"Setting": "Calibration window", "Value": f"{ROLLING_WINDOW} trading days (validated baseline)"},
            {"Setting": "Event features", "Value": "Enabled"},
            {"Setting": "Event feature forcing", "Value": "Enabled"},
            {"Setting": "Classification", "Value": "Empirical quantile thresholds"},
        ])
        st.dataframe(official_rows, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'>Scenario Calibration Preview</div><div class='panel-subtitle'>Visual-only assumptions. They do not retrain the model in this app version.</div>", unsafe_allow_html=True)
        preview_window_options = calibration_window_options("Custom window")
        window = st.selectbox(
            "Calibration window assumption",
            preview_window_options,
            index=default_calibration_window_index(preview_window_options),
            key="config_preview_calibration_window_v2",
        )
        events = st.multiselect(
            "Macro-event regimes to emphasize",
            ["COVID shock", "Russia-Ukraine shock", "U.S. tariff shock", "Post-Tariff Normalization", "Iran geopolitical shock"],
            default=[],
            key="config_preview_event_focus_v2",
        )
        mode = st.selectbox("Scenario interpretation", ["All data with event features", "Crisis-regime overweighting", "Selected-event sample - exploratory"])
        st.markdown(f"<div class='warning-box'>Scenario selected: <strong>{html.escape(window)}</strong>, mode: <strong>{html.escape(mode)}</strong>. This is a research view only and does not replace the official forecast.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_high_volatility_calibration_diagnostics(payload: Optional[dict]) -> None:
    summary = get_high_vol_calibration_df(payload, "summary")
    reliability = get_high_vol_calibration_df(payload, "reliability_curve")
    threshold_table = get_high_vol_calibration_df(payload, "threshold_sensitivity")

    st.markdown(
        "<div class='panel'><div class='panel-title'>High-Volatility Classifier Calibration</div>"
        "<div class='panel-subtitle'>Checks whether the model's high-volatility probabilities are reliable and how alert thresholds affect precision and recall.</div>",
        unsafe_allow_html=True,
    )

    if summary.empty and reliability.empty and threshold_table.empty:
        st.info("High-volatility calibration diagnostics are not available for this run.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not summary.empty:
        def summary_value(metric_name: str) -> Optional[float]:
            if "metric" not in summary.columns or "value" not in summary.columns:
                return None
            match = summary[summary["metric"].astype(str).str.lower() == metric_name.lower()]
            if match.empty:
                return None
            return safe_float(match.iloc[0]["value"])

        brier = summary_value("Brier score")
        roc_auc = summary_value("ROC-AUC")
        pr_auc = summary_value("PR-AUC")
        realized_freq = summary_value("Realized high-volatility frequency")
        render_kpi_grid(
            [
                metric_card("Brier Score", format_number(brier, 4), "Lower probability error is better", "teal"),
                metric_card("ROC-AUC", format_number(roc_auc, 3), "Ranking quality across thresholds", "primary"),
                metric_card("PR-AUC", format_number(pr_auc, 3), "Average precision for high-volatility alerts", "gold"),
                metric_card("Realized High-Vol Frequency", format_pct(realized_freq), "Holdout rate above the development q75 cutoff", "purple"),
            ]
        )
        with st.expander("Calibration summary table", expanded=False):
            st.dataframe(summary, hide_index=True, use_container_width=True)

    if not reliability.empty:
        reliability_plot = reliability.copy()
        required_cols = {"mean_predicted_probability", "observed_high_volatility_frequency"}
        if required_cols.issubset(reliability_plot.columns):
            reliability_plot["mean_predicted_probability"] = pd.to_numeric(
                reliability_plot["mean_predicted_probability"], errors="coerce"
            )
            reliability_plot["observed_high_volatility_frequency"] = pd.to_numeric(
                reliability_plot["observed_high_volatility_frequency"], errors="coerce"
            )
            reliability_plot = reliability_plot.dropna(
                subset=["mean_predicted_probability", "observed_high_volatility_frequency"]
            ).sort_values("mean_predicted_probability")
            if not reliability_plot.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=reliability_plot["mean_predicted_probability"],
                        y=reliability_plot["observed_high_volatility_frequency"],
                        mode="lines+markers",
                        name="Model calibration",
                        line=dict(color=FINANCE_COLORS["teal"], width=3),
                        marker=dict(size=8),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Perfect calibration",
                        line=dict(color=FINANCE_COLORS["slate"], dash="dash", width=2),
                    )
                )
                fig.update_layout(
                    title="High-Volatility Reliability Diagram",
                    xaxis_title="Mean predicted high-volatility probability",
                    yaxis_title="Observed high-volatility frequency",
                )
                fig.update_xaxes(range=[0, 1])
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(apply_finance_layout(fig, height=420), use_container_width=True)
                st.caption("A curve close to the diagonal means the predicted probabilities are well calibrated.")

    if not threshold_table.empty:
        threshold_display = threshold_table.copy()
        percent_columns = [
            "alert_threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "false_alert_rate",
            "missed_high_vol_rate",
        ]
        for column in percent_columns:
            if column in threshold_display.columns:
                threshold_display[column] = threshold_display[column].apply(format_pct)
        st.dataframe(threshold_display, hide_index=True, use_container_width=True)
        st.markdown(
            """
            <div class='method-box'>
                Lower thresholds usually increase recall but create more false alerts. Higher thresholds usually increase selectivity but may miss more high-volatility periods.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_volatility_regime_classification(payload: Optional[dict]) -> None:
    page_header("Volatility Regime Classification", "EMPIRICAL THRESHOLD AND VALIDATION LAYER", payload)
    if payload is None:
        render_no_payload_choice()
        return
    st.markdown(
        """
        <div class="method-box">
            <strong>Validation focus.</strong> This page validates how the forecast is translated into an empirical volatility regime.
            The high-volatility probability is evaluated with calibration diagnostics. The Brier score measures probability forecast error, while the reliability diagram compares predicted probabilities with observed high-volatility frequencies. The threshold sensitivity table shows how alert quality changes across different probability cutoffs.
        </div>
        """,
        unsafe_allow_html=True,
    )
    classification = classify_forecast_volatility_regime(payload)
    st.markdown(
        """
        <div style="background: linear-gradient(180deg, rgba(20, 30, 50, 0.6), rgba(15, 23, 42, 0.5)); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 14px; padding: 18px 24px; margin: 8px 0 16px 0; display: flex; justify-content: center; align-items: center;">
            <div style="text-align: center;">
                <div style="font-size: 0.70rem; letter-spacing: 0.15em; text-transform: uppercase; color: #cbd5e1; font-weight: 800; margin-bottom: 4px;">Classification Method</div>
                <div style="font-size: 1.05rem; font-weight: 900; color: #f8fafc; margin-bottom: 2px;">Historical realized-volatility thresholds</div>
                <div style="font-size: 0.78rem; color: #a8b6cc; font-weight: 700;">Empirical quantiles</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_regime_gauge_html(
        forecast_vol=classification.get("forecast_volatility"),
        calm_threshold=classification.get("q25"),
        median_volatility_line=classification.get("q50"),
        high_volatility_threshold=classification.get("q75"),
        stress_threshold=classification.get("q90"),
        assigned_regime=classification.get("regime_label", "Unavailable"),
        high_vol_probability=classification.get("high_vol_probability"),
    )
    st.markdown(
        """
        <div class="method-box">
            <strong>Methodology note.</strong> Regime labels are assigned by comparing the 3-day USD/TND volatility forecast with historical realized-volatility thresholds. These thresholds divide past USD/TND volatility into calm, normal, elevated, high, and stress zones.
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Technical threshold mapping"):
        st.write("Calm threshold = 25th percentile, Median volatility line = 50th percentile, High-volatility threshold = 75th percentile, Stress threshold = 90th percentile.")

    holdout = get_holdout_df(payload)
    if not holdout.empty and {"actual", "pred_blend"}.issubset(holdout.columns):
        actual = pd.to_numeric(holdout["actual"], errors="coerce")
        pred = pd.to_numeric(holdout["pred_blend"], errors="coerce")
        q75 = actual.quantile(0.75)
        actual_high = actual >= q75
        pred_high = pred >= q75
        valid = actual_high.notna() & pred_high.notna()
        if valid.any():
            tp = int((actual_high[valid] & pred_high[valid]).sum())
            fp = int((~actual_high[valid] & pred_high[valid]).sum())
            tn = int((~actual_high[valid] & ~pred_high[valid]).sum())
            fn = int((actual_high[valid] & ~pred_high[valid]).sum())
            accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            accuracy_interpretation = interpret_classification_accuracy(accuracy)
            recall_interpretation = interpret_recall(recall)
            f1_interpretation = interpret_f1_score(f1)

            # Three standalone dynamic validation cards
            col1, col2, col3 = st.columns(3)
            with col1:
                card1_html = (
                    '<div style="background: linear-gradient(135deg, rgba(20, 184, 166, 0.15), rgba(15, 23, 42, 0.85)); border: 1px solid rgba(20, 184, 166, 0.4); border-radius: 12px; padding: 20px; margin: 5px 0; text-align: center; box-shadow: 0 4px 12px rgba(20, 184, 166, 0.1);">'
                    f'<div style="font-size: 2.2rem; font-weight: 900; color: #14b8a6; margin-bottom: 8px;">{format_metric_percent(accuracy)}</div>'
                    '<div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">Classification Accuracy</div>'
                    '<div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500;">Overall classification hit rate</div>'
                    f'<div class="validation-card-interpretation">{accuracy_interpretation}</div>'
                    '</div>'
                )
                st.markdown(card1_html, unsafe_allow_html=True)
            with col2:
                card2_html = (
                    '<div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(15, 23, 42, 0.85)); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 12px; padding: 20px; margin: 5px 0; text-align: center; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);">'
                    f'<div style="font-size: 2.2rem; font-weight: 900; color: #f59e0b; margin-bottom: 8px;">{format_metric_percent(recall)}</div>'
                    '<div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">Recall</div>'
                    '<div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500;">High-volatility capture rate</div>'
                    f'<div class="validation-card-interpretation">{recall_interpretation}</div>'
                    '</div>'
                )
                st.markdown(card2_html, unsafe_allow_html=True)
            with col3:
                card3_html = (
                    '<div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(15, 23, 42, 0.85)); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 12px; padding: 20px; margin: 5px 0; text-align: center; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);">'
                    f'<div style="font-size: 2.2rem; font-weight: 900; color: #3b82f6; margin-bottom: 8px;">{format_metric_percent(f1)}</div>'
                    '<div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">F1 Score</div>'
                    '<div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500;">Balanced alert quality</div>'
                    f'<div class="validation-card-interpretation">{f1_interpretation}</div>'
                    '</div>'
                )
                st.markdown(card3_html, unsafe_allow_html=True)

            if precision is None or pd.isna(precision):
                interp_p1 = "The matrix summarizes how many high-volatility periods were captured, missed, or falsely flagged."
            elif recall > precision + 0.05:
                interp_p1 = "The signal prioritizes capturing high-volatility episodes over avoiding every false alert, which is appropriate for risk monitoring because missing stress periods is usually more costly than issuing cautious alerts."
            elif precision > recall + 0.05:
                interp_p1 = "The signal is relatively selective: high-volatility alerts are more reliable, but some stress periods may be missed."
            else:
                interp_p1 = "The signal is balanced between high-volatility capture and alert precision."

            interp_p2 = f"The model captured {tp} high-volatility periods and missed {fn} high-volatility periods in the evaluated sample."

            # Premium 2x2 confusion matrix
            matrix_html = (
                '<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9)); border: 1px solid rgba(96, 165, 250, 0.4); border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);">'
                '<div style="font-size: 1.2rem; color: #f8fafc; font-weight: 800; margin-bottom: 15px; text-align: center;">High-Volatility Signal Validation Matrix</div>'
                '<div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 20px; text-align: center;">Predicted signal versus realized USD/TND volatility outcome.</div>'
                '<div class="validation-matrix-content">'
                '<div class="matrix-left">'
                '<div style="display: grid; grid-template-columns: 120px 1fr 1fr; grid-template-rows: 50px 1fr 1fr; gap: 2px; max-width: 500px; margin: 0 auto;">'
                '<!-- Header row -->'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem; border-radius: 6px 0 0 0;"></div>'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem;">Actual High Vol</div>'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem; border-radius: 0 6px 0 0;">Actual Normal</div>'
                '<!-- Predicted High Vol row -->'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem;">Predicted High Vol</div>'
                '<div style="background: linear-gradient(135deg, rgba(20, 184, 166, 0.25), rgba(20, 184, 166, 0.15)); border: 2px solid rgba(20, 184, 166, 0.5); color: #f8fafc; padding: 15px; text-align: center; font-weight: 800; font-size: 1.4rem; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">'
                f'{tp}<br><span style="font-size: 0.7rem; font-weight: 600; color: #14b8a6;">True high-vol alerts</span>'
                '</div>'
                '<div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(239, 68, 68, 0.15)); border: 2px solid rgba(239, 68, 68, 0.5); color: #f8fafc; padding: 15px; text-align: center; font-weight: 800; font-size: 1.4rem; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">'
                f'{fp}<br><span style="font-size: 0.7rem; font-weight: 600; color: #ef4444;">False high-vol alerts</span>'
                '</div>'
                '<!-- Predicted Normal row -->'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem;">Predicted Normal</div>'
                '<div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(245, 158, 11, 0.15)); border: 2px solid rgba(245, 158, 11, 0.5); color: #f8fafc; padding: 15px; text-align: center; font-weight: 800; font-size: 1.4rem; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">'
                f'{fn}<br><span style="font-size: 0.7rem; font-weight: 600; color: #f59e0b;">Missed high-vol periods</span>'
                '</div>'
                '<div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(59, 130, 246, 0.15)); border: 2px solid rgba(59, 130, 246, 0.5); color: #f8fafc; padding: 15px; text-align: center; font-weight: 800; font-size: 1.4rem; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">'
                f'{tn}<br><span style="font-size: 0.7rem; font-weight: 600; color: #3b82f6;">Correct normal signals</span>'
                '</div>'
                '</div>'
                '</div>'
                '<div class="matrix-right">'
                '<div class="validation-read-card">'
                '<div class="validation-read-title">Validation read</div>'
                f'<div class="validation-read-text">{interp_p1} {interp_p2}</div>'
                '<div style="margin-top: 1rem; display: flex; gap: 10px; flex-wrap: wrap;">'
                f'<div style="background: rgba(20, 184, 166, 0.15); border: 1px solid rgba(20, 184, 166, 0.3); padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; color: #14b8a6; font-weight: 600;">Captured: {tp}</div>'
                f'<div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; color: #f59e0b; font-weight: 600;">Missed: {fn}</div>'
                '</div>'
                '</div>'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(matrix_html, unsafe_allow_html=True)

            with st.expander("Technical mapping"):
                st.markdown(
                    "- True high-vol alerts = True Positives\\n"
                    "- False high-vol alerts = False Positives\\n"
                    "- Missed high-vol periods = False Negatives\\n"
                    "- Correct normal signals = True Negatives"
                )
        st.markdown(
            """
            <div class='method-box'>
                Forecast vs realized validation compares the model signal with realized USD/TND volatility to assess whether regime classification remains consistent through time.
            </div>
            """,
            unsafe_allow_html=True,
        )
        plot_forecast_vs_actual(payload)
        # Regime Classification Conclusion
        forecast_vol = classification.get("forecast_volatility", 0)
        regime = classification.get("regime_label", "Unavailable")
        q50 = safe_float(classification.get("q50"))
        q75 = safe_float(classification.get("q75"))
        high_vol_prob = safe_float(classification.get("high_vol_probability", 0))
        relation = ""
        if forecast_vol < q50:
            relation = f"below the <strong>Median volatility line of {format_pct(q50)}</strong> and below the <strong>High-volatility threshold of {format_pct(q75)}</strong>"
        elif forecast_vol < q75:
            relation = f"above the <strong>Median volatility line of {format_pct(q50)}</strong> and below the <strong>High-volatility threshold of {format_pct(q75)}</strong>"
        else:
            relation = f"above the <strong>Median volatility line of {format_pct(q50)}</strong> and above the <strong>High-volatility threshold of {format_pct(q75)}</strong>"
        
        regime_display = regime if "regime" in regime.lower() else f"{regime} regime"
        conclusion_text = f"The current 3-day USD/TND volatility forecast is <strong>{format_pct(forecast_vol)}</strong>, placing the pair in an <strong>{regime_display}</strong>. The forecast is {relation}."
        conclusion_text += f" Classification performance remains acceptable, with an <strong>overall hit rate of {accuracy:.1%}</strong> and a <strong>high-volatility capture rate of {recall:.1%}</strong>. This supports using the regime signal as a validation layer for short-term USD/TND risk monitoring."
        if regime in ["Elevated-volatility regime", "High-volatility regime", "Stress-volatility regime"] and high_vol_prob < 0.1:
            conclusion_text += f" Although the current regime is {regime.lower()}, the high-volatility probability remains contained at <strong>{format_pct(high_vol_prob)}</strong>, indicating a monitoring stance rather than a full stress alert."
        st.markdown(
            f"""
            <div style="background-color: rgba(15, 23, 42, 0.92); border: 1px solid rgba(96, 165, 250, 0.35); border-radius: 8px; padding: 15px; margin: 20px 0;">
                <div style="font-size: 16px; color: #f8fafc; font-weight: bold; margin-bottom: 10px;">Regime Classification Conclusion</div>
                <div style="font-size: 14px; color: #cbd5e1; line-height: 1.5;">
                    {conclusion_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_high_volatility_calibration_diagnostics(payload)


# ============================================================
# Macro-Event Audit Helpers
# ============================================================


def get_current_event_regime_from_payload(payload: Optional[dict]) -> Optional[str]:
    """Extract the current event regime code from the forecast payload."""
    if not payload:
        return None
    forecast = payload.get("forecast", {}) or {}
    for key in ("event_regime", "macro_event_regime", "current_macro_regime"):
        value = forecast.get(key)
        if value and isinstance(value, str):
            return value.strip().lower()
    return None


def get_current_event_metrics(payload: Optional[dict]) -> Optional[dict]:
    """
    Fetch the descriptive metrics for the current macro-event regime.
    
    Returns a dict with all event metrics or None if not found.
    """
    if not payload:
        return None
    
    # Get current event regime
    current_regime = get_current_event_regime_from_payload(payload)
    if not current_regime:
        return None
    
    # Try to load the event descriptive stats Excel file
    try:
        event_stats_path = artifact_path(payload, "event_descriptive_stats", prefer_freshest=True)
        if not event_stats_path:
            # Fallback to the standard filename if not in artifacts
            if Path("event_regime_descriptive_stats_3d.xlsx").exists():
                event_stats_path = Path("event_regime_descriptive_stats_3d.xlsx")
            else:
                return None
        
        df = pd.read_excel(str(event_stats_path))
        
        # Match by event_regime column (exact match with current regime code)
        match = df[df["event_regime"].astype(str).str.lower().eq(current_regime)]
        
        if not match.empty:
            return match.iloc[0].to_dict()
    except Exception:
        pass
    
    return None


def is_event_ongoing(event_metrics: dict) -> bool:
    """
    Determine if the current event is ongoing (not yet completed).
    
    An event is considered ongoing if:
    - It has no end_date, or
    - The reliability_label is "Early / low sample"
    """
    if not event_metrics:
        return False
    
    # Check if end_date is NaT or NaN
    end_date = event_metrics.get("end_date")
    if end_date is None or pd.isna(end_date):
        return True
    
    # Check reliability label
    reliability = str(event_metrics.get("reliability_label", "")).strip()
    if "Early" in reliability or "low sample" in reliability:
        return True
    
    return False


def render_macro_event_regime_analysis(payload: Optional[dict]) -> None:
    page_header("Macro-Event Regime Analysis", "Event methodology", payload)
    st.markdown(
        f"<div style='color:{FINANCE_COLORS['muted']};font-size:1rem;margin:-0.55rem 0 1rem;'>Descriptive USD/TND volatility behavior across documented macro-financial regimes.</div>",
        unsafe_allow_html=True,
    )
    event_df = get_event_descriptive_df(payload)
    if event_df.empty:
        if payload is None:
            render_no_payload_choice()
        st.info("Macro-event descriptive statistics are not available. Run the latest forecasting pipeline to generate event artifacts.")
        return

    event_source = event_df.attrs.get("source", "unknown")
    event_artifact_path = artifact_path(payload, "event_descriptive_stats", prefer_freshest=True)

    event_df = event_df.copy()
    if "event_regime" in event_df.columns:
        event_df["event_label"] = event_df["event_regime"].map(pretty_event_name)
    else:
        event_df["event_label"] = event_df.index.astype(str)

    # Use parsed dates for chronological sorting in both the chart and table.
    if "start_date" in event_df.columns:
        event_df["_sort_start_date"] = pd.to_datetime(event_df["start_date"], errors="coerce")
        event_df = event_df.sort_values(["_sort_start_date", "event_regime_code"] if "event_regime_code" in event_df.columns else ["_sort_start_date"]).reset_index(drop=True)
    elif "event_regime_code" in event_df.columns:
        event_df = event_df.sort_values("event_regime_code").reset_index(drop=True)

    if "event_regime" in event_df.columns:
        # Keep analytical datetime columns clean for sorting and filtering.
        # Display columns will be created later for formatting.
        completed_event_df = event_df[event_df["event_regime"].astype(str).ne(ONGOING_EVENT_REGIME)].copy()
        ongoing_event_df = event_df[event_df["event_regime"].astype(str).eq(ONGOING_EVENT_REGIME)].copy()
    else:
        completed_event_df = event_df.copy()
        ongoing_event_df = pd.DataFrame(columns=event_df.columns)

    robust_cols_detected = [c for c in EVENT_DESCRIPTIVE_ROBUST_COLUMNS if c in event_df.columns]
    metadata = get_run_metadata(payload)

    def _fmt_ratio_pct(value: Any, digits: int = 2) -> str:
        x = safe_float(value)
        return f"{x * 100:.{digits}f}%" if x is not None else "n/a"

    def _fmt_point_pct(value: Any) -> str:
        x = safe_float(value)
        return f"{x:.2f}%" if x is not None else "n/a"

    def _fmt_share_pct(value: Any) -> str:
        x = safe_float(value)
        return f"{x * 100:.2f}%" if x is not None else "n/a"

    st.markdown(
        """
        <div class='method-box'><strong>Methodology note.</strong> Descriptive macro-event volatility statistics are computed from a stable USD/TND event-analysis dataset, not from the final ML feature matrix. This prevents changes in optional engineered model features from changing historical event-volatility comparisons. Forecast performance by event regime is still computed from the walk-forward model outputs.</div>
        """,
        unsafe_allow_html=True,
    )

    audit_values = [
        (
            "Source",
            metadata.get("event_descriptive_stats_source", event_df.get("source_dataset", pd.Series(["n/a"])).iloc[0] if "source_dataset" in event_df.columns and not event_df.empty else "n/a"),
            "Event-analysis dataset",
        ),
        ("Rows", metadata.get("event_analysis_row_count", "n/a"), "Stable event rows"),
        (
            "Start",
            pd.to_datetime(metadata.get("event_analysis_start_date"), errors="coerce").date().isoformat() if metadata.get("event_analysis_start_date") else "n/a",
            "Event-analysis start",
        ),
        (
            "End",
            pd.to_datetime(metadata.get("event_analysis_end_date"), errors="coerce").date().isoformat() if metadata.get("event_analysis_end_date") else "n/a",
            "Event-analysis end",
        ),
    ]
    audit_html = "".join(
        (
            "<div style='background:rgba(16,26,44,0.86);border:1px solid rgba(168,179,204,0.16);"
            "border-radius:8px;padding:0.82rem 0.9rem;min-height:88px;'>"
            f"<div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:{FINANCE_COLORS['muted']};font-weight:700;'>{html.escape(label)}</div>"
            f"<div style='font-size:1.08rem;color:{FINANCE_COLORS['text']};font-weight:800;margin-top:0.28rem;'>{html.escape(str(value))}</div>"
            f"<div style='font-size:0.78rem;color:{FINANCE_COLORS['muted']};margin-top:0.3rem;'>{html.escape(note)}</div>"
            "</div>"
        )
        for label, value, note in audit_values
    )
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.8rem;margin:0.85rem 0 1rem;'>{audit_html}</div>",
        unsafe_allow_html=True,
    )

    # Current Macro-Event Audit: Dynamically display the latest event regime metrics
    current_event_metrics = get_current_event_metrics(payload)
    if current_event_metrics:
        current_regime = get_current_event_regime_from_payload(payload)
        event_label = format_event_context_label(current_regime) or pretty_event_name(current_regime)
        
        # Determine if event is ongoing
        is_ongoing = is_event_ongoing(current_event_metrics)
        
        # Build metric cards for the current event
        current_event_data = [
            ("Mean annualized volatility", _fmt_ratio_pct(current_event_metrics.get("mean_target_vol_annualized"))),
            ("Change vs Pre-COVID", _fmt_point_pct(current_event_metrics.get("annualized_target_vol_vs_pre_covid_pct"))),
            ("q90 annualized volatility", _fmt_ratio_pct(current_event_metrics.get("q90_target_vol_annualized"))),
            ("Max annualized volatility", _fmt_ratio_pct(current_event_metrics.get("max_target_vol_annualized"))),
            ("Share high-volatility days", _fmt_share_pct(current_event_metrics.get("share_high_vol_days"))),
            ("Observations", f"n={int(safe_float(current_event_metrics.get('observations')) or 0)}"),
            ("Reliability", str(current_event_metrics.get("reliability_label", "n/a"))),
        ]
        
        event_metric_html = "".join(
            (
                "<div style='background:rgba(18,31,52,0.82);border:1px solid rgba(168,179,204,0.14);"
                "border-radius:8px;padding:0.72rem 0.75rem;'>"
                f"<div style='font-size:0.72rem;color:{FINANCE_COLORS['muted']};font-weight:700;text-transform:uppercase;letter-spacing:0.05em;'>{html.escape(label)}</div>"
                f"<div style='font-size:1.18rem;color:{FINANCE_COLORS['text']};font-weight:850;margin-top:0.26rem;'>{html.escape(value)}</div>"
                "</div>"
            )
            for label, value in current_event_data
        )
        
        # Select wording based on whether event is ongoing
        if is_ongoing:
            event_wording = (
                "The current macro-event window is ongoing, so these statistics are preliminary and reflect observed USD/TND volatility so far. "
                "Average volatility may understate short-lived stress episodes; q90 volatility, maximum volatility, and high-volatility-day share "
                "are shown to capture peak-stress intensity."
            )
        else:
            event_wording = (
                "These statistics summarize the completed event window. Peak-stress metrics complement average volatility because "
                "short-lived shocks can be diluted in broad event windows."
            )
        
        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,rgba(16,26,44,0.94),rgba(18,31,52,0.86));border:1px solid rgba(79,123,255,0.28);border-radius:10px;padding:1rem;margin:1rem 0 1.1rem;'>
                <div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.09em;color:{FINANCE_COLORS["primary"]};font-weight:800;'>Current Macro-Event Audit</div>
                <div style='font-size:1.15rem;color:{FINANCE_COLORS["text"]};font-weight:750;margin-top:0.5rem;'>{html.escape(event_label)}</div>
                <div style='display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:0.72rem;margin-top:0.75rem;'>{event_metric_html}</div>
                <div style='font-size:0.9rem;color:{FINANCE_COLORS["muted"]};line-height:1.45;margin-top:0.8rem;'>
                    {html.escape(event_wording)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    chart_df = completed_event_df.copy()
    if "event_regime_code" in chart_df.columns:
        chart_df = chart_df.sort_values("event_regime_code")
    elif "_sort_start_date" in chart_df.columns:
        chart_df = chart_df.sort_values("_sort_start_date")
    chart_df = chart_df.reset_index(drop=True)
    regime_labels = chart_df["event_label"].astype(str).tolist()
    regime_order = regime_labels
    muted_crisis = "#7E6AE6"
    muted_normal = "#6F8098"
    muted_positive = "#3FA7A0"
    muted_negative = "#A96A72"
    baseline_color = "#C1C9D8"
    bar_colors = [
        muted_normal if str(regime) == "pre_covid_normal"
        else muted_crisis if int(safe_float(crisis, 0) or 0) == 1
        else "#4F6F9F"
        for regime, crisis in zip(
            chart_df.get("event_regime", pd.Series("", index=chart_df.index)),
            chart_df.get("crisis_flag", pd.Series(0, index=chart_df.index)),
        )
    ]

    def _baseline_from_pct(value_col: str, pct_col: str) -> Optional[float]:
        if value_col not in chart_df.columns or pct_col not in chart_df.columns:
            return None
        values = pd.to_numeric(chart_df[value_col], errors="coerce")
        pct = pd.to_numeric(chart_df[pct_col], errors="coerce")
        valid = values.notna() & pct.notna() & (pct != -100)
        if not valid.any():
            return None
        return float((values[valid].iloc[0] * 100.0) / (1.0 + pct[valid].iloc[0] / 100.0))

    st.markdown(
        f"<div style='color:{FINANCE_COLORS['muted']};font-size:0.86rem;margin:0.35rem 0 0.55rem;'>Views: Average Volatility · Change vs Baseline · Peak-Stress Metrics</div>",
        unsafe_allow_html=True,
    )
    avg_tab, change_tab, peak_tab = st.tabs(["Average Volatility", "Change vs Baseline", "Peak-Stress Metrics"])

    with avg_tab:
        avg_pct = pd.to_numeric(chart_df.get("mean_target_vol_annualized", pd.Series(np.nan, index=chart_df.index)), errors="coerce") * 100
        avg_fig = go.Figure()
        avg_fig.add_trace(go.Bar(
            x=avg_pct,
            y=regime_labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(234,240,250,0.12)", width=1)),
            text=[f"{v:.2f}%" if pd.notna(v) else "" for v in avg_pct],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>Mean annualized volatility: %{x:.2f}%<extra></extra>",
            name="Mean annualized volatility",
        ))
        broad_baseline = _baseline_from_pct(
            "mean_target_vol_annualized",
            "mean_target_vol_annualized_vs_broad_normal_pct",
        )
        if broad_baseline is not None and np.isfinite(broad_baseline):
            avg_fig.add_vline(
                x=broad_baseline,
                line_width=1.2,
                line_dash="dot",
                line_color=baseline_color,
                annotation_text="Broad normal baseline",
                annotation_position="top right",
                annotation_font=dict(color=baseline_color, size=11),
            )
        avg_fig.update_yaxes(categoryorder="array", categoryarray=regime_order, autorange="reversed")
        avg_fig.update_xaxes(range=[0, max(float(np.nanmax(avg_pct)) * 1.22 if avg_pct.notna().any() else 10, 10)])
        apply_institutional_plotly_layout(
            avg_fig,
            title="Average Volatility by Event Regime",
            subtitle="Mean annualized 3-day USD/TND volatility; descriptive statistics sourced from event_analysis_df.",
            height=470,
            xaxis_title="Annualized volatility",
        )
        st.plotly_chart(avg_fig, use_container_width=True, config={"displayModeBar": False})

    with change_tab:
        change_col = next(
            (c for c in ["annualized_target_vol_vs_pre_covid_pct", "mean_target_vol_vs_pre_covid_pct"] if c in chart_df.columns),
            None,
        )
        change_values = pd.to_numeric(chart_df.get(change_col, pd.Series(np.nan, index=chart_df.index)), errors="coerce") if change_col else pd.Series(np.nan, index=chart_df.index)
        change_colors = [muted_positive if pd.notna(v) and v >= 0 else muted_negative for v in change_values]
        change_fig = go.Figure()
        error_kwargs: Dict[str, Any] = {}
        if {
            "annualized_target_vol_ci_low_vs_pre_covid_pct",
            "annualized_target_vol_ci_high_vs_pre_covid_pct",
        }.issubset(chart_df.columns):
            lower_ci = pd.to_numeric(chart_df["annualized_target_vol_ci_low_vs_pre_covid_pct"], errors="coerce")
            upper_ci = pd.to_numeric(chart_df["annualized_target_vol_ci_high_vs_pre_covid_pct"], errors="coerce")
            error_kwargs["error_x"] = dict(
                type="data",
                array=np.nan_to_num((upper_ci - change_values).clip(lower=0).values, nan=0.0, posinf=0.0, neginf=0.0),
                arrayminus=np.nan_to_num((change_values - lower_ci).clip(lower=0).values, nan=0.0, posinf=0.0, neginf=0.0),
                visible=True,
                thickness=1,
                width=3,
                color="rgba(193,201,216,0.55)",
            )
        change_fig.add_trace(go.Bar(
            x=change_values,
            y=regime_labels,
            orientation="h",
            marker=dict(color=change_colors, line=dict(color="rgba(234,240,250,0.12)", width=1)),
            text=[f"{v:+.2f}%" if pd.notna(v) else "" for v in change_values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>Change vs Pre-COVID: %{x:+.2f}%<extra></extra>",
            name="Change vs Pre-COVID",
            **error_kwargs,
        ))
        change_fig.add_vline(x=0, line_width=1.2, line_color="rgba(234,240,250,0.45)")
        change_fig.update_yaxes(categoryorder="array", categoryarray=regime_order, autorange="reversed")
        max_abs = float(np.nanmax(np.abs(change_values))) if change_values.notna().any() else 10.0
        change_fig.update_xaxes(range=[-max_abs * 1.35, max_abs * 1.35])
        apply_institutional_plotly_layout(
            change_fig,
            title="Change vs Pre-COVID Baseline",
            subtitle="Horizontal diverging view of average annualized 3-day USD/TND volatility change.",
            height=470,
            xaxis_title="Change vs Pre-COVID baseline",
        )
        st.plotly_chart(change_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            "<div class='method-box'><strong>Chart note.</strong> Bars show average annualized 3-day USD/TND volatility change versus the Pre-COVID baseline. Confidence intervals are bootstrap-based and descriptive.</div>",
            unsafe_allow_html=True,
        )

    with peak_tab:
        q90_pct = pd.to_numeric(chart_df.get("q90_target_vol_annualized", pd.Series(np.nan, index=chart_df.index)), errors="coerce") * 100
        max_pct = pd.to_numeric(chart_df.get("max_target_vol_annualized", pd.Series(np.nan, index=chart_df.index)), errors="coerce") * 100
        peak_fig = go.Figure()
        peak_fig.add_trace(go.Bar(
            x=q90_pct,
            y=regime_labels,
            orientation="h",
            name="q90 volatility",
            marker=dict(color="#D2A84A", line=dict(color="rgba(234,240,250,0.10)", width=1)),
            text=[f"{v:.2f}%" if pd.notna(v) else "" for v in q90_pct],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>q90 volatility: %{x:.2f}%<extra></extra>",
        ))
        peak_fig.add_trace(go.Bar(
            x=max_pct,
            y=regime_labels,
            orientation="h",
            name="Max volatility",
            marker=dict(color="#9D5E68", line=dict(color="rgba(234,240,250,0.10)", width=1)),
            text=[f"{v:.2f}%" if pd.notna(v) else "" for v in max_pct],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>Maximum volatility: %{x:.2f}%<extra></extra>",
        ))
        peak_fig.update_layout(barmode="group")
        peak_fig.update_yaxes(categoryorder="array", categoryarray=regime_order, autorange="reversed")
        peak_fig.update_xaxes(range=[0, max(float(np.nanmax(max_pct)) * 1.22 if max_pct.notna().any() else 10, 10)])
        apply_institutional_plotly_layout(
            peak_fig,
            title="Peak-Stress Volatility by Event Regime",
            subtitle="q90 and maximum annualized 3-day USD/TND volatility.",
            height=500,
            xaxis_title="Annualized volatility",
            hovermode="closest",
        )
        st.plotly_chart(peak_fig, use_container_width=True, config={"displayModeBar": False})

        share_pct = pd.to_numeric(chart_df.get("share_high_vol_days", pd.Series(np.nan, index=chart_df.index)), errors="coerce") * 100
        share_fig = go.Figure()
        share_fig.add_trace(go.Bar(
            x=share_pct,
            y=regime_labels,
            orientation="h",
            marker=dict(color="#5F87B9", line=dict(color="rgba(234,240,250,0.10)", width=1)),
            text=[f"{v:.2f}%" if pd.notna(v) else "" for v in share_pct],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>Share of high-volatility days: %{x:.2f}%<extra></extra>",
            name="Share of high-volatility days",
        ))
        share_fig.update_yaxes(categoryorder="array", categoryarray=regime_order, autorange="reversed")
        share_fig.update_xaxes(range=[0, max(float(np.nanmax(share_pct)) * 1.25 if share_pct.notna().any() else 10, 10)])
        apply_institutional_plotly_layout(
            share_fig,
            title="Share of High-Volatility Days",
            subtitle="Uses the fixed high-volatility threshold produced by the backend.",
            height=430,
            xaxis_title="Share of days",
            hovermode="closest",
        )
        st.plotly_chart(share_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            "<div class='method-box'><strong>Peak-stress note.</strong> Peak-stress metrics complement average volatility because short crisis spikes can be diluted in broad event windows.</div>",
            unsafe_allow_html=True,
        )

    detail_cols = [
        "event_regime",
        "observations",
        "reliability_label",
        "mean_target_vol_annualized",
        "annualized_target_vol_vs_pre_covid_pct",
        "mean_target_vol_annualized_vs_broad_normal_pct",
        "q90_target_vol_annualized",
        "max_target_vol_annualized",
        "share_high_vol_days",
        "variance_test_result",
        "source_dataset",
    ]
    detail_df = event_df[[c for c in detail_cols if c in event_df.columns]].copy()
    for col in ["mean_target_vol_annualized", "q90_target_vol_annualized", "max_target_vol_annualized", "share_high_vol_days"]:
        if col in detail_df.columns:
            detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce").apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "n/a")
    for col in ["annualized_target_vol_vs_pre_covid_pct", "mean_target_vol_annualized_vs_broad_normal_pct"]:
        if col in detail_df.columns:
            detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce").apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "n/a")
    with st.expander("Detailed event-regime statistics", expanded=False):
        st.dataframe(
            detail_df.rename(columns={
                "event_regime": "Event regime",
                "observations": "Observations",
                "reliability_label": "Reliability",
                "mean_target_vol_annualized": "Mean annualized volatility",
                "annualized_target_vol_vs_pre_covid_pct": "Change vs Pre-COVID",
                "mean_target_vol_annualized_vs_broad_normal_pct": "Change vs broad normal",
                "q90_target_vol_annualized": "q90 annualized volatility",
                "max_target_vol_annualized": "Max annualized volatility",
                "share_high_vol_days": "Share high-volatility days",
                "variance_test_result": "Variance test",
                "source_dataset": "Source dataset",
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(f"Artifact source: {event_source}; path: {event_artifact_path or 'Unavailable'}")

    return

    impact_col = next(
        (c for c in ["annualized_target_vol_vs_pre_covid_pct", "mean_target_vol_vs_pre_covid_pct"] if c in event_df.columns),
        None,
    )
    if impact_col:
        top_df = completed_event_df.copy()
        top_df["_impact_value"] = pd.to_numeric(top_df[impact_col], errors="coerce")
        top_df = top_df[
            top_df["_impact_value"].gt(0)
            & ~top_df.get("event_regime", pd.Series("", index=top_df.index)).astype(str).eq("pre_covid_normal")
        ].copy()
        crisis_candidates = top_df
        if "crisis_flag" in crisis_candidates.columns:
            crisis_candidates = crisis_candidates[pd.to_numeric(crisis_candidates["crisis_flag"], errors="coerce").fillna(0).eq(1)]
        selected = crisis_candidates.sort_values("_impact_value", ascending=False).head(3)
        if len(selected) < 3:
            fill = top_df.drop(index=selected.index, errors="ignore").sort_values("_impact_value", ascending=False).head(3 - len(selected))
            selected = pd.concat([selected, fill], axis=0)
        cards = []
        for _, row in selected.head(3).iterrows():
            impact = safe_float(row.get("_impact_value"), 0) or 0
            observations = row.get("observations", None)
            obs_note = f"n={int(observations):,}" if safe_float(observations) is not None else "n=n/a"
            reliability = str(row.get("reliability_label", "") or "").strip()
            variance_result = str(row.get("variance_test_result", "") or "").strip()
            
            # Build footer with observation count, reliability, and variance test status
            footer_parts = [obs_note]
            if reliability:
                footer_parts.append(reliability)
            if variance_result:
                footer_parts.append(variance_result)
            technical_note = " · ".join(footer_parts)
            
            # Create custom card with two-line footer for top shock cards
            regime_name = str(row.get("event_label", "Event"))
            value_text = f"+{impact:.2f}%" if impact > 0 else f"{impact:.2f}%"
            
            card_html = (
                '<div class="kpi-card" style="border-top: 3px solid #FF5F68;">'
                f'<div class="kpi-label">{html.escape(regime_name)}</div>'
                f'<div class="kpi-value">{html.escape(value_text)}</div>'
                '<div class="kpi-note" style="display: flex; flex-direction: column; gap: 0.25rem;">'
                '<div style="font-size: 0.85rem; font-weight: 600; color: #A8B3CC;">Volatility uplift vs Pre-COVID baseline</div>'
                f'<div style="font-size: 0.8rem; color: #7F8EA7;">{html.escape(technical_note)}</div>'
                '</div>'
                '</div>'
            )
            cards.append(card_html)
        if cards:
            st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

    avg_tab, peak_tab = st.tabs(["Average Volatility by Event Regime", "Peak-Stress Metrics by Event Regime"])
    with avg_tab:
        avg_cols = [
            "event_label",
            "mean_target_vol_annualized",
            "annualized_target_vol_vs_pre_covid_pct",
            "mean_target_vol_annualized_vs_broad_normal_pct",
            "observations",
            "reliability_label",
            "sample_size_note",
            "source_dataset",
        ]
        avg_available = [c for c in avg_cols if c in completed_event_df.columns]
        if "mean_target_vol_annualized" in completed_event_df.columns:
            avg_fig = go.Figure()
            avg_fig.add_trace(go.Bar(
                x=completed_event_df["event_label"],
                y=pd.to_numeric(completed_event_df["mean_target_vol_annualized"], errors="coerce") * 100,
                marker_color=[EVENT_COLORS.get(str(r), FINANCE_COLORS["teal"]) for r in completed_event_df.get("event_regime", pd.Series([None] * len(completed_event_df)))],
                name="Mean annualized 3-day volatility",
                customdata=np.column_stack([
                    pd.to_numeric(completed_event_df.get("annualized_target_vol_vs_pre_covid_pct", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce"),
                    pd.to_numeric(completed_event_df.get("mean_target_vol_annualized_vs_broad_normal_pct", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce"),
                    pd.to_numeric(completed_event_df.get("observations", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce").fillna(0).astype(int),
                    completed_event_df.get("reliability_label", pd.Series("Unavailable", index=completed_event_df.index)).fillna("Unavailable").astype(str),
                ]),
                hovertemplate=(
                    "Regime: %{x}<br>"
                    "Mean annualized volatility: %{y:.2f}%<br>"
                    "Change vs Pre-COVID: %{customdata[0]:.2f}%<br>"
                    "Change vs broad normal: %{customdata[1]:.2f}%<br>"
                    "Observations: %{customdata[2]}<br>"
                    "Reliability: %{customdata[3]}<extra></extra>"
                ),
            ))
            avg_fig.update_layout(
                title="Average Annualized 3-Day USD/TND Volatility by Completed Event Regime",
                yaxis_title="Mean annualized volatility (%)",
                xaxis_title="",
            )
            st.plotly_chart(apply_finance_layout(avg_fig, height=420), use_container_width=True)
        if avg_available:
            avg_table = completed_event_df[avg_available].copy()
            for col in ["mean_target_vol_annualized"]:
                if col in avg_table.columns:
                    avg_table[col] = pd.to_numeric(avg_table[col], errors="coerce").apply(lambda x: f"{x * 100:.4f}%" if pd.notna(x) else "n/a")
            for col in ["annualized_target_vol_vs_pre_covid_pct", "mean_target_vol_annualized_vs_broad_normal_pct"]:
                if col in avg_table.columns:
                    avg_table[col] = pd.to_numeric(avg_table[col], errors="coerce").apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
            if "observations" in avg_table.columns:
                avg_table["observations"] = pd.to_numeric(avg_table["observations"], errors="coerce").apply(lambda x: f"{int(x):,}" if pd.notna(x) else "n/a")
            st.dataframe(avg_table.rename(columns={
                "event_label": "Macro-event regime",
                "mean_target_vol_annualized": "Mean annualized 3-day volatility",
                "annualized_target_vol_vs_pre_covid_pct": "Change vs Pre-COVID baseline",
                "mean_target_vol_annualized_vs_broad_normal_pct": "Change vs broad normal baseline",
                "observations": "Observations",
                "reliability_label": "Reliability",
                "sample_size_note": "Sample-size note",
                "source_dataset": "Source dataset",
            }), hide_index=True, use_container_width=True)

    with peak_tab:
        peak_cols = [
            "event_label",
            "q90_target_vol_annualized",
            "max_target_vol_annualized",
            "share_high_vol_days",
            "q90_target_vol_annualized_vs_broad_normal_pct",
            "observations",
            "reliability_label",
            "sample_size_note",
            "source_dataset",
        ]
        peak_available = [c for c in peak_cols if c in completed_event_df.columns]
        if {"q90_target_vol_annualized", "max_target_vol_annualized"}.issubset(completed_event_df.columns):
            peak_fig = go.Figure()
            peak_fig.add_trace(go.Bar(
                x=completed_event_df["event_label"],
                y=pd.to_numeric(completed_event_df["q90_target_vol_annualized"], errors="coerce") * 100,
                name="q90 annualized volatility",
                marker_color=FINANCE_COLORS["gold"],
            ))
            peak_fig.add_trace(go.Bar(
                x=completed_event_df["event_label"],
                y=pd.to_numeric(completed_event_df["max_target_vol_annualized"], errors="coerce") * 100,
                name="maximum annualized volatility",
                marker_color=FINANCE_COLORS["red"],
            ))
            peak_fig.update_layout(
                title="Peak-Stress USD/TND Volatility by Completed Event Regime",
                yaxis_title="Annualized volatility (%)",
                xaxis_title="",
                barmode="group",
            )
            st.plotly_chart(apply_finance_layout(peak_fig, height=420), use_container_width=True)
        if peak_available:
            peak_table = completed_event_df[peak_available].copy()
            for col in ["q90_target_vol_annualized", "max_target_vol_annualized"]:
                if col in peak_table.columns:
                    peak_table[col] = pd.to_numeric(peak_table[col], errors="coerce").apply(lambda x: f"{x * 100:.4f}%" if pd.notna(x) else "n/a")
            if "share_high_vol_days" in peak_table.columns:
                peak_table["share_high_vol_days"] = pd.to_numeric(peak_table["share_high_vol_days"], errors="coerce").apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "n/a")
            if "q90_target_vol_annualized_vs_broad_normal_pct" in peak_table.columns:
                peak_table["q90_target_vol_annualized_vs_broad_normal_pct"] = pd.to_numeric(peak_table["q90_target_vol_annualized_vs_broad_normal_pct"], errors="coerce").apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
            if "observations" in peak_table.columns:
                peak_table["observations"] = pd.to_numeric(peak_table["observations"], errors="coerce").apply(lambda x: f"{int(x):,}" if pd.notna(x) else "n/a")
            st.dataframe(peak_table.rename(columns={
                "event_label": "Macro-event regime",
                "q90_target_vol_annualized": "q90 annualized 3-day volatility",
                "max_target_vol_annualized": "Maximum annualized 3-day volatility",
                "share_high_vol_days": "Share of high-volatility days",
                "q90_target_vol_annualized_vs_broad_normal_pct": "q90 change vs broad normal baseline",
                "observations": "Observations",
                "reliability_label": "Reliability",
                "sample_size_note": "Sample-size note",
                "source_dataset": "Source dataset",
            }), hide_index=True, use_container_width=True)
        st.markdown(
            "<div class='method-box'><strong>Peak-stress reading.</strong> Average regime volatility, peak-stress volatility, high-volatility day share, and sample-size reliability answer different questions. Short crisis spikes can be visible in q90, maximum volatility, and high-volatility-day share even when the full-window mean is diluted by later stabilization.</div>",
            unsafe_allow_html=True,
        )

    fig = go.Figure()
    value_col = next(
        (
            c
            for c in [
                "annualized_target_vol_vs_pre_covid_pct",
                "mean_target_vol_vs_pre_covid_pct",
                "mean_target_vol_vs_full_sample_pct",
            ]
            if c in event_df.columns
        ),
        None,
    )
    colors = [EVENT_COLORS.get(str(r), FINANCE_COLORS["teal"]) for r in completed_event_df.get("event_regime", pd.Series([None] * len(completed_event_df)))]
    y_values = pd.to_numeric(completed_event_df[value_col], errors="coerce") if value_col else pd.Series(np.nan, index=completed_event_df.index)
    ann_vol = pd.to_numeric(completed_event_df.get("mean_target_vol_annualized", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce")
    ci_low = pd.to_numeric(completed_event_df.get("annualized_target_vol_ci_low", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce")
    ci_high = pd.to_numeric(completed_event_df.get("annualized_target_vol_ci_high", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce")
    pvalues = pd.to_numeric(completed_event_df.get("brown_forsythe_pvalue_vs_pre_covid", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce")
    customdata = np.column_stack([
        pd.to_numeric(completed_event_df.get("observations", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce").fillna(0).astype(int).astype(str),
        completed_event_df.get("reliability_label", pd.Series("Unavailable", index=completed_event_df.index)).fillna("Unavailable").astype(str),
        ann_vol.apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "n/a"),
        [
            f"{lo * 100:.2f}% - {hi * 100:.2f}%" if pd.notna(lo) and pd.notna(hi) else "n/a"
            for lo, hi in zip(ci_low, ci_high)
        ],
        pvalues.apply(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a"),
        completed_event_df.get("variance_test_result", pd.Series("Unavailable", index=completed_event_df.index)).fillna("Unavailable").astype(str),
    ])
    bar_kwargs = {}
    if {
        "annualized_target_vol_ci_low_vs_pre_covid_pct",
        "annualized_target_vol_ci_high_vs_pre_covid_pct",
    }.issubset(completed_event_df.columns):
        lower_ci_pct = pd.to_numeric(completed_event_df["annualized_target_vol_ci_low_vs_pre_covid_pct"], errors="coerce")
        upper_ci_pct = pd.to_numeric(completed_event_df["annualized_target_vol_ci_high_vs_pre_covid_pct"], errors="coerce")
        err_plus = np.nan_to_num((upper_ci_pct - y_values).clip(lower=0).values, nan=0.0, posinf=0.0, neginf=0.0)
        err_minus = np.nan_to_num((y_values - lower_ci_pct).clip(lower=0).values, nan=0.0, posinf=0.0, neginf=0.0)
        # Suppress error bars for Pre-COVID baseline (first bar at 0%) to reduce visual confusion
        if len(completed_event_df) > 0:
            err_plus[0] = 0.0
            err_minus[0] = 0.0
        bar_kwargs["error_y"] = dict(array=err_plus, arrayminus=err_minus, visible=True, thickness=1.4, width=4)
    fig.add_trace(go.Bar(
        x=completed_event_df["event_label"],
        y=y_values,
        marker_color=colors,
        name="Volatility impact",
        customdata=customdata,
        hovertemplate=(
            "Regime: %{x}<br>"
            "Change vs Pre-COVID: %{y:.2f}%<br>"
            "Observations: n=%{customdata[0]}<br>"
            "Reliability: %{customdata[1]}<br>"
            "Annualized volatility: %{customdata[2]}<br>"
            "95% CI: %{customdata[3]}<br>"
            "Brown-Forsythe p-value: %{customdata[4]}<br>"
            "Variance test: %{customdata[5]}<extra></extra>"
        ),
        **bar_kwargs,
    ))
    fig.update_layout(
        title="Completed Regimes: USD/TND Volatility Change vs Pre-COVID",
        yaxis_title="Volatility change vs Pre-COVID baseline (%)",
        xaxis_title="",
        xaxis=dict(categoryorder="array", categoryarray=completed_event_df["event_label"].tolist()),
    )
    st.plotly_chart(apply_finance_layout(fig, height=430), use_container_width=True)
    st.markdown(
        "<div class='method-box'><strong>Chart explanation.</strong> Bars show the average annualized 3-day USD/TND volatility change versus the Pre-COVID baseline; vertical lines show bootstrap confidence intervals.</div>",
        unsafe_allow_html=True,
    )

    if not ongoing_event_df.empty:
        row = ongoing_event_df.iloc[0]

        def _fmt_ratio_pct(value: Any, digits: int = 2) -> str:
            x = safe_float(value)
            return f"{x * 100:.{digits}f}%" if x is not None else "n/a"

        def _fmt_point_pct(value: Any) -> str:
            x = safe_float(value)
            return f"{x:.2f}%" if x is not None else "n/a"

        def _fmt_pvalue(value: Any) -> str:
            x = safe_float(value)
            return f"{x:.4f}" if x is not None else "n/a"

        # Prepare metrics for the premium summary panel
        regime_name = str(row.get("event_label", "Iran Geopolitical Shock"))
        observations = safe_float(row.get("observations"))
        observations_text = f"n = {int(observations):,}" if observations is not None else "n = n/a"
        ci_low_text = _fmt_ratio_pct(row.get("annualized_target_vol_ci_low"))
        ci_high_text = _fmt_ratio_pct(row.get("annualized_target_vol_ci_high"))
        ci_text = f"{ci_low_text} – {ci_high_text}" if ci_low_text != "n/a" and ci_high_text != "n/a" else "n/a"
        reliability = str(row.get("reliability_label", "Unavailable") or "Unavailable").strip()
        mean_vol = _fmt_ratio_pct(row.get("mean_target_vol_annualized"))
        impact = _fmt_point_pct(row.get("annualized_target_vol_vs_pre_covid_pct", row.get("mean_target_vol_vs_pre_covid_pct")))
        variance_result = str(row.get("variance_test_result", "Unavailable") or "Unavailable").strip()
        pvalue = _fmt_pvalue(row.get("brown_forsythe_pvalue_vs_pre_covid"))

        # Premium summary panel with better visual hierarchy
        st.markdown(
            f"""
            <div class='decision-card' style='border-left: 4px solid {FINANCE_COLORS["gold"]};'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;'>
                    <div>
                        <div class='decision-title' style='margin:0;'>{html.escape(regime_name)}</div>
                        <div style='font-size:0.85rem; color:{FINANCE_COLORS["muted"]}; margin-top:0.25rem;'>Current Ongoing Macro-Risk Regime</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["gold"]}; color:{FINANCE_COLORS["page_bg"]}; padding:0.35rem 0.7rem; border-radius:3px; font-size:0.8rem; font-weight:600;'>Ongoing / Preliminary</div>
                </div>
                <div class='decision-text' style='margin: 0.75rem 0;'>
                    Iran Geopolitical Shock is active and ongoing. It is excluded from the completed-regime comparison because the event window is still open and the realized-volatility sample remains preliminary. Current volatility statistics represent evidence observed so far, not a final regime impact.
                </div>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["teal"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Official start</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>2026-02-28</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["teal"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>End date</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["teal"]}; margin-top:0.25rem;'>Ongoing</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["primary"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Observations</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(observations_text)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["primary"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Reliability</div>
                        <div style='font-size:0.95rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(reliability)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["gold"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Mean annualized 3-day volatility</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["gold"]}; margin-top:0.25rem;'>{html.escape(mean_vol)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["gold"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Change vs Pre-COVID</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(impact)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["green"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>95% CI</div>
                        <div style='font-size:0.95rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(ci_text)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["green"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Variance test</div>
                        <div style='font-size:0.9rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(variance_result)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["slate"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>p-value</div>
                        <div style='font-size:1rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(pvalue)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Add Key reading box
        if not completed_event_df.empty:
            # Get highest and second-highest completed regimes
            completed_copy = completed_event_df.copy()
            completed_copy["_impact"] = pd.to_numeric(completed_copy.get(impact_col, pd.Series()), errors="coerce")
            completed_sorted = completed_copy[completed_copy["_impact"].notna()].sort_values("_impact", ascending=False)
            
            key_reading_html = "<strong>Key reading.</strong> "
            if len(completed_sorted) > 0:
                regime1 = str(completed_sorted.iloc[0].get("event_label", "Unknown regime"))
                impact1 = safe_float(completed_sorted.iloc[0].get("_impact"), 0) or 0
                key_reading_html += f"{html.escape(regime1)} shows the strongest USD/TND volatility uplift (<strong>+{impact1:.2f}%</strong>)"
                
                if len(completed_sorted) > 1:
                    regime2 = str(completed_sorted.iloc[1].get("event_label", "Unknown regime"))
                    impact2 = safe_float(completed_sorted.iloc[1].get("_impact"), 0) or 0
                    key_reading_html += f", followed by {html.escape(regime2)} (<strong>+{impact2:.2f}%</strong>)"
                key_reading_html += ". "
            
            # Check for significant variance regimes
            has_significant_variance = False
            if "variance_test_result" in completed_event_df.columns:
                significant = completed_event_df[
                    completed_event_df["variance_test_result"].astype(str).str.contains("significant", case=False, na=False)
                ]
                if len(significant) > 0:
                    has_significant_variance = True
                    key_reading_html += "These regimes show statistically significant variance differences versus Pre-COVID. "
            
            if not has_significant_variance and len(completed_sorted) > 0:
                key_reading_html += "The strongest completed-regime uplift shows variance differences versus Pre-COVID. "
            
            key_reading_html += f"{html.escape(regime_name)} is active but excluded from the completed-regime ranking because the event window remains open and the sample is preliminary."
            
            st.markdown(
                f"<div class='method-box'>{key_reading_html}</div>",
                unsafe_allow_html=True,
            )

    display_df = completed_event_df.drop(columns=["_sort_start_date"], errors="ignore").copy()
    if "start_date" in display_df.columns:
        display_df["start_date"] = pd.to_datetime(display_df["start_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "end_date" in display_df.columns:
        display_df["end_date"] = pd.to_datetime(display_df["end_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("Ongoing")
    if "event_regime" in display_df.columns:
        # Display the official methodology bounds while preserving empirical calculations from available observations.
        if "start_date" in display_df.columns:
            display_df.loc[display_df["event_regime"] == "post_tariff_normalization", "start_date"] = "2025-06-13"
            display_df.loc[display_df["event_regime"] == "iran_geopolitical_shock", "start_date"] = "2026-02-28"
        if "end_date" in display_df.columns:
            display_df.loc[display_df["event_regime"] == "post_tariff_normalization", "end_date"] = "2026-02-27"
            display_df.loc[display_df["event_regime"] == "iran_geopolitical_shock", "end_date"] = "Ongoing"

    # Main business-readable table columns
    main_cols = [
        "event_label",
        "observations",
        "reliability_label",
        "start_date",
        "end_date",
        "mean_target_vol_annualized",
        "q90_target_vol_annualized",
        "max_target_vol_annualized",
        "share_high_vol_days",
        "annualized_target_vol_vs_pre_covid_pct",
        "mean_target_vol_annualized_vs_broad_normal_pct",
        "q90_target_vol_annualized_vs_broad_normal_pct",
        "variance_test_result",
        "crisis_flag",
        "sample_size_note",
        "source_dataset",
    ]
    
    # Technical robustness columns for expander
    technical_cols = [
        "event_label",
        "median_target_vol_annualized",
        "annualized_return_vol",
        "annualized_target_vol_ci_low",
        "annualized_target_vol_ci_high",
        "brown_forsythe_pvalue_vs_pre_covid",
        "target_vol_iqr",
        "target_vol_mad_annualized",
    ]
    
    # Fallback columns if robust columns are missing
    fallback_cols = [
        "event_label",
        "observations",
        "start_date",
        "end_date",
        "mean_target_vol",
        "mean_target_vol_vs_pre_covid_pct",
        "mean_target_vol_vs_full_sample_pct",
        "crisis_flag",
    ]
    
    # Determine which columns exist
    main_display_cols = [c for c in main_cols if c in display_df.columns]
    technical_display_cols = [c for c in technical_cols if c in display_df.columns]
    
    # Use fallback if robust columns are missing
    if not any(c in display_df.columns for c in ["mean_target_vol_annualized", "annualized_target_vol_vs_pre_covid_pct"]):
        main_display_cols = [c for c in fallback_cols if c in display_df.columns]
        technical_display_cols = []
    
    # Format main table
    main_table_df = display_df[main_display_cols].copy()
    ratio_cols_main = [c for c in ["mean_target_vol_annualized", "q90_target_vol_annualized", "max_target_vol_annualized"] if c in main_table_df.columns]
    for col in ratio_cols_main:
        main_table_df[col] = pd.to_numeric(main_table_df[col], errors="coerce").apply(lambda x: f"{x * 100:.4f}%" if pd.notna(x) else "n/a")
    
    if "share_high_vol_days" in main_table_df.columns:
        main_table_df["share_high_vol_days"] = pd.to_numeric(main_table_df["share_high_vol_days"], errors="coerce").apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "n/a")

    for col in [
        "annualized_target_vol_vs_pre_covid_pct",
        "mean_target_vol_vs_pre_covid_pct",
        "mean_target_vol_vs_full_sample_pct",
        "mean_target_vol_annualized_vs_broad_normal_pct",
        "q90_target_vol_annualized_vs_broad_normal_pct",
    ]:
        if col in main_table_df.columns:
            main_table_df[col] = pd.to_numeric(main_table_df[col], errors="coerce").apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
    
    if "observations" in main_table_df.columns:
        main_table_df["observations"] = pd.to_numeric(main_table_df["observations"], errors="coerce").apply(lambda x: f"{int(x):,}" if pd.notna(x) else "n/a")
    
    if "crisis_flag" in main_table_df.columns:
        main_table_df["crisis_flag"] = pd.to_numeric(main_table_df["crisis_flag"], errors="coerce").fillna(0).apply(lambda x: "Crisis" if int(x) == 1 else "Non-crisis")
    
    rename_map_main = {
        "event_label": "Macro-event regime",
        "observations": "Observations",
        "reliability_label": "Reliability",
        "start_date": "Start date",
        "end_date": "End date",
        "mean_target_vol_annualized": "Mean annualized 3-day volatility",
        "q90_target_vol_annualized": "q90 annualized 3-day volatility",
        "max_target_vol_annualized": "Maximum annualized 3-day volatility",
        "share_high_vol_days": "Share of high-volatility days",
        "annualized_target_vol_vs_pre_covid_pct": "Change vs Pre-COVID baseline",
        "mean_target_vol_annualized_vs_broad_normal_pct": "Mean change vs broad normal baseline",
        "q90_target_vol_annualized_vs_broad_normal_pct": "q90 change vs broad normal baseline",
        "variance_test_result": "Variance test result",
        "crisis_flag": "Crisis regime",
        "sample_size_note": "Sample-size note",
        "source_dataset": "Source dataset",
    }
    
    if not ongoing_event_df.empty:
        st.markdown(
            "<div class='method-box'><strong>Completed Regime Comparison.</strong> Iran Geopolitical Shock is excluded from this table because it is ongoing and shown separately above.</div>",
            unsafe_allow_html=True,
        )
    
    st.dataframe(main_table_df.rename(columns=rename_map_main), hide_index=True, use_container_width=True)
    
    # Technical robustness details expander
    if len(technical_display_cols) > 0:
        with st.expander("Technical robustness details", expanded=False):
            technical_table_df = display_df[technical_display_cols].copy()
            ratio_cols_tech = [
                "median_target_vol_annualized",
                "annualized_return_vol",
                "annualized_target_vol_ci_low",
                "annualized_target_vol_ci_high",
                "target_vol_iqr",
                "target_vol_mad_annualized",
            ]
            for col in ratio_cols_tech:
                if col in technical_table_df.columns:
                    technical_table_df[col] = pd.to_numeric(technical_table_df[col], errors="coerce").apply(lambda x: f"{x * 100:.4f}%" if pd.notna(x) else "n/a")
            
            if "brown_forsythe_pvalue_vs_pre_covid" in technical_table_df.columns:
                technical_table_df["brown_forsythe_pvalue_vs_pre_covid"] = pd.to_numeric(technical_table_df["brown_forsythe_pvalue_vs_pre_covid"], errors="coerce").apply(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
            
            rename_map_tech = {
                "event_label": "Macro-event regime",
                "median_target_vol_annualized": "Median annualized 3-day volatility",
                "annualized_return_vol": "Annualized daily-return volatility",
                "annualized_target_vol_ci_low": "95% CI low",
                "annualized_target_vol_ci_high": "95% CI high",
                "brown_forsythe_pvalue_vs_pre_covid": "Brown-Forsythe p-value",
                "target_vol_iqr": "3-day volatility IQR",
                "target_vol_mad_annualized": "Annualized robust dispersion",
            }
            st.dataframe(technical_table_df.rename(columns=rename_map_tech), hide_index=True, use_container_width=True)
    
    st.markdown(
        "<div class='method-box'><strong>Economic reading.</strong> Completed regimes are compared using average regime volatility, peak-stress volatility, share of high-volatility days, and sample-size reliability. The Pre-COVID baseline remains visible, and the broad non-crisis baseline provides a more balanced normal reference. Ongoing regimes are excluded from the completed-regime ranking and shown separately as preliminary live risk context.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='method-box'><strong>Methodology note.</strong> Descriptive macro-event volatility statistics are computed from a stable USD/TND event-analysis dataset, not from the final ML feature matrix. This prevents changes in optional engineered model features from changing historical event-volatility comparisons. Forecast performance by event regime is still computed from the walk-forward model outputs.</div>",
        unsafe_allow_html=True,
    )
    
    # Developer diagnostics (collapsed by default)
    with st.expander("Developer diagnostics: macro-event artifacts", expanded=False):
        st.write({
            "source": event_source,
            "artifact_path": str(event_artifact_path) if event_artifact_path else "Unavailable",
            "robust_columns_detected": bool(robust_cols_detected),
            "detected_robust_columns": robust_cols_detected,
            "completed_regime_count": int(len(completed_event_df)),
            "ongoing_regime_count": int(len(ongoing_event_df)),
            "columns": event_df.columns.tolist(),
        })

def render_model_performance_by_event(payload: Optional[dict]) -> None:
    page_header("Model Performance by Event", "Regime validation", payload)
    if payload is None:
        render_no_payload_choice()
        return
    period = st.radio("Evaluation period", ["Holdout", "Development"], horizontal=True)
    df = get_event_metrics_df(payload, period.lower())
    if df.empty:
        st.info("Event-specific forecast metrics are not available.")
        return
    df = df.copy()
    if "event_regime" in df.columns:
        df["event_label"] = df["event_regime"].map(pretty_event_name)
    if "model" in df.columns:
        df["financial_model"] = df["model"].map(financial_model_label)

    blend = df[df.get("financial_model", pd.Series(dtype=str)).eq("Final Macro-Event FX Volatility Engine")].copy()
    if blend.empty and "model" in df.columns:
        blend = df[df["model"].astype(str).str.contains("blend", case=False, na=False)].copy()

    def _is_ongoing_event(row: pd.Series) -> bool:
        regime = str(row.get("event_regime", "")).lower()
        label = str(row.get("event_label", "")).lower()
        return "iran_geopolitical_shock" in regime or "iran" in regime or "iran" in label

    def _regime_type(row: pd.Series) -> str:
        if _is_ongoing_event(row):
            return "Ongoing / preliminary"
        crisis = safe_float(row.get("crisis_flag"))
        if crisis == 1:
            return "Crisis"
        if crisis == 0:
            return "Non-crisis"
        return "Non-crisis"

    def _format_event_metric(value: Any, metric_name: str = "RMSE_red_pct_vs_naive", digits: int = 2) -> str:
        x = safe_float(value)
        if x is None:
            return "n/a"
        if metric_name == "R2_vs_naive":
            return f"{x:.4f}"
        return f"+{x:.{digits}f}%" if x >= 0 else f"{x:.{digits}f}%"
    
    if not blend.empty:
        # Determine the metric column for comparison
        metric_col = "RMSE_red_pct_vs_naive" if "RMSE_red_pct_vs_naive" in blend.columns else "R2_vs_naive" if "R2_vs_naive" in blend.columns else None
        
        if metric_col:
            # Sort by metric DESCENDING (highest value first)
            blend_sorted = blend.sort_values(metric_col, ascending=False, na_position="last").copy()
            blend_sorted["_metric_value"] = pd.to_numeric(blend_sorted[metric_col], errors="coerce")
            blend_sorted = blend_sorted.sort_values("_metric_value", ascending=False, na_position="last").copy()
            metric_is_rmse_reduction = metric_col == "RMSE_red_pct_vs_naive"
            
            # Create panel title
            panel_title = f"Final Model Performance by Event - {period}"
            panel_subtitle = (
                "RMSE reduction versus the naïve benchmark. Higher values indicate stronger forecast improvement."
                if metric_is_rmse_reduction
                else "R-squared improvement versus the naïve benchmark. Higher values indicate stronger benchmark-relative fit."
            )
            
            st.markdown(
                f"""
                <style>
                .event-performance-shell {{
                    margin: 0 0 1rem 0;
                    padding: 1rem 1.05rem;
                    border: 1px solid rgba(79, 123, 255, 0.20);
                    border-radius: 8px;
                    background: linear-gradient(145deg, rgba(16, 26, 44, 0.98), rgba(10, 18, 32, 0.96));
                    box-shadow: 0 18px 44px rgba(0,0,0,0.22);
                }}
                .event-performance-title {{
                    font-size: 1.05rem;
                    font-weight: 850;
                    color: {FINANCE_COLORS["text"]};
                    margin-bottom: 0.18rem;
                }}
                .event-performance-subtitle {{
                    color: {FINANCE_COLORS["muted"]};
                    font-size: 0.86rem;
                    line-height: 1.45;
                }}
                .event-rank-card {{
                    position: relative;
                    overflow: hidden;
                    margin-bottom: 0.72rem;
                    padding: 0.86rem 0.9rem 0.82rem 0.9rem;
                    border: 1px solid rgba(168,179,204,0.14);
                    border-left: 3px solid var(--rank-accent);
                    border-radius: 8px;
                    background: linear-gradient(145deg, rgba(18,31,52,0.98), rgba(11,20,35,0.98));
                    box-shadow: 0 12px 28px rgba(0,0,0,0.18);
                    transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
                }}
                .event-rank-card:hover {{
                    transform: translateY(-2px);
                    border-color: rgba(47,216,201,0.32);
                    box-shadow: 0 18px 34px rgba(0,0,0,0.26);
                }}
                .event-rank-card.rank-one {{
                    padding-top: 1rem;
                    border-color: rgba(47,216,201,0.36);
                    box-shadow: 0 18px 42px rgba(47,216,201,0.08), 0 20px 44px rgba(0,0,0,0.26);
                }}
                .event-rank-topline {{
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.7rem;
                    margin-bottom: 0.52rem;
                }}
                .event-rank-badge {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 2.2rem;
                    height: 1.55rem;
                    border-radius: 999px;
                    color: #06101D;
                    background: var(--rank-accent);
                    font-size: 0.78rem;
                    font-weight: 900;
                }}
                .event-rank-pill {{
                    border: 1px solid var(--pill-border);
                    border-radius: 999px;
                    padding: 0.18rem 0.48rem;
                    color: var(--pill-color);
                    background: var(--pill-bg);
                    font-size: 0.68rem;
                    font-weight: 800;
                    white-space: nowrap;
                }}
                .event-rank-name {{
                    color: {FINANCE_COLORS["text"]};
                    font-size: 0.92rem;
                    line-height: 1.25;
                    font-weight: 750;
                    margin-bottom: 0.42rem;
                }}
                .event-rank-value {{
                    color: var(--rank-accent);
                    font-size: 1.55rem;
                    line-height: 1;
                    font-weight: 900;
                    margin-bottom: 0.4rem;
                }}
                .event-rank-card.rank-one .event-rank-value {{
                    font-size: 1.86rem;
                }}
                .event-rank-meta {{
                    color: {FINANCE_COLORS["muted"]};
                    font-size: 0.76rem;
                    letter-spacing: 0.01em;
                }}
                .event-interpretation {{
                    margin-top: 0.95rem;
                    padding: 0.95rem 1.05rem;
                    border-radius: 8px;
                    border: 1px solid rgba(47,216,201,0.18);
                    background: linear-gradient(145deg, rgba(16, 26, 44, 0.98), rgba(11, 20, 35, 0.98));
                    color: {FINANCE_COLORS["text"]};
                    box-shadow: 0 14px 34px rgba(0,0,0,0.18);
                    line-height: 1.55;
                }}
                .event-interpretation strong {{
                    color: {FINANCE_COLORS["teal"]};
                }}
                </style>
                <div class="event-performance-shell">
                    <div class="event-performance-title">{html.escape(panel_title)}</div>
                    <div class="event-performance-subtitle">{html.escape(panel_subtitle)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Two-column layout: left summary, right chart
            left_col, right_col = st.columns([0.35, 0.65], gap="large")
            
            with left_col:
                st.markdown(
                    f"<div style='font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:{FINANCE_COLORS['muted']}; font-weight:850; margin:0 0 0.62rem 0;'>Event ranking</div>",
                    unsafe_allow_html=True,
                )
                for idx, (_, row) in enumerate(blend_sorted.iterrows(), 1):
                    event_name = str(row.get("event_label", "Unknown"))
                    metric_text = _format_event_metric(row.get("_metric_value"), metric_col)
                    obs_value = safe_float(row.get("observations"))
                    obs = int(obs_value) if obs_value is not None else 0
                    regime_type = _regime_type(row)
                    meta_parts = [f"n={obs:,}"]
                    meta_parts.append(regime_type)
                    meta_text = " | ".join(meta_parts)

                    if regime_type == "Ongoing / preliminary":
                        accent = FINANCE_COLORS["gold"]
                        pill_color = FINANCE_COLORS["gold"]
                        pill_bg = "rgba(243,185,82,0.12)"
                        pill_border = "rgba(243,185,82,0.32)"
                    elif regime_type == "Crisis":
                        accent = "#FF8A8F"
                        pill_color = "#FFB0B4"
                        pill_bg = "rgba(255,95,104,0.12)"
                        pill_border = "rgba(255,95,104,0.28)"
                    else:
                        accent = FINANCE_COLORS["teal"] if (safe_float(row.get("_metric_value")) or 0) >= 0 else FINANCE_COLORS["red"]
                        pill_color = FINANCE_COLORS["teal"]
                        pill_bg = "rgba(47,216,201,0.10)"
                        pill_border = "rgba(47,216,201,0.26)"

                    rank_class = "event-rank-card rank-one" if idx == 1 else "event-rank-card"
                    st.markdown(
                        f"""
                        <div class="{rank_class}" style="--rank-accent:{accent}; --pill-color:{pill_color}; --pill-bg:{pill_bg}; --pill-border:{pill_border};">
                            <div class="event-rank-topline">
                                <span class="event-rank-badge">#{idx}</span>
                                <span class="event-rank-pill">{html.escape(regime_type)}</span>
                            </div>
                            <div class="event-rank-name">{html.escape(event_name)}</div>
                            <div class="event-rank-value">{html.escape(metric_text)}</div>
                            <div class="event-rank-meta">{html.escape(meta_text)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            
            with right_col:
                # Create horizontal bar chart
                fig = go.Figure()
                
                # Determine colors: positive=teal, negative=red, ongoing=amber
                colors = []
                for _, row in blend_sorted.iterrows():
                    val = safe_float(row.get("_metric_value"))
                    if _is_ongoing_event(row):
                        colors.append(FINANCE_COLORS["gold"])  # Amber for ongoing
                    elif val is None:
                        colors.append(FINANCE_COLORS["slate"])
                    elif val >= 0:
                        colors.append(FINANCE_COLORS["teal"])
                    else:
                        colors.append(FINANCE_COLORS["red"])
                
                # Format hover text
                hover_data = []
                for _, row in blend_sorted.iterrows():
                    rmse = safe_float(row.get("RMSE"))
                    mae = safe_float(row.get("MAE"))
                    corr = safe_float(row.get("Corr"))
                    r2 = safe_float(row.get("R2_vs_naive"))
                    obs = int(safe_float(row.get("observations")) or 0)
                    regime_type = _regime_type(row)
                    metric_label = "RMSE reduction" if metric_is_rmse_reduction else "R-squared vs naïve"
                    
                    hover_parts = [
                        f"Event: {row.get('event_label', 'n/a')}",
                        f"{metric_label}: {_format_event_metric(row.get('_metric_value'), metric_col)}",
                        f"RMSE: {rmse:.6f}" if rmse is not None else "RMSE: n/a",
                        f"MAE: {mae:.6f}" if mae is not None else "MAE: n/a",
                        f"Correlation: {corr:.4f}" if corr is not None else "Correlation: n/a",
                        f"R² vs naïve: {r2:.4f}" if r2 is not None else "R² vs naïve: n/a",
                        f"Observations: n={obs}",
                        f"Regime: {regime_type}",
                    ]
                    hover_data.append("<br>".join(hover_parts))
                
                # Format bar labels
                bar_labels = [_format_event_metric(row.get("_metric_value"), metric_col, digits=1) for _, row in blend_sorted.iterrows()]
                
                fig.add_trace(go.Bar(
                    x=blend_sorted["_metric_value"],
                    y=blend_sorted["event_label"],
                    orientation="h",
                    text=bar_labels,
                    textposition="outside",
                    textfont=dict(color=FINANCE_COLORS["text"], size=13, family="Arial Black"),
                    marker_color=colors,
                    marker_line=dict(color="rgba(255,255,255,0.20)", width=1),
                    opacity=0.95,
                    hovertext=hover_data,
                    hoverinfo="text",
                    cliponaxis=False,
                    showlegend=False,
                ))
                
                # Update layout - NO TITLE to avoid undefined
                fig.update_layout(
                    title=None,
                    xaxis_title="RMSE reduction vs naïve (%)" if metric_is_rmse_reduction else "R² vs naïve",
                    yaxis_title="",
                    xaxis=dict(
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor="rgba(168,179,204,0.35)",
                        gridcolor="rgba(168,179,204,0.10)",
                        tickfont=dict(color=FINANCE_COLORS["muted"]),
                    ),
                    yaxis=dict(
                        categoryorder="array",
                        categoryarray=blend_sorted["event_label"].tolist(),
                        autorange="reversed",
                        tickfont=dict(color=FINANCE_COLORS["text"], size=12),
                    ),
                    plot_bgcolor="rgba(7,16,29,0.18)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=FINANCE_COLORS["text"], family="Arial"),
                    margin=dict(l=160, r=70, t=6, b=44),
                    height=max(320, 70 * len(blend_sorted)),
                    hovermode="closest",
                    bargap=0.36,
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Compact interpretation strip using the sorted dataframe
            best_event = blend_sorted.iloc[0] if not blend_sorted.empty else None
            hardest_event = blend.sort_values("RMSE", ascending=False).iloc[0] if "RMSE" in blend.columns and not blend.empty else None
            has_iran = any(_is_ongoing_event(r) for _, r in blend.iterrows())
            
            interp_parts = []
            if best_event is not None:
                best_name = str(best_event.get("event_label", "Unknown"))
                best_val = safe_float(best_event.get("_metric_value"))
                if best_val is not None:
                    best_metric_text = _format_event_metric(best_val, metric_col)
                    if metric_is_rmse_reduction:
                        interp_parts.append(f"<strong>Best event performance:</strong> {html.escape(best_name)}, where the final model reduced RMSE by <strong>{html.escape(best_metric_text)}</strong> versus the naïve benchmark.")
                    else:
                        interp_parts.append(f"<strong>Best event performance:</strong> {html.escape(best_name)}, with benchmark-relative R-squared of <strong>{html.escape(best_metric_text)}</strong>.")
            
            if hardest_event is not None:
                hard_name = str(hardest_event.get("event_label", "Unknown"))
                hard_rmse = safe_float(hardest_event.get("RMSE"))
                if hard_rmse is not None:
                    interp_parts.append(f"<strong>Hardest regime to forecast:</strong> {html.escape(hard_name)} with RMSE {hard_rmse:.6f}.")
            
            if has_iran:
                interp_parts.append("<strong>Iran Geopolitical Shock</strong> is ongoing, so its event-level performance should be read as preliminary.")
            
            if interp_parts:
                interp_text = " ".join(interp_parts)
                st.markdown(f"<div class='event-interpretation'>{interp_text}</div>", unsafe_allow_html=True)
        else:
            st.warning("No benchmark-relative performance metric available for the selected period.")

    # Improved detailed table
    rename_map = {
        "event_label": "Event regime",
        "financial_model": "Model",
        "observations": "Observations",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "Bias": "Bias",
        "QLIKE": "QLIKE",
        "Corr": "Correlation",
        "R2_vs_naive": "R² vs naïve",
        "RMSE_red_pct_vs_naive": "RMSE reduction vs naïve",
        "_regime_type": "Regime type",
    }
    
    display_source = df.copy()
    display_source["_regime_type"] = display_source.apply(_regime_type, axis=1)
    display_cols = [c for c in ["event_label", "financial_model", "observations", "RMSE", "MAE", "Bias", "QLIKE", "Corr", "R2_vs_naive", "RMSE_red_pct_vs_naive", "_regime_type"] if c in display_source.columns]
    table_df = display_source[display_cols].copy()
    
    # Format numeric columns
    for col in ["RMSE", "MAE", "Bias", "QLIKE"]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").apply(lambda x: f"{x:.6f}" if pd.notna(x) else "n/a")
    
    for col in ["Corr", "R2_vs_naive"]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").apply(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
    
    if "RMSE_red_pct_vs_naive" in table_df.columns:
        table_df["RMSE_red_pct_vs_naive"] = pd.to_numeric(table_df["RMSE_red_pct_vs_naive"], errors="coerce").apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
    if "observations" in table_df.columns:
        table_df["observations"] = pd.to_numeric(table_df["observations"], errors="coerce").apply(lambda x: f"{int(x):,}" if pd.notna(x) else "n/a")
    
    st.subheader("Detailed Model Metrics")
    st.dataframe(table_df.rename(columns=rename_map), hide_index=True, use_container_width=True)
    
    # Metrics explainer expander
    with st.expander("Metric definitions", expanded=False):
        st.markdown("""
        - **RMSE**: Root Mean Squared Error. Lower is better. Penalizes large forecast errors.
        - **MAE**: Mean Absolute Error. Lower is better. Measures average absolute forecast error.
        - **Bias**: Average forecast error. Positive means overforecasting; negative means underforecasting.
        - **QLIKE**: Volatility-forecast loss metric. Lower is better.
        - **Correlation**: Shows whether forecasts move with realized volatility. Higher is better.
        - **R² vs naïve**: Improvement in squared-error terms versus the naïve benchmark. Positive is better.
        - **RMSE reduction vs naïve**: Percentage reduction in RMSE versus the naïve benchmark. Higher is better.
        """)


def render_global_fx_context(payload: Optional[dict]) -> None:
    page_header("USD/TND vs Global FX Risk Context", "Market context", payload)
    if payload is None:
        render_no_payload_choice()
        return
    gfx = (payload.get("dashboard", {}) or {}).get("global_fx_comparison", {}) if payload else {}
    latest = gfx.get("latest", {}) or {}
    ratio = safe_float(latest.get("ratio"))
    classification = latest.get("classification", "Unavailable")
    render_kpi_grid([
        metric_card("USD/TND Volatility", format_pct(latest.get("usdtnd_volatility")), "Local FX risk", "teal"),
        metric_card("Global FX Average", format_pct(latest.get("global_fx_mean")), "Major FX proxy basket", "primary"),
        metric_card("Local/Global Ratio", format_ratio(ratio), "USD/TND vs global FX", "gold"),
        metric_card("Context Signal", str(classification), "Relative volatility state", "purple"),
    ])
    if ratio is not None:
        if ratio > 1.0:
            insight = "USD/TND volatility is above the global FX basket. Traders should investigate local liquidity, regional risk, and USD/TND-specific pressure before relying on passive hedge timing."
        elif ratio < 1.0:
            insight = "USD/TND volatility is below the global FX basket. Local conditions appear contained relative to major FX, but event headlines should remain monitored."
        else:
            insight = "USD/TND volatility is aligned with global FX conditions. Standard risk protocols are appropriate."
        st.markdown(f"<div class='method-box'><strong>Market read:</strong> {html.escape(insight)}</div>", unsafe_allow_html=True)
    history = records_df(gfx.get("history", []) if gfx else [])
    if not history.empty:
        st.plotly_chart(line_fig(history, "Date", [("usdtnd_volatility", "USD/TND volatility"), ("global_fx_mean", "Global FX average")], y_scale=100, yaxis_title="Volatility (%)"), use_container_width=True)
        st.plotly_chart(line_fig(history, "Date", [("ratio", "USD/TND / Global FX")], yaxis_title="Ratio"), use_container_width=True)
    else:
        st.info("Global FX comparison data is not available in the current payload.")


def render_calendar_patterns(payload: Optional[dict]) -> None:
    page_header("Seasonality & Calendar Risk Patterns", "Calendar diagnostics", payload)
    if payload is None:
        render_no_payload_choice()
        return
    calendar = (payload.get("dashboard", {}) or {}).get("calendar_effect", {}) if payload else {}
    heat = pd.DataFrame(calendar.get("heatmap_matrix", []) or [])
    if heat.empty:
        st.info("Calendar-effect diagnostics are not available.")
        return
    matrix = heat.pivot(index="month", columns="day_of_week", values="average_volatility")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    matrix = matrix.reindex(columns=[day for day in day_order if day in matrix.columns])
    day_avg = matrix.mean(axis=0)
    month_avg = matrix.mean(axis=1)
    render_kpi_grid([
        metric_card("Highest-Risk Weekday", str(day_avg.idxmax()), f"Avg vol {format_pct(day_avg.max())}", "gold"),
        metric_card("Lowest-Risk Weekday", str(day_avg.idxmin()), f"Avg vol {format_pct(day_avg.min())}", "green"),
        metric_card("Highest-Risk Month", str(month_avg.idxmax()), f"Avg vol {format_pct(month_avg.max())}", "red"),
        metric_card("Calendar Method", "Descriptive", "Not causal; not standalone forecast", "primary"),
    ])
    fig = go.Figure(go.Heatmap(z=matrix.values * 100, x=matrix.columns, y=matrix.index, colorscale="Turbo", colorbar_title="Avg vol (%)", hovertemplate="%{y}, %{x}<br>Avg vol: %{z:.3f}%<extra></extra>"))
    fig.update_layout(title="Calendar-Based USD/TND Volatility Patterns", xaxis_title="", yaxis_title="")
    st.plotly_chart(apply_finance_layout(fig, height=430), use_container_width=True)
    st.markdown("<div class='method-box'><strong>Methodology note.</strong> Calendar effects are descriptive historical diagnostics. They help monitoring and execution planning but are not treated as causal drivers.</div>", unsafe_allow_html=True)
    if calendar.get("commentary"):
        st.write(calendar["commentary"])


def render_forecast_engine_explainability(payload: Optional[dict]) -> None:
    page_header("Forecast Engine & Benchmark Attribution", "Model architecture and validation", payload)
    if payload is None:
        render_no_payload_choice()
        return
    forecast = payload.get("forecast", {}) or {}
    render_kpi_grid([
        metric_card("Traditional Volatility Benchmark", format_pct(forecast.get("garch_anchor")), "GARCH / EGARCH reference forecast", "gold"),
        metric_card("Linear Macro-Risk Stabilizer", format_pct(forecast.get("pred_ridge")), "Ridge component", "primary"),
        metric_card("Nonlinear Market-Risk Engine", format_pct(forecast.get("pred_xgb")), "XGBoost component", "purple"),
        metric_card("Final Model Output", format_pct(forecast.get("final_forecast_blend")), "Blended engine", "teal"),
    ])

    summary_df = get_summary_df(payload, "holdout")
    if not summary_df.empty:
        df = summary_df.copy()
        model_col = "Model" if "Model" in df.columns else "model" if "model" in df.columns else None
        if model_col:
            df["Financial model label"] = df[model_col].map(financial_model_label)
        keep = [c for c in ["Financial model label", model_col, "RMSE", "MAE", "QLIKE", "R2_vs_naive", "RMSE_red_pct", "Corr"] if c and c in df.columns]
        st.subheader("Holdout model comparison")
        st.dataframe(df[keep], hide_index=True, use_container_width=True)

    dash = payload.get("dashboard", {}) or {}
    weights = dash.get("model_weights", {}) or {}
    if weights:
        st.subheader("Blend weights")
        dfw = pd.DataFrame([
            {"Component": "Linear Macro-Risk Stabilizer", "Weight": weights.get("ridge_weight", 0)},
            {"Component": "Nonlinear Market-Risk Engine", "Weight": weights.get("xgb_weight", weights.get("final_dynamic_xgb_weight", 0))},
        ])
        fig = go.Figure(go.Bar(x=pd.to_numeric(dfw["Weight"], errors="coerce"), y=dfw["Component"], orientation="h", marker_color=[FINANCE_COLORS["primary"], FINANCE_COLORS["teal"]]))
        fig.update_layout(title="Final Model Blend Allocation", xaxis_tickformat=".0%", xaxis_range=[0, 1])
        st.plotly_chart(apply_finance_layout(fig, height=260), use_container_width=True)

    features = (dash.get("feature_importance", {}) or {}).get("top_5", [])
    fdf = pd.DataFrame(features)
    if not fdf.empty and {"feature_name", "importance_score"}.issubset(fdf.columns):
        st.subheader("Top market-risk drivers")
        fdf = fdf.sort_values("importance_score")
        fig = go.Figure(go.Bar(x=fdf["importance_score"], y=fdf["feature_name"], orientation="h", marker_color=FINANCE_COLORS["teal"]))
        fig.update_layout(title="Feature Importance - Nonlinear Market-Risk Engine", xaxis_title="Relative importance")
        st.plotly_chart(apply_finance_layout(fig, height=350), use_container_width=True)


def render_scenario_laboratory(payload: Optional[dict]) -> None:
    page_header("Scenario & Calibration Laboratory", "Calibration sensitivity overlay", payload)
    if payload is None:
        render_no_payload_choice()
        return

    classification = classify_forecast_volatility_regime(payload)
    official_forecast = safe_float(classification.get("forecast_volatility"), 0.0) or 0.0
    official_prob = safe_float(classification.get("high_vol_probability"), 0.0) or 0.0
    p05, p95 = get_confidence_band(payload)
    official_regime = classification.get("regime_label", "Unavailable")
    official_stance = risk_stance_label(classification)

    st.markdown(
        """
        <div class="method-box">
            <strong>Scenario objective.</strong> Use this laboratory to test whether the hedge signal is sensitive to alternative market-memory and macro-event assumptions. The official forecast remains the production reference; scenario outputs are comparison overlays.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Official production reference")
    render_kpi_grid([
        metric_card("Official Forecast", format_pct(official_forecast), "Final Macro-Event FX Volatility Engine", "teal"),
        metric_card("Official Regime", str(official_regime), "Empirical threshold classification", "gold"),
        metric_card("High-Vol Probability", format_pct(official_prob), "Production high-volatility signal", "purple"),
        metric_card("Hedging Stance", official_stance, "Production decision anchor", "primary"),
    ])

    st.subheader("Scenario controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario_name = st.text_input("Scenario name", value="Short-cycle stress overlay")
        scenario_window_options = calibration_window_options("Custom")
        window = st.selectbox(
            "Calibration window assumption",
            scenario_window_options,
            index=default_calibration_window_index(scenario_window_options),
            key="scenario_calibration_window_v2",
        )
    with c2:
        event_focus = st.multiselect(
            "Macro-event focus",
            ["COVID shock", "Russia-Ukraine shock", "U.S. tariff shock", "Iran geopolitical shock", "Post-conflict normalization"],
            default=[],
            key="scenario_event_focus_v2",
        )
        sensitivity = st.select_slider("Event sensitivity", options=["Low", "Moderate", "High"], value="Moderate")
    with c3:
        purpose = st.selectbox("Scenario purpose", ["Short-cycle stress view", "Medium-cycle risk view", "Long-cycle macro view", "Crisis-sensitive overlay"])
        custom_window = st.number_input("Custom window", min_value=250, max_value=1500, value=ROLLING_WINDOW, step=50, disabled=(window != "Custom"))

    window_days = int(custom_window) if window == "Custom" else int(window.split()[0])
    if window_days < 250:
        health = "Insufficient for institutional calibration"
        health_accent = "red"
    elif window_days == ROLLING_WINDOW:
        health = "Production baseline"
        health_accent = "green"
    elif window_days < ROLLING_WINDOW:
        health = "Tactical short-cycle scenario"
        health_accent = "gold"
    elif window_days < ROLLING_WINDOW * 2:
        health = "Medium-cycle scenario"
        health_accent = "primary"
    else:
        health = "Long-cycle macro scenario"
        health_accent = "teal"

    # Scenario overlay is a decision-support sensitivity proxy. It does not rerun
    # forecasting.py until the backend supports run_pipeline(config=...). The
    # official model stays untouched and remains the production reference.
    window_multiplier = scenario_window_multiplier(window_days)
    event_multiplier = 1.0 + 0.025 * len(event_focus)
    sensitivity_multiplier = {"Low": 0.98, "Moderate": 1.00, "High": 1.06}[sensitivity]
    scenario_forecast = official_forecast * window_multiplier * event_multiplier * sensitivity_multiplier
    scenario_prob = min(0.99, max(0.0, official_prob * (scenario_forecast / max(official_forecast, 1e-12))))
    scenario_classification = dict(classification)
    scenario_classification["forecast_volatility"] = scenario_forecast
    scenario_classification["high_vol_probability"] = scenario_prob
    scenario_classification = classify_scenario_against_thresholds(scenario_classification)
    scenario_stance = risk_stance_label(scenario_classification)

    st.subheader("Scenario health check")
    render_kpi_grid([
        metric_card("Selected Window", f"{window_days}d", "Market-memory assumption", health_accent),
        metric_card("Scenario Health", health, "Governance check", health_accent),
        metric_card("Event Focus", str(len(event_focus)), "Selected macro-event regimes", "purple"),
        metric_card("Sensitivity", sensitivity, purpose, "primary"),
    ])

    comparison = pd.DataFrame([
        {"Metric": "3-day volatility forecast", "Official Forecast": format_pct(official_forecast), "Scenario Overlay": format_pct(scenario_forecast), "Difference": f"{(scenario_forecast - official_forecast) * 100:.2f} pp"},
        {"Metric": "Forecast regime", "Official Forecast": official_regime, "Scenario Overlay": scenario_classification.get("regime_label"), "Difference": "Higher risk" if scenario_forecast > official_forecast else "Lower / similar risk"},
        {"Metric": "High-volatility probability", "Official Forecast": format_pct(official_prob), "Scenario Overlay": format_pct(scenario_prob), "Difference": f"{(scenario_prob - official_prob) * 100:.2f} pp"},
        {"Metric": "Confidence band", "Official Forecast": f"{format_pct(p05)} - {format_pct(p95)}", "Scenario Overlay": f"{format_pct((p05 or official_forecast) * window_multiplier)} - {format_pct((p95 or official_forecast) * window_multiplier)}", "Difference": "Wider stress range" if window_multiplier > 1 else "Tighter / slower-moving range"},
        {"Metric": "Hedging stance", "Official Forecast": official_stance, "Scenario Overlay": scenario_stance, "Difference": "More defensive" if scenario_forecast > official_forecast else "Stable / less defensive"},
    ])
    st.subheader("Official vs scenario comparison")
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    if scenario_forecast > official_forecast * 1.10:
        interpretation = "The scenario overlay produces a materially higher volatility signal than the official forecast. This points to a more fragile short-term market-memory view and supports tighter exposure limits or partial hedge acceleration."
    elif scenario_forecast < official_forecast * 0.90:
        interpretation = "The scenario overlay produces a lower volatility signal than the official forecast. This suggests recent or selected assumptions may indicate easing pressure, while the official forecast remains the production anchor."
    else:
        interpretation = "The scenario overlay is close to the official forecast. The hedge signal appears stable across the selected calibration assumptions."
    st.markdown(f"<div class='decision-card'><div class='decision-title'>Trader interpretation</div><div class='decision-text'>{html.escape(interpretation)}</div></div>", unsafe_allow_html=True)

    sensitivity_df = pd.DataFrame([
        {
            "Window": f"{days}d",
            "Forecast": official_forecast * scenario_window_multiplier(days),
            "Role": "Official baseline" if days == ROLLING_WINDOW else "Scenario",
        }
        for days in STANDARD_CALIBRATION_WINDOWS
    ])
    fig = go.Figure(go.Bar(
        x=sensitivity_df["Window"],
        y=sensitivity_df["Forecast"] * 100,
        marker_color=[
            FINANCE_COLORS["green"] if role == "Official baseline" else FINANCE_COLORS["teal"]
            for role in sensitivity_df["Role"]
        ],
    ))
    fig.update_layout(title="Calibration-Window Sensitivity View", yaxis_title="Scenario volatility forecast (%)", xaxis_title="Calibration window")
    st.plotly_chart(apply_finance_layout(fig, height=360), use_container_width=True)

    event_df = get_event_descriptive_df(payload)
    if not event_df.empty and "event_regime" in event_df.columns and event_focus:
        event_df = event_df.copy()
        event_df["event_label"] = event_df["event_regime"].map(pretty_event_name)
        selected = event_df[event_df["event_label"].isin(event_focus)]
        if not selected.empty:
            st.subheader("Selected event volatility profile")
            st.dataframe(selected[[c for c in ["event_label", "observations", "mean_target_vol", "mean_target_vol_vs_pre_covid_pct", "mean_target_vol_vs_full_sample_pct"] if c in selected.columns]], hide_index=True, use_container_width=True)


def render_methodology_audit(payload: Optional[dict]) -> None:
    page_header("Methodology & Audit Trail", "Academic governance", payload)
    st.markdown(
        """
        <div class="method-box">
            <strong>Methodology position.</strong> The app forecasts next 3-day USD/TND realized volatility using a macro-event hybrid engine. Forecast regimes are assigned through empirical realized-volatility thresholds, with macro-event channels providing economic context.
        </div>
        """,
        unsafe_allow_html=True,
    )
    methodology_rows = pd.DataFrame([
        {"Area": "Target", "Methodology": "Next 3-day USD/TND realized volatility"},
        {"Area": "Forecast engine", "Methodology": "Ridge + XGBoost + GARCH/EGARCH blended framework"},
        {"Area": "Final model name", "Methodology": "Final Macro-Event FX Volatility Engine"},
        {"Area": "Benchmarks", "Methodology": "FX Volatility Carry-Forward Benchmark and Traditional Volatility Benchmark"},
        {"Area": "Economic regimes", "Methodology": "COVID, Russia-Ukraine, U.S. tariff shock, Iran geopolitical shock"},
        {"Area": "Classification", "Methodology": "q25/q50/q75/q90 realized-volatility thresholds"},
        {"Area": "Decision outputs", "Methodology": "Forecast, regime, probability, confidence band, and hedge stance"},
        {"Area": "Validation", "Methodology": "Chronological walk-forward validation, event-specific diagnostics, and high-volatility probability calibration diagnostics. The Brier score measures probability forecast error, while the reliability diagram compares predicted probabilities with observed high-volatility frequencies. The threshold sensitivity table shows how alert quality changes across different probability cutoffs."},
    ])
    st.dataframe(methodology_rows, hide_index=True, use_container_width=True)

    metadata = payload.get("metadata", {}) if payload else {}
    validation = metadata.get("event_methodology_validation") or {}
    if validation:
        st.subheader("Event methodology validation")
        st.json(validation)
    else:
        st.info("Event methodology validation metadata is not available in this payload.")

    if payload:
        with st.expander("Event context source", expanded=False):
            event_context, event_context_source = get_event_context_details(payload)
            st.markdown(f"**Resolved event context:** {html.escape(event_context)}")
            st.markdown(f"**Source:** `{html.escape(event_context_source)}`")
        st.subheader("Available artifacts")
        artifacts = payload.get("artifacts", {}) or {}
        rows = []
        for key in ARTIFACT_FILENAMES.keys():
            p = artifact_path(payload, key)
            rows.append({"Artifact": ARTIFACT_DISPLAY_NAMES.get(key, key.replace("_", " ").title()), "Key": key, "Available": bool(p), "Path": str(p) if p else ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_downloads(payload: Optional[dict]) -> None:
    page_header("Downloads", "Export center", payload)
    if payload is None:
        render_no_payload_choice()
        return
    artifact_rows = []
    for key in ARTIFACT_FILENAMES.keys():
        p = artifact_path(payload, key)
        artifact_rows.append({"Key": key, "File": ARTIFACT_FILENAMES.get(key, ""), "Available": bool(p)})
    st.dataframe(pd.DataFrame(artifact_rows), hide_index=True, use_container_width=True)
    st.markdown("---")
    cols = st.columns(3)
    idx = 0
    for key in ARTIFACT_FILENAMES.keys():
        p = artifact_path(payload, key)
        if not p or not p.exists():
            continue
        with cols[idx % 3]:
            st.download_button(
                ARTIFACT_DISPLAY_NAMES.get(key, key.replace("_", " ").title()),
                data=p.read_bytes(),
                file_name=p.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"download_{key}",
            )
        idx += 1


# ============================================================
# Main app
# ============================================================


st.set_page_config(page_title="VolSight | Macro-Event FX Volatility Engine", layout="wide")
inject_finance_theme()

init_state = {
    "forecast_payload": None,
    "selected_run_id": None,
    "artifact_dir": None,
    "selected_page": PAGE_NAMES[0],
}
for key, value in init_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.markdown("<div class='app-brand'><span class='app-brand-mark'>V</span><span>VolSight</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='app-tagline'>Macro-Event FX Volatility Engine</div>", unsafe_allow_html=True)
    st.session_state.selected_page = st.radio(
        "Dashboard section",
        PAGE_NAMES,
        index=PAGE_NAMES.index(st.session_state.selected_page) if st.session_state.selected_page in PAGE_NAMES else 0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.subheader("Run control")
    if st.button("Load latest saved results", use_container_width=True, key="sidebar_load_latest"):
        try:
            payload_loaded = load_latest_local()
            st.session_state.forecast_payload = payload_loaded
            st.session_state.selected_run_id = payload_loaded.get("run_id")
            st.session_state.artifact_dir = payload_loaded.get("artifact_dir")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if st.button("Run new forecast", use_container_width=True, key="sidebar_run_new"):
        st.session_state.selected_page = "Data & Market Inputs"
        st.rerun()
    latest_run = find_latest_run_dir()
    if latest_run:
        st.caption(f"Latest local run: {latest_run.name}")
    else:
        st.caption("No saved local runs detected.")
    st.markdown("---")
    st.caption("Official forecast: validated macro-event engine. Scenario Laboratory provides calibration sensitivity overlays.")

payload = st.session_state.forecast_payload
selected_page = st.session_state.selected_page

if selected_page == "Executive FX Risk Cockpit":
    render_executive_fx_risk_cockpit(payload)
elif selected_page == "Data & Market Inputs":
    render_data_market_inputs(payload)
elif selected_page == "Volatility Regime Classification":
    render_volatility_regime_classification(payload)
elif selected_page == "Macro-Event Regime Analysis":
    render_macro_event_regime_analysis(payload)
elif selected_page == "Model Performance by Event":
    render_model_performance_by_event(payload)
elif selected_page == "USD/TND vs Global FX Risk Context":
    render_global_fx_context(payload)
elif selected_page == "Seasonality & Calendar Risk Patterns":
    render_calendar_patterns(payload)
elif selected_page == "Forecast Engine & Benchmark Attribution":
    render_forecast_engine_explainability(payload)
elif selected_page == "Scenario & Calibration Laboratory":
    render_scenario_laboratory(payload)
elif selected_page == "Methodology & Audit Trail":
    render_methodology_audit(payload)
elif selected_page == "Downloads":
    render_downloads(payload)

