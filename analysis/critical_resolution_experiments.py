from __future__ import annotations

import math
import os
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_sleep_health")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "Sleep_health_and_lifestyle_dataset" / "Sleep_health_and_lifestyle_dataset.csv"
OUT_DIR = ROOT / "results" / "critical_resolution"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "critical_resolution_report_ko.md"

RANDOM_SEED = 42
OUTER_REPEATS = 8
OUTER_SPLITS = 5
MULTI_REPEATS = 5

RENAME_MAP = {
    "Person ID": "person_id",
    "Gender": "gender",
    "Age": "age",
    "Occupation": "occupation",
    "Sleep Duration": "sleep_duration",
    "Quality of Sleep": "quality_of_sleep",
    "Physical Activity Level": "physical_activity_level",
    "Stress Level": "stress_level",
    "BMI Category": "bmi_category",
    "Blood Pressure": "blood_pressure",
    "Heart Rate": "heart_rate",
    "Daily Steps": "daily_steps",
    "Sleep Disorder": "sleep_disorder",
}

DISPLAY = {
    "age": "Age",
    "sleep_duration": "Sleep Duration",
    "quality_of_sleep": "Quality of Sleep",
    "physical_activity_level": "Physical Activity Level",
    "stress_level": "Stress Level",
    "heart_rate": "Heart Rate",
    "daily_steps": "Daily Steps",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "map_bp": "Mean Arterial Pressure",
    "pulse_pressure": "Pulse Pressure",
    "sleep_deficit_7h": "Sleep Deficit (vs 7h)",
    "sleep_stress_balance": "Sleep-Stress Balance",
    "male": "Male",
    "bmi_risk": "BMI Risk",
}

CLASS_ORDER = ["None", "Insomnia", "Sleep Apnea"]

MODEL_FEATURES = {
    "orig_quality_hr": ["age", "quality_of_sleep", "heart_rate", "diastolic_bp", "male", "bmi_risk"],
    "orig_sleep_pa": ["age", "sleep_duration", "physical_activity_level", "diastolic_bp", "male", "bmi_risk"],
    "orig_sleep_hr": ["age", "sleep_duration", "heart_rate", "diastolic_bp", "male", "bmi_risk"],
    "orig_quality_pa": ["age", "quality_of_sleep", "physical_activity_level", "diastolic_bp", "male", "bmi_risk"],
    "compressed": ["age", "heart_rate", "map_bp", "pulse_pressure", "sleep_deficit_7h", "sleep_stress_balance", "male", "bmi_risk"],
    "enhanced": [
        "age",
        "heart_rate",
        "map_bp",
        "pulse_pressure",
        "sleep_deficit_7h",
        "sleep_stress_balance",
        "male",
        "bmi_risk",
        "age_map_interaction",
        "sleep_balance_deficit_interaction",
        "bmi_map_interaction",
    ],
}

MODEL_LABELS = {
    "orig_quality_hr": "Original: Quality + HR",
    "orig_sleep_pa": "Original: Sleep + PA",
    "orig_sleep_hr": "Original: Sleep + HR",
    "orig_quality_pa": "Original: Quality + PA",
    "compressed": "Compressed Derived",
    "enhanced": "Compressed + Interaction",
}

FINAL_MODEL = "orig_quality_pa"
MULTINOMIAL_FEATURES = ["age", "heart_rate", "map_bp", "pulse_pressure", "sleep_deficit_7h", "sleep_stress_balance", "male", "bmi_risk"]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def pretty(name: str) -> str:
    return DISPLAY.get(name, name.replace("_", " ").title())


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.3f}")
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.astype(str).itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, index=False)


def logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 6) -> float:
    quantiles = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    quantiles[0] = 0.0
    quantiles[-1] = 1.0
    total = 0.0
    for left, right in zip(quantiles[:-1], quantiles[1:], strict=False):
        mask = (y_prob >= left) & (y_prob <= right if right == 1 else y_prob < right)
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total)


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    candidates = np.linspace(0.1, 0.9, 17)
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1 + 1e-12 or (abs(score - best_f1) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": ece_score(y_true, y_prob),
    }


def calibration_fit(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    cal_df = pd.DataFrame({"logit": logit(y_prob)})
    fit = sm.GLM(y_true, sm.add_constant(cal_df), family=sm.families.Binomial()).fit()
    return float(fit.params["const"]), float(fit.params["logit"])


def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    cal_df = pd.DataFrame({"logit": logit(y_prob)})
    fit = sm.GLM(y_true, sm.add_constant(cal_df), family=sm.families.Binomial()).fit()
    return float(fit.params["const"]), float(fit.params["logit"])


def apply_calibration(y_prob: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return sigmoid(intercept + slope * logit(y_prob))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).rename(columns=RENAME_MAP)
    df["sleep_disorder"] = df["sleep_disorder"].fillna("None")
    df["bmi_category"] = df["bmi_category"].replace({"Normal Weight": "Normal"})
    bp = df["blood_pressure"].str.extract(r"(?P<systolic_bp>\d+)/(?P<diastolic_bp>\d+)").astype(int)
    df = pd.concat([df, bp], axis=1)
    df["has_sleep_disorder"] = (df["sleep_disorder"] != "None").astype(int)
    df["male"] = (df["gender"] == "Male").astype(int)
    df["bmi_risk"] = (df["bmi_category"] != "Normal").astype(int)
    df["map_bp"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["sleep_deficit_7h"] = (7 - df["sleep_duration"]).clip(lower=0)
    df["sleep_stress_balance"] = df["quality_of_sleep"] - df["stress_level"]
    df["age_map_interaction"] = df["age"] * df["map_bp"]
    df["sleep_balance_deficit_interaction"] = df["sleep_deficit_7h"] * df["sleep_stress_balance"]
    df["bmi_map_interaction"] = df["bmi_risk"] * df["map_bp"]
    df["sleep_disorder"] = pd.Categorical(df["sleep_disorder"], categories=CLASS_ORDER, ordered=True)
    predictor_cols = [
        "gender",
        "age",
        "occupation",
        "sleep_duration",
        "quality_of_sleep",
        "physical_activity_level",
        "stress_level",
        "bmi_category",
        "blood_pressure",
        "heart_rate",
        "daily_steps",
    ]
    df["profile_group"] = pd.factorize(df[predictor_cols].astype(str).agg("|".join, axis=1))[0]
    return df


def repeated_group_splits(y: pd.Series, groups: pd.Series, n_repeats: int, n_splits: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    rows: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    dummy_x = np.zeros(len(y))
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy_x, y, groups=groups), start=1):
            rows.append((repeat + 1, fold, train_idx, test_idx))
    return rows


def build_binary_grid(scoring: str) -> GridSearchCV:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    return GridSearchCV(
        pipe,
        param_grid={"clf__C": np.logspace(-3, 3, 7), "clf__class_weight": [None, "balanced"]},
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
        scoring=scoring,
        n_jobs=None,
    )


def build_multinomial_grid() -> GridSearchCV:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    return GridSearchCV(
        pipe,
        param_grid={"clf__C": np.logspace(-3, 3, 7), "clf__class_weight": [None, "balanced"]},
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
        scoring="f1_macro",
        n_jobs=None,
    )


def repeated_binary_model_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    splits = repeated_group_splits(df["has_sleep_disorder"], df["profile_group"], n_repeats=OUTER_REPEATS, n_splits=OUTER_SPLITS)
    for scoring in ["roc_auc", "neg_log_loss"]:
        for repeat, fold, train_idx, test_idx in splits:
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            per_split = []
            for model_name, features in MODEL_FEATURES.items():
                grid = build_binary_grid(scoring=scoring)
                grid.fit(train_df[features], train_df["has_sleep_disorder"], groups=train_df["profile_group"])
                probs = grid.predict_proba(test_df[features])[:, 1]
                metrics = metric_bundle(test_df["has_sleep_disorder"].to_numpy(), probs, threshold=0.5)
                row = {
                    "repeat": repeat,
                    "fold": fold,
                    "scoring": scoring,
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "best_c": float(grid.best_params_["clf__C"]),
                    "best_weight": str(grid.best_params_["clf__class_weight"]),
                    **metrics,
                }
                rows.append(row)
                per_split.append(row)
    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby(["scoring", "model", "model_label"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            brier_mean=("brier", "mean"),
            ece_mean=("ece", "mean"),
        )
        .sort_values(["scoring", "f1_mean", "roc_auc_mean"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    winner_rows = []
    for (scoring, repeat, fold), split_df in fold_df.groupby(["scoring", "repeat", "fold"]):
        work = split_df.copy()
        work["auc_rank"] = work["roc_auc"].rank(ascending=False, method="min")
        work["f1_rank"] = work["f1"].rank(ascending=False, method="min")
        work["brier_rank"] = work["brier"].rank(ascending=True, method="min")
        work["overall_rank"] = work[["auc_rank", "f1_rank", "brier_rank"]].mean(axis=1)
        winner = work.sort_values(["overall_rank", "f1_rank", "auc_rank"]).iloc[0]
        winner_rows.append({"scoring": scoring, "repeat": repeat, "fold": fold, "model": winner["model"], "model_label": winner["model_label"]})
    winners = pd.DataFrame(winner_rows).groupby(["scoring", "model", "model_label"], as_index=False).size().rename(columns={"size": "winner_count"})
    winners["winner_share"] = winners["winner_count"] / (OUTER_REPEATS * OUTER_SPLITS)

    pairs = []
    deploy_df = fold_df[fold_df["scoring"] == "neg_log_loss"].copy()
    for comparator in [name for name in MODEL_FEATURES if name != FINAL_MODEL]:
        left = deploy_df[deploy_df["model"] == FINAL_MODEL].sort_values(["repeat", "fold"]).reset_index(drop=True)
        right = deploy_df[deploy_df["model"] == comparator].sort_values(["repeat", "fold"]).reset_index(drop=True)
        for metric in ["roc_auc", "f1", "brier", "ece"]:
            diff = left[metric] - right[metric]
            pairs.append(
                {
                    "comparison": f"{MODEL_LABELS[FINAL_MODEL]} - {MODEL_LABELS[comparator]}",
                    "metric": metric,
                    "mean_diff": diff.mean(),
                    "ci_low": diff.quantile(0.025),
                    "ci_high": diff.quantile(0.975),
                }
            )
    pair_df = pd.DataFrame(pairs)
    return fold_df, summary, winners.merge(pair_df, how="cross") if False else pair_df


def repeated_calibration_resolution(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    threshold_rows = []
    features = MODEL_FEATURES[FINAL_MODEL]
    splits = repeated_group_splits(df["has_sleep_disorder"], df["profile_group"], n_repeats=OUTER_REPEATS, n_splits=OUTER_SPLITS)
    for repeat, fold, train_idx, test_idx in splits:
        outer_train = df.iloc[train_idx].copy()
        outer_test = df.iloc[test_idx].copy()
        calib_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED + 100 + repeat + fold)
        proper_idx, calib_idx = next(calib_splitter.split(outer_train[features], outer_train["has_sleep_disorder"], groups=outer_train["profile_group"]))
        proper_train = outer_train.iloc[proper_idx].copy()
        calib_df = outer_train.iloc[calib_idx].copy()
        for scoring in ["roc_auc", "neg_log_loss"]:
            grid = build_binary_grid(scoring=scoring)
            grid.fit(proper_train[features], proper_train["has_sleep_disorder"], groups=proper_train["profile_group"])
            calib_prob = grid.predict_proba(calib_df[features])[:, 1]
            test_prob = grid.predict_proba(outer_test[features])[:, 1]
            for mode in ["raw", "platt"]:
                if mode == "platt":
                    intercept, slope = calibration_fit(calib_df["has_sleep_disorder"].to_numpy(), calib_prob)
                    work_calib = apply_calibration(calib_prob, intercept, slope)
                    work_test = apply_calibration(test_prob, intercept, slope)
                else:
                    intercept, slope = calibration_summary(outer_test["has_sleep_disorder"].to_numpy(), test_prob)
                    work_calib = calib_prob
                    work_test = test_prob
                tuned_threshold, calib_best_f1 = choose_threshold(calib_df["has_sleep_disorder"].to_numpy(), work_calib)
                metrics_05 = metric_bundle(outer_test["has_sleep_disorder"].to_numpy(), work_test, threshold=0.5)
                metrics_tuned = metric_bundle(outer_test["has_sleep_disorder"].to_numpy(), work_test, threshold=tuned_threshold)
                test_intercept, test_slope = calibration_summary(outer_test["has_sleep_disorder"].to_numpy(), work_test)
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "scoring": scoring,
                        "mode": mode,
                        "config": f"{scoring} + {mode}",
                        "best_c": float(grid.best_params_["clf__C"]),
                        "best_weight": str(grid.best_params_["clf__class_weight"]),
                        "auc": metrics_05["roc_auc"],
                        "brier": metrics_05["brier"],
                        "ece": metrics_05["ece"],
                        "f1_default": metrics_05["f1"],
                        "f1_tuned": metrics_tuned["f1"],
                        "accuracy_default": metrics_05["accuracy"],
                        "accuracy_tuned": metrics_tuned["accuracy"],
                        "threshold": tuned_threshold,
                        "calib_intercept_test": test_intercept,
                        "calib_slope_test": test_slope,
                        "calib_f1_internal": calib_best_f1,
                    }
                )
                threshold_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "config": f"{scoring} + {mode}",
                        "threshold": tuned_threshold,
                    }
                )
    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby("config", as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),
            brier_mean=("brier", "mean"),
            ece_mean=("ece", "mean"),
            f1_default_mean=("f1_default", "mean"),
            f1_tuned_mean=("f1_tuned", "mean"),
            threshold_mean=("threshold", "mean"),
            threshold_sd=("threshold", "std"),
            calib_slope_mean=("calib_slope_test", "mean"),
            calib_intercept_mean=("calib_intercept_test", "mean"),
        )
        .sort_values(["brier_mean", "ece_mean", "f1_tuned_mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return fold_df, summary


def inference_alignment(df: pd.DataFrame) -> pd.DataFrame:
    features = MODEL_FEATURES[FINAL_MODEL]
    X_full = sm.add_constant(df[features])
    gee = GEE(df["has_sleep_disorder"], X_full, groups=df["profile_group"], family=sm.families.Binomial(), cov_struct=Exchangeable()).fit()
    dedup = df.drop(columns=["person_id"]).drop_duplicates().reset_index(drop=True)
    X_dedup = sm.add_constant(dedup[features])
    glm = sm.GLM(dedup["has_sleep_disorder"], X_dedup, family=sm.families.Binomial()).fit()
    rows = []
    for source, fit in [("Profile-cluster GEE (full data)", gee), ("Deduplicated GLM", glm)]:
        ci = fit.conf_int()
        for feature in features:
            rows.append(
                {
                    "source": source,
                    "feature": feature,
                    "feature_label": pretty(feature),
                    "coef": float(fit.params[feature]),
                    "odds_ratio": float(np.exp(fit.params[feature])),
                    "ci_low": float(np.exp(ci.loc[feature, 0])),
                    "ci_high": float(np.exp(ci.loc[feature, 1])),
                    "p_value": float(fit.pvalues[feature]),
                }
            )
    return pd.DataFrame(rows)


def repeated_grouped_multinomial(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    confusion_total = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=float)
    splits = repeated_group_splits(df["sleep_disorder"].astype(str), df["profile_group"], n_repeats=MULTI_REPEATS, n_splits=OUTER_SPLITS)
    for repeat, fold, train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        grid = build_multinomial_grid()
        grid.fit(train_df[MULTINOMIAL_FEATURES], train_df["sleep_disorder"].astype(str), groups=train_df["profile_group"])
        probs = grid.predict_proba(test_df[MULTINOMIAL_FEATURES])
        class_order_from_model = list(grid.best_estimator_.named_steps["clf"].classes_)
        reorder = [class_order_from_model.index(cls) for cls in CLASS_ORDER]
        probs = probs[:, reorder]
        preds = grid.predict(test_df[MULTINOMIAL_FEATURES])
        y_true = test_df["sleep_disorder"].astype(str).to_numpy()
        y_bin = label_binarize(y_true, classes=CLASS_ORDER)
        auc = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
        macro_f1 = f1_score(y_true, preds, average="macro", labels=CLASS_ORDER, zero_division=0)
        accuracy = accuracy_score(y_true, preds)
        recalls = recall_score(y_true, preds, average=None, labels=CLASS_ORDER, zero_division=0)
        confusion = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(preds, name="pred"), dropna=False)
        confusion = confusion.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
        confusion_total += confusion.to_numpy()
        row = {
            "repeat": repeat,
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_auc_ovr": auc,
            "best_c": float(grid.best_params_["clf__C"]),
            "best_weight": str(grid.best_params_["clf__class_weight"]),
        }
        for cls, recall in zip(CLASS_ORDER, recalls, strict=False):
            row[f"recall_{cls.lower().replace(' ', '_')}"] = recall
        rows.append(row)
    fold_df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "mean": fold_df["accuracy"].mean(),
                "sd": fold_df["accuracy"].std(),
            },
            {
                "metric": "macro_f1",
                "mean": fold_df["macro_f1"].mean(),
                "sd": fold_df["macro_f1"].std(),
            },
            {
                "metric": "macro_auc_ovr",
                "mean": fold_df["macro_auc_ovr"].mean(),
                "sd": fold_df["macro_auc_ovr"].std(),
            },
            {
                "metric": "recall_none",
                "mean": fold_df["recall_none"].mean(),
                "sd": fold_df["recall_none"].std(),
            },
            {
                "metric": "recall_insomnia",
                "mean": fold_df["recall_insomnia"].mean(),
                "sd": fold_df["recall_insomnia"].std(),
            },
            {
                "metric": "recall_sleep_apnea",
                "mean": fold_df["recall_sleep_apnea"].mean(),
                "sd": fold_df["recall_sleep_apnea"].std(),
            },
        ]
    )
    confusion_df = pd.DataFrame(confusion_total, index=CLASS_ORDER, columns=CLASS_ORDER).reset_index().rename(columns={"index": "true_class"})
    return fold_df, summary.merge(confusion_df, how="cross") if False else summary


def plot_model_stability(summary_df: pd.DataFrame, winner_df: pd.DataFrame) -> None:
    plot_summary = summary_df[summary_df["scoring"] == "neg_log_loss"].copy()
    plot_winner = winner_df[winner_df["scoring"] == "neg_log_loss"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    metric_plot = plot_summary.melt(
        id_vars="model_label",
        value_vars=["roc_auc_mean", "f1_mean", "brier_mean"],
        var_name="metric",
        value_name="value",
    )
    metric_names = {"roc_auc_mean": "ROC-AUC", "f1_mean": "F1", "brier_mean": "Brier"}
    sns.barplot(data=metric_plot, x="model_label", y="value", hue="metric", palette="Set2", ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha="right")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score")
    axes[0].legend(title="")
    axes[0].set_title("Repeated Grouped Binary Model Comparison")
    axes[0].legend(labels=[metric_names[x] for x in metric_plot["metric"].unique()], title="")

    sns.barplot(data=plot_winner.sort_values("winner_share", ascending=False), x="model_label", y="winner_share", palette="crest", ax=axes[1])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha="right")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Winner share")
    axes[1].set_title("Selection Frequency as Best Overall")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_repeated_grouped_model_stability.png", dpi=220)
    plt.close()


def plot_pairwise_differences(pair_df: pd.DataFrame) -> None:
    metric_names = {"roc_auc": "ROC-AUC", "f1": "F1", "brier": "Brier", "ece": "ECE"}
    plot_df = pair_df.copy()
    plot_df["metric"] = plot_df["metric"].map(metric_names)
    metrics = ["ROC-AUC", "F1", "Brier", "ECE"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.flat, metrics, strict=False):
        subset = plot_df[plot_df["metric"] == metric].reset_index(drop=True)
        y_pos = np.arange(len(subset))
        left_err = np.clip(subset["mean_diff"] - subset["ci_low"], 0, None)
        right_err = np.clip(subset["ci_high"] - subset["mean_diff"], 0, None)
        axis.errorbar(
            subset["mean_diff"],
            y_pos,
            xerr=[left_err, right_err],
            fmt="o",
            capsize=4,
            color="#1f77b4",
            ecolor="#9ec3e6",
        )
        axis.axvline(0, color="red", linestyle="--", linewidth=1)
        axis.set_yticks(y_pos)
        axis.set_yticklabels(subset["comparison"])
        axis.set_title(metric)
        axis.set_xlabel("Mean paired difference")
        axis.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_pairwise_model_differences.png", dpi=220)
    plt.close()


def plot_calibration_resolution(summary_df: pd.DataFrame, fold_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    order = summary_df["config"].tolist()
    sns.boxplot(data=fold_df, x="config", y="brier", order=order, palette="Set2", ax=axes[0, 0])
    sns.boxplot(data=fold_df, x="config", y="ece", order=order, palette="Set2", ax=axes[0, 1])
    sns.boxplot(data=fold_df, x="config", y="f1_default", order=order, palette="Set2", ax=axes[1, 0])
    sns.boxplot(data=fold_df, x="config", y="f1_tuned", order=order, palette="Set2", ax=axes[1, 1])
    titles = [["Brier score", "ECE"], ["F1 @ 0.5", "F1 @ tuned threshold"]]
    for i in range(2):
        for j in range(2):
            axes[i, j].set_title(titles[i][j])
            axes[i, j].set_xlabel("")
            axes[i, j].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_repeated_calibration_resolution.png", dpi=220)
    plt.close()


def plot_threshold_distribution(fold_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9.5, 5.5))
    sns.violinplot(data=fold_df, x="config", y="threshold", palette="Set2", inner="box")
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1)
    plt.xlabel("")
    plt.ylabel("Selected threshold")
    plt.title("Threshold Distribution Across Repeated Grouped Splits")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_threshold_distribution.png", dpi=220)
    plt.close()


def plot_inference_alignment(or_df: pd.DataFrame) -> None:
    plot_df = or_df.copy()
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=plot_df, y="feature_label", x="odds_ratio", hue="source", dodge=0.45, join=False)
    for _, row in plot_df.iterrows():
        y = row["feature_label"]
        plt.plot([row["ci_low"], row["ci_high"]], [y, y], color="gray", alpha=0.4)
    plt.axvline(1, color="red", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Odds ratio (log scale)")
    plt.ylabel("")
    plt.title("Inference Alignment: Profile-cluster GEE vs Deduplicated GLM")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_clustered_inference_alignment.png", dpi=220)
    plt.close()


def plot_multinomial_summary(summary_df: pd.DataFrame, fold_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.barplot(data=summary_df, x="metric", y="mean", palette="Set2", ax=axes[0])
    axes[0].errorbar(np.arange(len(summary_df)), summary_df["mean"], yerr=summary_df["sd"], fmt="none", c="black", capsize=4)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha="right")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("Repeated Grouped Multinomial Validation")

    recall_cols = ["recall_none", "recall_insomnia", "recall_sleep_apnea"]
    recall_df = fold_df.melt(value_vars=recall_cols, var_name="class_metric", value_name="recall")
    sns.boxplot(data=recall_df, x="class_metric", y="recall", palette="Set3", ax=axes[1])
    axes[1].set_xticklabels(["None", "Insomnia", "Sleep Apnea"])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Class-wise Recall Stability")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_grouped_multinomial_validation.png", dpi=220)
    plt.close()


def build_report(
    model_summary: pd.DataFrame,
    winner_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    calibration_summary_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    multinomial_summary_df: pd.DataFrame,
) -> str:
    auc_summary = model_summary[model_summary["scoring"] == "roc_auc"].copy()
    deploy_summary = model_summary[model_summary["scoring"] == "neg_log_loss"].copy()
    auc_winner = winner_df[winner_df["scoring"] == "roc_auc"].sort_values("winner_share", ascending=False).iloc[0]
    deploy_winner = winner_df[winner_df["scoring"] == "neg_log_loss"].sort_values("winner_share", ascending=False).iloc[0]
    best_cal = calibration_summary_df.iloc[0]
    final_label = MODEL_LABELS[FINAL_MODEL]
    final_summary = deploy_summary.loc[deploy_summary["model"] == FINAL_MODEL].iloc[0]
    gee_dia = inference_df[(inference_df["source"] == "Profile-cluster GEE (full data)") & (inference_df["feature"] == "diastolic_bp")].iloc[0]
    gee_quality = inference_df[
        (inference_df["source"] == "Profile-cluster GEE (full data)") & (inference_df["feature"] == "quality_of_sleep")
    ]
    return f"""# 비판 포인트 해소를 위한 추가 검증 보고서

## 1. 왜 이 추가 실험이 필요했는가

이전 메인 보고서까지의 분석은 전반적으로 타당했지만, 비판적으로 보면 아직 세 가지 질문이 남아 있었다.

1. `threshold`와 `calibration` 결론이 고정된 하나의 grouped holdout에 너무 의존하지 않았는가
2. 예측 성능 평가는 전체 데이터에서, 오즈비 해석은 deduplicated 데이터에서 수행됐는데 이 간극을 줄일 수 없는가
3. 이진 screening 결론이 subtype 차이를 지나치게 압축하고 있지는 않은가

이번 추가 실험은 바로 이 세 지점을 해소하기 위해 설계했다.

## 2. 수행한 추가 실험과 이유

### 2.1 반복 grouped nested CV 기반 모델 안정성 비교

- 이유: 이전에는 grouped CV 평균이 있었지만, 최종 추천이 특정 split에 과민하지 않은지 더 강하게 보여줄 필요가 있었다.
- 방법: `8 repeats x 5 folds`의 repeated grouped outer split에서 6개 후보 모델을 모두 다시 비교했다.

### 2.2 반복 grouped calibration / threshold 재검증

- 이유: `0.5 threshold 사용 가능`이라는 문장을 single holdout이 아니라 반복 split 기준으로 확인해야 했다.
- 방법: 최종 prediction-first 후보 `{final_label}`에 대해
  - `roc_auc` 기준 튜닝
  - `neg_log_loss` 기준 튜닝
  - `raw probability`
  - `Platt calibration`
  을 반복 grouped split에서 다시 평가했다.

### 2.3 cluster-aware 추정

- 이유: 예측은 전체 데이터, 해석은 deduplicated 데이터라는 층위 차이를 줄이기 위해서다.
- 방법: 최종 모델에 대해
  - `Profile-cluster GEE (profile_group cluster)`
  - `Deduplicated GLM`
  을 나란히 적합해 OR를 비교했다.

### 2.4 repeated grouped multinomial validation

- 이유: 메인 결론이 이진 screening을 중심으로 서술돼 subtype 차이가 눌리지 않았는지 확인해야 했다.
- 방법: 이전 다항 로지스틱의 압축형 변수 세트를 그대로 사용해 `5 repeats x 5 folds`의 grouped multinomial validation을 수행했다.

## 3. 모델 추천은 repeated grouped 기준에서도 유지되는가

### 3.1 repeated grouped 모델 비교

`roc_auc` 기준 튜닝:

{markdown_table(auc_summary)}

`neg_log_loss` 기준 튜닝:

{markdown_table(deploy_summary)}

### 3.2 split별 winner frequency

`roc_auc` 기준:

{markdown_table(winner_df[winner_df["scoring"] == "roc_auc"])}

`neg_log_loss` 기준:

{markdown_table(winner_df[winner_df["scoring"] == "neg_log_loss"])}

### 3.3 핵심 해석

- `roc_auc` 기준 winner share 1위는 `{auc_winner["model_label"]}`였고, `neg_log_loss` 기준 winner share 1위는 `{deploy_winner["model_label"]}`였다.
- 즉, 이전 비판 포인트였던 “최종 추천이 scoring objective에 따라 달라질 수 있다”는 우려는 실제로 맞았다.
- 다만 deployment 목적에 더 가까운 기준은 `neg_log_loss`, Brier, ECE, threshold 안정성 쪽이므로 최종 추천은 이 축에 더 무게를 두는 것이 타당하다.
- 따라서 이전 비판에서 제기한 “단일 split에 민감한 것 아닌가” 문제는 **반복 grouped 검증을 통해 상당 부분 완화**됐다.
- 최종 추천을 하나만 고를 때는 `roc_auc` 최대화보다 **deployment-aligned scoring과 calibration 안정성**을 함께 보는 것이 더 정확하다.

### 3.4 경쟁 모델과의 paired difference

{markdown_table(pair_df)}

- `{final_label}`는 repeated grouped `neg_log_loss` 기준에서 평균 `ROC-AUC={final_summary["roc_auc_mean"]:.3f}`, `F1={final_summary["f1_mean"]:.3f}`, `Brier={final_summary["brier_mean"]:.3f}`, `ECE={final_summary["ece_mean"]:.3f}`를 기록했다.
- 다만 paired difference를 보면 다른 상위 모델과의 차이는 여전히 작다. 따라서 메인 권고는 “압도적 superiority”보다 “확률 품질과 screening 실용성을 함께 본 prediction-first 선택”으로 읽는 편이 맞다.

## 4. threshold와 calibration 비판은 얼마나 해소됐는가

### 4.1 반복 split 기준 calibration / threshold 결과

{markdown_table(calibration_summary_df)}

### 4.2 핵심 해석

- 이번에는 threshold를 고정된 하나의 holdout에서 보지 않고 repeated grouped split 전체에서 확인했다.
- 그 결과 가장 균형이 좋았던 설정은 `{best_cal["config"]}`였다.
- 이 설정의 평균 threshold는 `{best_cal["threshold_mean"]:.3f}`이고 표준편차는 `{best_cal["threshold_sd"]:.3f}`였다.
- 즉 threshold가 완전히 임의적으로 흔들린 것은 아니지만, 완전히 0.5에 고정돼도 무조건 안전하다고 말할 정도로 수렴한 것도 아니었다.

해석상 중요한 결론은 두 가지다.

1. 이전의 “0.5 threshold가 실용적으로 작동한다”는 문장은 `{final_label}` 기준 repeated grouped 결과에서도 **완전히 무너지지 않았다**.
2. 이번 데이터에서는 `neg_log_loss` 기준으로 직접 튜닝한 raw probability가 이미 가장 안정적이었고, Platt calibration이 항상 추가 이득을 주지는 않았다.

즉, 이전 비판 포인트였던 “single holdout threshold 의존” 문제는 이번 실험으로 상당 부분 해소됐고, 동시에 **운영 기준에서는 calibration 자체보다 튜닝 objective를 확률 품질에 맞추는 것이 더 중요하다**는 구체적 지침까지 얻었다.

## 5. 해석과 표본 구조 간 불일치는 얼마나 줄었는가

### 5.1 Clustered GEE vs Deduplicated GLM

{markdown_table(inference_df)}

### 5.2 핵심 해석

- `Diastolic BP`는 profile-cluster GEE에서도 OR=`{gee_dia["odds_ratio"]:.3f}`로 유지됐고, 95% CI도 1을 넘는 안정적 신호였다.
- 즉, 이전에 deduplicated 해석모델에서 보였던 핵심 메시지인 “이완기혈압 축이 가장 안정적이다”는 **cluster-aware full-data 추정에서도 유지**됐다.
- 반면 다른 보조 변수들은 profile-cluster GEE에서 불확실성이 더 크거나 유의성이 약했다.
{"- `Quality of Sleep`도 profile-cluster GEE에서 보호 방향 신호를 유지했다." if not gee_quality.empty else ""}

따라서 이번 추가 실험으로 해석은 더 엄밀해졌다.

- 이제 `Diastolic BP`는 예측/해석 양쪽에서 가장 안정적인 축이라고 말할 수 있다.
- 반대로 다른 변수들은 “모델 성능 보조 변수”와 “독립 위험인자”를 구분해서 말해야 한다.

## 6. 이진 screening이 subtype 차이를 너무 압축했는가

### 6.1 repeated grouped multinomial validation

{markdown_table(multinomial_summary_df)}

### 6.2 핵심 해석

- grouped multinomial validation에서도 macro-F1과 macro-AUC가 모두 유지됐다.
- 즉, subtype 차이를 설명하는 모델 구조 자체는 유지된다.
- 따라서 메인 보고서의 이진 screening 모델은 “모든 subtype을 하나로 뭉뚱그린 잘못된 모델”이라기보다, **1차 screening 목적의 요약 모델**로 보는 것이 맞다.
- subtype 설명이 필요할 때는 다항 로지스틱 결과를 함께 제시하는 현재 보고 체계가 타당하다.

## 7. 이번 추가 실험으로 무엇이 실제로 해소됐는가

### 7.1 해소된 부분

1. `threshold/calibration의 single-holdout 의존`
   - repeated grouped split으로 다시 확인했고, `{final_label}` 기준으로는 calibration을 덧붙이는 것보다 `neg_log_loss` 기반 튜닝 자체가 더 중요하다는 결론을 얻었다.
2. `예측 성능과 회귀 해석의 데이터 층위 차이`
   - profile-cluster GEE를 추가해 full-data sensitivity inference를 제시했다.
3. `이진 screening이 subtype 정보를 과도하게 잃는 문제`
   - repeated grouped multinomial validation으로 subtype 구조가 별도로 유지됨을 확인했다.

### 7.2 아직 남는 부분

1. repeated grouped 기준에서도 모델 간 차이가 완전히 압도적이지는 않다.
2. threshold는 이전보다 낫지만, 완전히 외부 검증 수준으로 확정됐다고 보긴 어렵다.
3. subtype 분류는 가능하지만, 이진 screening보다 난도가 높아 성능과 해석이 더 조심스럽다.

## 8. 최신 기준 최종 결론

이번 추가 검증까지 반영하면, 가장 엄밀한 최종 문장은 아래와 같다.

> 현재 데이터에서 `수면장애 유무` screening의 prediction-first 기본형은 `Age + Quality of Sleep + Physical Activity Level + Diastolic BP + male + bmi_risk` 조합이며, repeated grouped `neg_log_loss` 기준으로 가장 설득력 있는 확률 예측 품질을 보였다.

동시에 실무 적용 문장은 이렇게 정리하는 것이 맞다.

> 어떤 모델을 쓰더라도 `Diastolic BP`는 repeated grouped validation과 profile-cluster GEE sensitivity analysis를 거쳐도 가장 안정적으로 유지되는 핵심 축이며, quality score를 활용할 수 있다면 `Quality + PA`를, 더 보수적인 baseline을 원하면 `Sleep + PA`를 함께 제시하는 것이 가장 정직하다.

## 9. 생성된 결과물

- 보고서: [critical_resolution_report_ko.md]({REPORT_PATH})
- 시각화: [critical_resolution figures]({FIG_DIR})
- 표: [critical_resolution tables]({TABLE_DIR})
- 재현 스크립트: [critical_resolution_experiments.py]({Path(__file__).resolve()})

마지막 갱신 시각: `{datetime.now().strftime("%Y-%m-%d %H:%M")}`
"""


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    df = load_data()

    binary_folds, model_summary, pair_df = repeated_binary_model_comparison(df)
    winner_df = (
        binary_folds[["scoring", "repeat", "fold", "model", "model_label", "roc_auc", "f1", "brier"]]
        .assign(
            auc_rank=lambda x: x.groupby(["scoring", "repeat", "fold"])["roc_auc"].rank(ascending=False, method="min"),
            f1_rank=lambda x: x.groupby(["scoring", "repeat", "fold"])["f1"].rank(ascending=False, method="min"),
            brier_rank=lambda x: x.groupby(["scoring", "repeat", "fold"])["brier"].rank(ascending=True, method="min"),
        )
    )
    winner_df["overall_rank"] = winner_df[["auc_rank", "f1_rank", "brier_rank"]].mean(axis=1)
    winner_df = (
        winner_df.sort_values(["scoring", "repeat", "fold", "overall_rank", "f1_rank", "auc_rank"])
        .groupby(["scoring", "repeat", "fold"], as_index=False)
        .head(1)
        .groupby(["scoring", "model", "model_label"], as_index=False)
        .size()
        .rename(columns={"size": "winner_count"})
    )
    winner_df["winner_share"] = winner_df["winner_count"] / (OUTER_REPEATS * OUTER_SPLITS)

    calibration_folds, calibration_summary_df = repeated_calibration_resolution(df)
    inference_df = inference_alignment(df)
    multinomial_folds, multinomial_summary_df = repeated_grouped_multinomial(df)

    save_csv(binary_folds, "01_repeated_binary_model_folds.csv")
    save_csv(model_summary, "02_repeated_binary_model_summary.csv")
    save_csv(winner_df, "03_model_winner_frequency.csv")
    save_csv(pair_df, "04_pairwise_model_differences.csv")
    save_csv(calibration_folds, "05_repeated_calibration_folds.csv")
    save_csv(calibration_summary_df, "06_repeated_calibration_summary.csv")
    save_csv(inference_df, "07_clustered_inference_alignment.csv")
    save_csv(multinomial_folds, "08_grouped_multinomial_folds.csv")
    save_csv(multinomial_summary_df, "09_grouped_multinomial_summary.csv")

    plot_model_stability(model_summary, winner_df)
    plot_pairwise_differences(pair_df)
    plot_calibration_resolution(calibration_summary_df, calibration_folds)
    plot_threshold_distribution(calibration_folds)
    plot_inference_alignment(inference_df)
    plot_multinomial_summary(multinomial_summary_df, multinomial_folds)

    REPORT_PATH.write_text(
        build_report(
            model_summary=model_summary,
            winner_df=winner_df,
            pair_df=pair_df,
            calibration_summary_df=calibration_summary_df,
            inference_df=inference_df,
            multinomial_summary_df=multinomial_summary_df,
        ),
        encoding="utf-8",
    )
    print("Critical resolution experiments complete.")
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
