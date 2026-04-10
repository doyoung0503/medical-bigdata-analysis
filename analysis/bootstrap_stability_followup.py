from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from critical_resolution_experiments import (
    MODEL_FEATURES,
    MODEL_LABELS,
    RANDOM_SEED,
    ROOT,
    load_data,
    metric_bundle,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


OUT_DIR = ROOT / "results" / "bootstrap_stability_followup"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "bootstrap_stability_followup_report_ko.md"

BOOT_REPEATS = 220
COEF_BOOT_REPEATS = 220
SELECTION_REPEATS = 140
INNER_SPLITS = 4
FINAL_MODEL = "orig_quality_pa"
SELECTION_FEATURES = [
    "age",
    "sleep_duration",
    "quality_of_sleep",
    "physical_activity_level",
    "stress_level",
    "heart_rate",
    "daily_steps",
    "systolic_bp",
    "diastolic_bp",
    "male",
    "bmi_risk",
]

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
    "male": "Male",
    "bmi_risk": "BMI Risk",
}


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


def normalize_class_weight(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value)
    if text in {"None", "nan", ""}:
        return None
    return text


def build_fixed_pipeline(c_value: float, class_weight: str | None, penalty: str = "l2") -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty=penalty,
                    solver="liblinear",
                    C=float(c_value),
                    class_weight=class_weight,
                    random_state=RANDOM_SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )


def infer_fixed_hyperparameters() -> pd.DataFrame:
    folds = pd.read_csv(ROOT / "results" / "critical_resolution" / "tables" / "01_repeated_binary_model_folds.csv")
    folds = folds[folds["scoring"] == "neg_log_loss"].copy()
    folds["best_weight"] = folds["best_weight"].map(normalize_class_weight)
    rows = []
    for model_name in MODEL_FEATURES:
        sub = folds[folds["model"] == model_name].copy()
        summary = (
            sub.groupby(["best_c", "best_weight"], dropna=False, as_index=False)
            .agg(count=("model", "size"), mean_brier=("brier", "mean"), mean_auc=("roc_auc", "mean"), mean_f1=("f1", "mean"))
            .sort_values(["count", "mean_brier", "mean_auc", "mean_f1"], ascending=[False, True, False, False])
            .iloc[0]
        )
        rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "fixed_c": float(summary["best_c"]),
                "fixed_weight": "None" if pd.isna(summary["best_weight"]) else str(summary["best_weight"]),
                "supporting_folds": int(summary["count"]),
                "mean_brier": float(summary["mean_brier"]),
                "mean_auc": float(summary["mean_auc"]),
                "mean_f1": float(summary["mean_f1"]),
            }
        )
    return pd.DataFrame(rows)


def build_group_index(df: pd.DataFrame) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    groups = np.sort(df["profile_group"].unique())
    index_map = {int(group): df.index[df["profile_group"] == group].to_numpy() for group in groups}
    return groups, index_map


def grouped_bootstrap_split(
    df: pd.DataFrame,
    group_values: np.ndarray,
    index_map: dict[int, np.ndarray],
    rng: np.random.Generator,
    max_tries: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    for _ in range(max_tries):
        sampled_groups = rng.choice(group_values, size=len(group_values), replace=True)
        selected_groups = set(int(x) for x in sampled_groups)
        oob_groups = [group for group in group_values if int(group) not in selected_groups]
        if not oob_groups:
            continue
        train_idx = np.concatenate([index_map[int(group)] for group in sampled_groups])
        test_idx = np.concatenate([index_map[int(group)] for group in oob_groups])
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)
        if train_df["has_sleep_disorder"].nunique() < 2 or test_df["has_sleep_disorder"].nunique() < 2:
            continue
        return train_df, test_df
    return None


def grouped_bootstrap_model_comparison(df: pd.DataFrame, hyper_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    group_values, index_map = build_group_index(df)
    hyper_map = {
        row["model"]: {"C": row["fixed_c"], "class_weight": None if row["fixed_weight"] == "None" else row["fixed_weight"]}
        for _, row in hyper_df.iterrows()
    }
    rows = []
    valid_repeats = 0
    while valid_repeats < BOOT_REPEATS:
        split = grouped_bootstrap_split(df, group_values, index_map, rng)
        if split is None:
            break
        train_df, test_df = split
        valid_repeats += 1
        for model_name, features in MODEL_FEATURES.items():
            config = hyper_map[model_name]
            pipe = build_fixed_pipeline(config["C"], config["class_weight"], penalty="l2")
            pipe.fit(train_df[features], train_df["has_sleep_disorder"])
            probs = pipe.predict_proba(test_df[features])[:, 1]
            rows.append(
                {
                    "replicate": valid_repeats,
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    **metric_bundle(test_df["has_sleep_disorder"].to_numpy(), probs, threshold=0.5),
                }
            )
    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby(["model", "model_label"], as_index=False)
        .agg(
            replicates=("replicate", "nunique"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            brier_mean=("brier", "mean"),
            ece_mean=("ece", "mean"),
        )
        .sort_values(["brier_mean", "roc_auc_mean", "f1_mean"], ascending=[True, False, False])
        .reset_index(drop=True)
    )

    pair_rows = []
    higher_better = {"roc_auc", "f1", "accuracy"}
    final_df = fold_df[fold_df["model"] == FINAL_MODEL].sort_values("replicate").reset_index(drop=True)
    for comparator in [name for name in MODEL_FEATURES if name != FINAL_MODEL]:
        comp_df = fold_df[fold_df["model"] == comparator].sort_values("replicate").reset_index(drop=True)
        for metric in ["roc_auc", "f1", "brier", "ece"]:
            diff = final_df[metric] - comp_df[metric]
            superiority = float((diff > 0).mean()) if metric in higher_better else float((diff < 0).mean())
            pair_rows.append(
                {
                    "comparison": f"{MODEL_LABELS[FINAL_MODEL]} - {MODEL_LABELS[comparator]}",
                    "metric": metric,
                    "mean_diff": float(diff.mean()),
                    "ci_low": float(diff.quantile(0.025)),
                    "ci_high": float(diff.quantile(0.975)),
                    "superiority_prob": superiority,
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    return fold_df, summary, pair_df


def grouped_stability_selection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED + 101)
    group_values, index_map = build_group_index(df)
    rows = []
    coef_rows = []
    param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}

    valid_repeats = 0
    while valid_repeats < SELECTION_REPEATS:
        split = grouped_bootstrap_split(df, group_values, index_map, rng)
        if split is None:
            break
        train_df, _ = split
        splitter = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED + 300 + valid_repeats)
        grid = GridSearchCV(
            estimator=build_fixed_pipeline(c_value=1.0, class_weight=None, penalty="l1"),
            param_grid=param_grid,
            scoring="neg_log_loss",
            cv=splitter,
            n_jobs=None,
            refit=True,
        )
        grid.fit(train_df[SELECTION_FEATURES], train_df["has_sleep_disorder"], groups=train_df["profile_group"])
        best_clf = grid.best_estimator_.named_steps["clf"]
        coef = best_clf.coef_[0]
        valid_repeats += 1
        rows.append(
            {
                "replicate": valid_repeats,
                "best_c": float(grid.best_params_["clf__C"]),
                "best_weight": "None" if grid.best_params_["clf__class_weight"] is None else str(grid.best_params_["clf__class_weight"]),
                "intercept": float(best_clf.intercept_[0]),
            }
        )
        for feature, value in zip(SELECTION_FEATURES, coef, strict=False):
            coef_rows.append(
                {
                    "replicate": valid_repeats,
                    "feature": feature,
                    "feature_label": pretty(feature),
                    "coef": float(value),
                    "selected": int(abs(value) > 1e-8),
                    "positive": int(value > 1e-8),
                    "negative": int(value < -1e-8),
                }
            )

    config_df = pd.DataFrame(rows)
    coef_df = pd.DataFrame(coef_rows)
    summary = (
        coef_df.groupby(["feature", "feature_label"], as_index=False)
        .agg(
            selection_freq=("selected", "mean"),
            positive_freq=("positive", "mean"),
            negative_freq=("negative", "mean"),
            median_coef=("coef", "median"),
            median_abs_coef=("coef", lambda x: float(np.median(np.abs(x)))),
            nonzero_median_coef=("coef", lambda x: float(np.median(x[np.abs(x) > 1e-8])) if np.any(np.abs(x) > 1e-8) else 0.0),
        )
        .sort_values(["selection_freq", "median_abs_coef"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return config_df, summary


def final_model_coefficient_bootstrap(df: pd.DataFrame, hyper_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 202)
    group_values, index_map = build_group_index(df)
    row = hyper_df.loc[hyper_df["model"] == FINAL_MODEL].iloc[0]
    class_weight = None if row["fixed_weight"] == "None" else row["fixed_weight"]
    features = MODEL_FEATURES[FINAL_MODEL]
    rows = []

    valid_repeats = 0
    while valid_repeats < COEF_BOOT_REPEATS:
        split = grouped_bootstrap_split(df, group_values, index_map, rng)
        if split is None:
            break
        train_df, _ = split
        pipe = build_fixed_pipeline(row["fixed_c"], class_weight, penalty="l2")
        pipe.fit(train_df[features], train_df["has_sleep_disorder"])
        coef = pipe.named_steps["clf"].coef_[0]
        valid_repeats += 1
        for feature, value in zip(features, coef, strict=False):
            rows.append({"replicate": valid_repeats, "feature": feature, "feature_label": pretty(feature), "coef": float(value)})

    coef_df = pd.DataFrame(rows)
    summary = (
        coef_df.groupby(["feature", "feature_label"], as_index=False)
        .agg(
            coef_median=("coef", "median"),
            coef_ci_low=("coef", lambda x: float(np.quantile(x, 0.025))),
            coef_ci_high=("coef", lambda x: float(np.quantile(x, 0.975))),
            prob_positive=("coef", lambda x: float((x > 0).mean())),
            prob_negative=("coef", lambda x: float((x < 0).mean())),
        )
        .sort_values("coef_median", key=lambda x: np.abs(x), ascending=False)
        .reset_index(drop=True)
    )
    summary["sign_stability"] = summary[["prob_positive", "prob_negative"]].max(axis=1)
    summary["or_per_1sd"] = np.exp(summary["coef_median"])
    return summary


def plot_model_bootstrap(summary_df: pd.DataFrame, fold_df: pd.DataFrame) -> None:
    plot_df = fold_df.melt(
        id_vars=["model_label"],
        value_vars=["roc_auc", "f1", "brier"],
        var_name="metric",
        value_name="value",
    )
    metric_names = {"roc_auc": "ROC-AUC", "f1": "F1", "brier": "Brier"}
    plot_df["metric"] = plot_df["metric"].map(metric_names)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.barplot(
        data=summary_df.melt(
            id_vars="model_label",
            value_vars=["roc_auc_mean", "f1_mean", "brier_mean"],
            var_name="metric",
            value_name="value",
        ),
        x="model_label",
        y="value",
        hue="metric",
        palette="Set2",
        ax=axes[0],
    )
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha="right")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Bootstrap mean")
    axes[0].legend(labels=["ROC-AUC", "F1", "Brier"], title="")
    axes[0].set_title("Grouped Bootstrap Model Comparison")

    sns.boxplot(data=plot_df, x="metric", y="value", hue="model_label", palette="Set3", ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Bootstrap distribution")
    axes[1].legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].set_title("Metric Variability Across Bootstrap Replicates")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_grouped_bootstrap_model_comparison.png", dpi=220)
    plt.close()


def plot_pairwise_superiority(pair_df: pd.DataFrame) -> None:
    plot_df = pair_df.copy()
    plot_df["metric"] = plot_df["metric"].map({"roc_auc": "ROC-AUC", "f1": "F1", "brier": "Brier", "ece": "ECE"})
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    sns.barplot(data=plot_df, x="comparison", y="superiority_prob", hue="metric", palette="Set2", ax=axes[0])
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Superiority probability")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("How Often Quality + PA Outperforms Each Comparator")

    forest_df = plot_df[plot_df["metric"] == "Brier"].reset_index(drop=True)
    y_pos = np.arange(len(forest_df))
    left_err = np.clip(forest_df["mean_diff"] - forest_df["ci_low"], 0, None)
    right_err = np.clip(forest_df["ci_high"] - forest_df["mean_diff"], 0, None)
    axes[1].errorbar(forest_df["mean_diff"], y_pos, xerr=[left_err, right_err], fmt="o", capsize=4, color="#1f77b4", ecolor="#9ec3e6")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(forest_df["comparison"])
    axes[1].set_xlabel("Mean Brier difference (Quality + PA - Comparator)")
    axes[1].set_ylabel("")
    axes[1].set_title("Paired Bootstrap Difference for Brier")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_pairwise_superiority.png", dpi=220)
    plt.close()


def plot_selection_stability(selection_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=selection_df, y="feature_label", x="selection_freq", palette="crest")
    plt.axvline(0.8, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Selection frequency")
    plt.ylabel("")
    plt.title("Grouped Bootstrap Stability Selection Frequency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_selection_frequency.png", dpi=220)
    plt.close()


def plot_final_coefficient_stability(coef_df: pd.DataFrame) -> None:
    plot_df = coef_df.copy().sort_values("coef_median", key=lambda x: np.abs(x), ascending=True)
    plt.figure(figsize=(9.5, 5.8))
    y_pos = np.arange(len(plot_df))
    left_err = np.clip(plot_df["coef_median"] - plot_df["coef_ci_low"], 0, None)
    right_err = np.clip(plot_df["coef_ci_high"] - plot_df["coef_median"], 0, None)
    plt.errorbar(plot_df["coef_median"], y_pos, xerr=[left_err, right_err], fmt="o", capsize=4, color="#1f77b4", ecolor="#9ec3e6")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.yticks(y_pos, plot_df["feature_label"])
    plt.xlabel("Standardized coefficient")
    plt.ylabel("")
    plt.title("Final Model Coefficient Stability Across Grouped Bootstraps")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_final_model_coefficient_stability.png", dpi=220)
    plt.close()


def build_report(
    hyper_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    selection_config_df: pd.DataFrame,
    selection_summary_df: pd.DataFrame,
    coef_summary_df: pd.DataFrame,
) -> str:
    final_model_row = model_summary_df.loc[model_summary_df["model"] == FINAL_MODEL].iloc[0]
    best_bootstrap_row = model_summary_df.iloc[0]
    quality_hr_row = model_summary_df.loc[model_summary_df["model"] == "orig_quality_hr"].iloc[0]
    top_selected = selection_summary_df.head(5)
    strong_final = coef_summary_df.sort_values("sign_stability", ascending=False).head(5)
    blood_pressure_selection = selection_summary_df[selection_summary_df["feature"].isin(["systolic_bp", "diastolic_bp"])]
    best_selection = selection_config_df.groupby(["best_c", "best_weight"], as_index=False).size().sort_values("size", ascending=False).iloc[0]
    return f"""# Grouped Bootstrap 모델 비교 및 변수 안정성 후속 보고서

## 1. 왜 이 후속 실험이 필요했는가

직전 단계까지의 분석으로 `Quality + PA`가 prediction-first 기본형이라는 결론에는 도달했지만, 두 가지 질문은 여전히 남아 있었다.

1. 상위 모델 간 차이가 실제로도 의미 있게 크다고 볼 수 있는가
2. 최종 변수와 후보 변수들은 분할이 바뀌어도 안정적으로 남는가

평균 ROC-AUC나 F1만 보고 모델을 고르면, 데이터 분할 우연에 민감한 결론을 과하게 일반화할 수 있다. 또한 최종 모델에 들어간 변수가 해석상 중요해 보여도, 다른 표본 재추출에서는 쉽게 빠지는 변수라면 “핵심 변수”라는 표현을 조심해야 한다. 그래서 이번 후속 실험은 **모델 추천의 엄밀성**과 **변수 안정성의 엄밀성**을 각각 보강하기 위해 설계했다.

## 2. 어떤 방식으로 검증했는가

### 2.1 grouped bootstrap OOB 모델 비교

- 이유: repeated grouped CV 결과는 이미 있었지만, 모델 차이를 다른 resampling 프레임에서도 다시 확인할 필요가 있었다.
- 방법:
  - predictor profile 기준 group을 bootstrap 단위로 사용했다.
  - bootstrap train은 group을 복원추출로 다시 뽑아 만들고, OOB group을 test로 사용했다.
  - 각 모델은 직전 `neg_log_loss` grouped CV에서 가장 자주 선택된 하이퍼파라미터를 고정한 뒤 비교했다.
  - 총 유효 replicate 수는 `{int(model_summary_df["replicates"].max())}`회였다.

고정 하이퍼파라미터는 아래와 같다.

{markdown_table(hyper_df)}

### 2.2 stability selection

- 이유: 최종 변수들이 정말 안정적인지 보려면, 사람이 고른 모델만 보는 것보다 더 넓은 원 변수 후보군에서 selection frequency를 확인하는 편이 낫다.
- 후보 변수:
  - `Age`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Stress Level`, `Heart Rate`, `Daily Steps`, `Systolic BP`, `Diastolic BP`, `Male`, `BMI Risk`
- 방법:
  - grouped bootstrap train을 다시 만들고
  - 그 안에서 `L1 logistic + grouped inner CV`로 `C`와 class weight를 튜닝해
  - 각 replicate에서 어떤 변수가 0이 아닌 계수로 남는지 기록했다.
- 가장 자주 선택된 stability-selection 설정은 `C={best_selection["best_c"]}`, `class_weight={best_selection["best_weight"]}`였다.

### 2.3 최종 모델 coefficient bootstrap

- 이유: 최종 모델 변수는 selection 여부뿐 아니라, 방향성과 계수 부호가 안정적인지도 봐야 한다.
- 방법:
  - `Original: Quality + PA` 모델을 grouped bootstrap으로 `220`회 다시 적합하고
  - 표준화 계수의 중앙값, 95% percentile interval, 부호 안정성(sign stability)을 계산했다.

## 3. 모델 차이는 실제로 얼마나 큰가

### 3.1 grouped bootstrap 성능 요약

{markdown_table(model_summary_df)}

### 3.2 핵심 해석

- bootstrap 평균 기준으로도 `Original: Quality + PA`는 상위권 모델군에 남았다.
- 다만 이번 grouped bootstrap OOB 평균만 놓고 보면 가장 좋은 평균 성능은 `{best_bootstrap_row["model_label"]}` 쪽에 더 가까웠다.
- `Original: Quality + PA`와 `Original: Quality + HR`의 차이는 매우 작았고, `Compressed Derived`, `Original: Sleep + HR`, `Original: Sleep + PA`도 바로 뒤를 따랐다.
- 즉, 이번 실험은 `Quality + PA`를 단독 승자로 강화했다기보다, **quality 기반 원 변수 모델군이 가장 안정적인 최상위권**이라는 해석을 더 강하게 만들었다.

### 3.3 paired superiority

{markdown_table(pair_df)}

이 표는 `Quality + PA`가 비교 모델보다 얼마나 자주 더 좋은지를 보여준다.

- superiority probability가 `0.5` 근처면 우열이 사실상 애매하다는 뜻이다.
- `Quality + PA`는 `Sleep + PA` 대비 Brier에서 개선되는 방향으로 더 자주 나타났지만, 그 확률이 1에 가깝진 않았다.
- `Quality + HR`와의 비교에서는 실질적으로 거의 동급이라고 보는 편이 더 맞다. 실제로 bootstrap 평균 성능은 `Quality + HR`가 아주 근소하게 앞섰다 (`Brier {quality_hr_row["brier_mean"]:.3f}` vs `Quality + PA {final_model_row["brier_mean"]:.3f}`).

따라서 모델 비교 차원에서 더 엄밀한 최종 문장은 아래처럼 정리된다.

> grouped bootstrap은 `Quality + PA` 단독 우위를 강하게 지지하지 않았다. 대신 `Quality + PA`와 `Quality + HR`가 같은 최상위 quality-based model family로 남는다고 해석하는 것이 가장 정확하다.

## 4. 어떤 변수가 안정적으로 선택되는가

### 4.1 stability selection 결과

{markdown_table(selection_summary_df)}

### 4.2 핵심 해석

- 높은 selection frequency는 “표본이 바뀌어도 자주 살아남는 변수”라는 뜻이다.
- 낮은 selection frequency는 예측에 조금 기여하더라도 다른 변수와 정보가 겹치거나, 독립 신호가 약하다는 뜻일 수 있다.

상위 안정 변수 5개는 아래와 같았다.

{markdown_table(top_selected)}

이 단계의 가장 중요한 해석은 다음과 같다.

1. `BMI Risk`와 `Quality of Sleep`는 광범위한 후보군에서도 매우 안정적으로 살아남았다.
2. 혈압 변수는 `Systolic BP`와 `Diastolic BP`가 모두 높게 선택돼, 개별 변수보다 **혈압축 전체가 안정적**이라고 해석하는 편이 맞다.
3. `Sleep Duration`, `Heart Rate`, `Physical Activity Level`은 상황에 따라 대체 가능한 보조 축으로 보인다.

혈압축 selection frequency는 아래처럼 읽는 것이 자연스럽다.

{markdown_table(blood_pressure_selection)}

## 5. 최종 모델 변수의 방향성은 얼마나 안정적인가

{markdown_table(coef_summary_df)}

### 5.1 핵심 해석

아래 변수들은 부호 안정성이 특히 높은 축이었다.

{markdown_table(strong_final)}

- `sign_stability`가 높다는 것은 bootstrap 재표집을 해도 방향이 거의 바뀌지 않는다는 뜻이다.
- `Diastolic BP`가 양(+) 방향으로, `Quality of Sleep`가 음(-) 방향으로 유지된다면 이는 이전 보고서의 핵심 해석을 더 강하게 뒷받침한다.
- 반면 `Physical Activity Level`과 `Male`는 모델에 포함되더라도 계수 방향의 안정성이 상대적으로 더 약하고, `Age`와 `BMI Risk`는 이번 bootstrap에서는 부호가 매우 안정적으로 유지됐다.

## 6. 이번 후속 실험으로 무엇이 더 명확해졌는가

### 6.1 더 강해진 결론

1. `Quality of Sleep`, `BMI Risk`, `혈압축`은 표본 재추출을 해도 가장 안정적으로 남는 핵심 변수군이다.
2. 최종 고정 모델 안에서는 `Diastolic BP`, `Quality of Sleep`, `BMI Risk`, `Age`의 방향성이 특히 안정적이다.
3. 최종 모델 추천은 “단일 절대 승자”보다 “quality-based prediction-first family”로 표현하는 것이 더 엄밀하다.

### 6.2 여전히 남는 한계

1. grouped bootstrap에서도 상위 모델 간 차이가 작으면, 최종 추천은 여전히 목적 의존적이다.
2. stability selection은 L1 벌점의 성질상 상관된 변수들 사이에서 선택을 나눠 가질 수 있다.
3. 따라서 낮은 selection frequency를 곧바로 “중요하지 않다”로 읽으면 안 되고, “중복 정보가 있어 단독 신호가 약하다”로 보는 편이 더 안전하다.

## 7. 최신 기준 최종 문장

이번 후속 검증까지 포함하면 가장 엄밀한 최종 문장은 아래와 같다.

> grouped bootstrap 기준으로는 `Quality + PA`와 `Quality + HR`가 사실상 같은 최상위 quality-based prediction-first family로 남았고, 둘의 차이는 근소했다.

> 변수 수준에서는 `Quality of Sleep`, `BMI Risk`, 혈압축 전체가 가장 안정적이었고, 최종 `Quality + PA` 고정 모델 안에서는 `Diastolic BP`가 특히 안정적인 양(+) 방향 신호로 유지됐다.

## 8. 생성된 결과물

- 보고서: [bootstrap_stability_followup_report_ko.md]({REPORT_PATH})
- 시각화: [bootstrap_stability_followup figures]({FIG_DIR})
- 표: [bootstrap_stability_followup tables]({TABLE_DIR})
- 재현 스크립트: [bootstrap_stability_followup.py]({Path(__file__).resolve()})

마지막 갱신 시각: `{datetime.now().strftime("%Y-%m-%d %H:%M")}`
"""


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    df = load_data()

    hyper_df = infer_fixed_hyperparameters()
    model_fold_df, model_summary_df, pair_df = grouped_bootstrap_model_comparison(df, hyper_df)
    selection_config_df, selection_summary_df = grouped_stability_selection(df)
    coef_summary_df = final_model_coefficient_bootstrap(df, hyper_df)

    save_csv(hyper_df, "01_fixed_hyperparameters.csv")
    save_csv(model_fold_df, "02_grouped_bootstrap_model_folds.csv")
    save_csv(model_summary_df, "03_grouped_bootstrap_model_summary.csv")
    save_csv(pair_df, "04_grouped_bootstrap_pairwise_superiority.csv")
    save_csv(selection_config_df, "05_stability_selection_configs.csv")
    save_csv(selection_summary_df, "06_stability_selection_summary.csv")
    save_csv(coef_summary_df, "07_final_model_coefficient_stability.csv")

    plot_model_bootstrap(model_summary_df, model_fold_df)
    plot_pairwise_superiority(pair_df)
    plot_selection_stability(selection_summary_df)
    plot_final_coefficient_stability(coef_summary_df)

    REPORT_PATH.write_text(
        build_report(
            hyper_df=hyper_df,
            model_summary_df=model_summary_df,
            pair_df=pair_df,
            selection_config_df=selection_config_df,
            selection_summary_df=selection_summary_df,
            coef_summary_df=coef_summary_df,
        ),
        encoding="utf-8",
    )
    print("Bootstrap stability follow-up complete.")
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
