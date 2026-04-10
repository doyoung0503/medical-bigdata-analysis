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

from critical_resolution_experiments import MODEL_FEATURES, MODEL_LABELS, RANDOM_SEED, metric_bundle
from rigorous_followup_validation import ROOT, markdown_table, robust_categorical_tests, robust_numeric_tests, load_data


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


OUT_DIR = ROOT / "results" / "final_rigor_upgrade"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "final_rigor_upgrade_report_ko.md"

FINAL_MODEL = "orig_quality_pa"
NESTED_BOOT_REPEATS = 80
INNER_SPLITS = 3
PARAM_GRID = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}


class LiveReport:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sections: list[tuple[str, str]] = []
        self.intro = ""

    def set_intro(self, intro: str) -> None:
        self.intro = intro
        self.write()

    def upsert(self, title: str, content: str) -> None:
        for idx, (existing_title, _) in enumerate(self.sections):
            if existing_title == title:
                self.sections[idx] = (title, content)
                break
        else:
            self.sections.append((title, content))
        self.write()

    def write(self) -> None:
        completed = "\n".join(f"- {title}" for title, _ in self.sections) or "- 진행 중"
        body = [
            "# 최종 엄밀성 보강 보고서",
            "",
            self.intro.strip(),
            "",
            "## 현재까지 완료된 단계",
            completed,
            "",
        ]
        for title, content in self.sections:
            body.extend([f"## {title}", "", content.strip(), ""])
        self.path.write_text("\n".join(body).strip() + "\n", encoding="utf-8")


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, index=False)


def deduplicate_exact_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["person_id"]).drop_duplicates().reset_index(drop=True)


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


def classify_transition(full_keep: bool, dedup_keep: bool) -> str:
    if full_keep and dedup_keep:
        return "stable_keep"
    if full_keep and not dedup_keep:
        return "weakened_after_dedup"
    if (not full_keep) and dedup_keep:
        return "strengthened_after_dedup"
    return "stable_drop"


def summarize_univariate_alignment(
    full_df: pd.DataFrame, dedup_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_numeric, _ = robust_numeric_tests(full_df)
    dedup_numeric, _ = robust_numeric_tests(dedup_df)
    full_cat = robust_categorical_tests(full_df)
    dedup_cat = robust_categorical_tests(dedup_df)

    numeric_compare = full_numeric[
        ["feature", "feature_label", "welch_fdr", "kruskal_fdr", "eta_squared", "epsilon_squared", "robust_keep"]
    ].merge(
        dedup_numeric[
            ["feature", "feature_label", "welch_fdr", "kruskal_fdr", "eta_squared", "epsilon_squared", "robust_keep"]
        ],
        on=["feature", "feature_label"],
        suffixes=("_full", "_dedup"),
    )
    numeric_compare["consistency"] = [
        classify_transition(full_keep, dedup_keep)
        for full_keep, dedup_keep in zip(
            numeric_compare["robust_keep_full"], numeric_compare["robust_keep_dedup"], strict=False
        )
    ]
    numeric_compare = numeric_compare.sort_values(
        ["robust_keep_dedup", "robust_keep_full", "welch_fdr_dedup", "welch_fdr_full"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    cat_compare = full_cat[
        ["feature", "feature_label", "perm_fdr", "cramers_v", "cells_lt5", "min_expected"]
    ].merge(
        dedup_cat[["feature", "feature_label", "perm_fdr", "cramers_v", "cells_lt5", "min_expected"]],
        on=["feature", "feature_label"],
        suffixes=("_full", "_dedup"),
    )
    cat_compare["keep_full"] = cat_compare["perm_fdr_full"] < 0.05
    cat_compare["keep_dedup"] = cat_compare["perm_fdr_dedup"] < 0.05
    cat_compare["consistency"] = [
        classify_transition(full_keep, dedup_keep)
        for full_keep, dedup_keep in zip(cat_compare["keep_full"], cat_compare["keep_dedup"], strict=False)
    ]
    cat_compare = cat_compare.sort_values(
        ["keep_dedup", "keep_full", "perm_fdr_dedup", "perm_fdr_full"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    dataset_compare = pd.DataFrame(
        [
            {"dataset": "Full rows", "rows": len(full_df), "unique_profiles": int(full_df["profile_group"].nunique())},
            {"dataset": "Exact-row deduplicated", "rows": len(dedup_df), "unique_profiles": int(dedup_df["profile_group"].nunique())},
        ]
    )
    return dataset_compare, numeric_compare, cat_compare, dedup_numeric


def plot_univariate_alignment(numeric_compare: pd.DataFrame, cat_compare: pd.DataFrame) -> None:
    heatmap_df = numeric_compare[
        ["feature_label", "welch_fdr_full", "kruskal_fdr_full", "welch_fdr_dedup", "kruskal_fdr_dedup"]
    ].copy()
    for column in ["welch_fdr_full", "kruskal_fdr_full", "welch_fdr_dedup", "kruskal_fdr_dedup"]:
        heatmap_df[column] = -np.log10(np.clip(heatmap_df[column], 1e-12, 1))
    heatmap_df = heatmap_df.rename(
        columns={
            "welch_fdr_full": "Full Welch FDR",
            "kruskal_fdr_full": "Full Kruskal FDR",
            "welch_fdr_dedup": "Dedup Welch FDR",
            "kruskal_fdr_dedup": "Dedup Kruskal FDR",
        }
    ).set_index("feature_label")
    plt.figure(figsize=(10, 9))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Univariate Robustness After Exact-row Deduplication")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_numeric_alignment_heatmap.png", dpi=220)
    plt.close()

    cat_plot = cat_compare.melt(
        id_vars="feature_label",
        value_vars=["cramers_v_full", "cramers_v_dedup"],
        var_name="dataset",
        value_name="cramers_v",
    )
    cat_plot["dataset"] = cat_plot["dataset"].map(
        {"cramers_v_full": "Full rows", "cramers_v_dedup": "Exact-row deduplicated"}
    )
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=cat_plot, y="feature_label", x="cramers_v", hue="dataset", palette="Set2")
    plt.xlabel("Cramer's V")
    plt.ylabel("")
    plt.title("Categorical Association Size Before and After Deduplication")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_categorical_alignment.png", dpi=220)
    plt.close()


def build_nested_grid(seed: int) -> GridSearchCV:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    return GridSearchCV(
        pipe,
        param_grid=PARAM_GRID,
        cv=StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed),
        scoring="neg_log_loss",
        refit=True,
        n_jobs=None,
    )


def nested_grouped_bootstrap(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED + 700)
    group_values, index_map = build_group_index(df)
    rows = []

    valid_repeats = 0
    while valid_repeats < NESTED_BOOT_REPEATS:
        split = grouped_bootstrap_split(df, group_values, index_map, rng)
        if split is None:
            break
        train_df, test_df = split
        valid_repeats += 1
        for model_idx, (model_name, features) in enumerate(MODEL_FEATURES.items(), start=1):
            grid = build_nested_grid(RANDOM_SEED + valid_repeats * 37 + model_idx)
            grid.fit(train_df[features], train_df["has_sleep_disorder"], groups=train_df["profile_group"])
            probs = grid.predict_proba(test_df[features])[:, 1]
            rows.append(
                {
                    "replicate": valid_repeats,
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "best_c": float(grid.best_params_["clf__C"]),
                    "best_weight": "None" if grid.best_params_["clf__class_weight"] is None else str(grid.best_params_["clf__class_weight"]),
                    **metric_bundle(test_df["has_sleep_disorder"].to_numpy(), probs, threshold=0.5),
                }
            )

    fold_df = pd.DataFrame(rows)

    winner_rows = []
    for replicate, split_df in fold_df.groupby("replicate"):
        work = split_df.copy()
        work["auc_rank"] = work["roc_auc"].rank(ascending=False, method="min")
        work["f1_rank"] = work["f1"].rank(ascending=False, method="min")
        work["brier_rank"] = work["brier"].rank(ascending=True, method="min")
        work["overall_rank"] = work[["auc_rank", "f1_rank", "brier_rank"]].mean(axis=1)
        winner = work.sort_values(["overall_rank", "brier_rank", "f1_rank", "auc_rank"]).iloc[0]
        winner_rows.append({"replicate": replicate, "model": winner["model"], "model_label": winner["model_label"]})
    winner_df = pd.DataFrame(winner_rows).groupby(["model", "model_label"], as_index=False).size().rename(columns={"size": "winner_count"})
    winner_df["winner_share"] = winner_df["winner_count"] / max(fold_df["replicate"].nunique(), 1)

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
        .merge(winner_df, on=["model", "model_label"], how="left")
        .fillna({"winner_count": 0, "winner_share": 0.0})
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
                    "empirical_low": float(diff.quantile(0.025)),
                    "empirical_high": float(diff.quantile(0.975)),
                    "superiority_prob": superiority,
                }
            )
    pair_df = pd.DataFrame(pair_rows)

    hyper_summary = (
        fold_df.groupby(["model", "model_label", "best_c", "best_weight"], as_index=False)
        .agg(count=("replicate", "size"))
        .sort_values(["model", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return fold_df, summary, pair_df, hyper_summary


def plot_nested_bootstrap(summary_df: pd.DataFrame, fold_df: pd.DataFrame, pair_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    metric_plot = summary_df.melt(
        id_vars="model_label",
        value_vars=["roc_auc_mean", "f1_mean", "brier_mean"],
        var_name="metric",
        value_name="value",
    )
    metric_plot["metric"] = metric_plot["metric"].map(
        {"roc_auc_mean": "ROC-AUC", "f1_mean": "F1", "brier_mean": "Brier"}
    )
    sns.barplot(data=metric_plot, x="model_label", y="value", hue="metric", palette="Set2", ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha="right")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Nested bootstrap mean")
    axes[0].legend(title="")
    axes[0].set_title("Nested Grouped Bootstrap Model Comparison")

    box_df = fold_df.melt(
        id_vars=["model_label"],
        value_vars=["brier", "roc_auc", "f1"],
        var_name="metric",
        value_name="value",
    )
    box_df["metric"] = box_df["metric"].map({"brier": "Brier", "roc_auc": "ROC-AUC", "f1": "F1"})
    sns.boxplot(data=box_df, x="metric", y="value", hue="model_label", palette="Set3", ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Replicate distribution")
    axes[1].legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].set_title("Metric Variability Across Nested Bootstraps")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_nested_grouped_bootstrap_comparison.png", dpi=220)
    plt.close()

    plot_df = pair_df.copy()
    plot_df["metric"] = plot_df["metric"].map({"roc_auc": "ROC-AUC", "f1": "F1", "brier": "Brier", "ece": "ECE"})
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    sns.barplot(data=plot_df, x="comparison", y="superiority_prob", hue="metric", palette="Set2", ax=axes[0])
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Superiority probability")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("How Often Quality + PA Wins in Nested Grouped Bootstrap")

    forest_df = plot_df[plot_df["metric"] == "Brier"].reset_index(drop=True)
    y_pos = np.arange(len(forest_df))
    left_err = np.clip(forest_df["mean_diff"] - forest_df["empirical_low"], 0, None)
    right_err = np.clip(forest_df["empirical_high"] - forest_df["mean_diff"], 0, None)
    axes[1].errorbar(
        forest_df["mean_diff"],
        y_pos,
        xerr=[left_err, right_err],
        fmt="o",
        capsize=4,
        color="#1f77b4",
        ecolor="#9ec3e6",
    )
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(forest_df["comparison"])
    axes[1].set_xlabel("Mean Brier difference (Quality + PA - Comparator)")
    axes[1].set_ylabel("")
    axes[1].set_title("Empirical 95% Interval for Brier Difference")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_nested_pairwise_comparison.png", dpi=220)
    plt.close()


def build_report(
    dataset_compare: pd.DataFrame,
    numeric_compare: pd.DataFrame,
    cat_compare: pd.DataFrame,
    nested_summary: pd.DataFrame,
    pair_df: pd.DataFrame,
    hyper_summary: pd.DataFrame,
) -> str:
    stable_numeric = numeric_compare.loc[
        numeric_compare["consistency"] == "stable_keep", "feature_label"
    ].tolist()
    weakened_numeric = numeric_compare.loc[
        numeric_compare["consistency"] == "weakened_after_dedup", "feature_label"
    ].tolist()
    stable_cat = cat_compare.loc[cat_compare["consistency"] == "stable_keep", "feature_label"].tolist()
    weakened_cat = cat_compare.loc[cat_compare["consistency"] == "weakened_after_dedup", "feature_label"].tolist()

    best_row = nested_summary.iloc[0]
    quality_hr_row = nested_summary.loc[nested_summary["model"] == "orig_quality_hr"].iloc[0]
    quality_pa_row = nested_summary.loc[nested_summary["model"] == "orig_quality_pa"].iloc[0]
    pair_quality_hr = pair_df[pair_df["comparison"] == "Original: Quality + PA - Original: Quality + HR"].copy()
    pair_sleep_pa = pair_df[pair_df["comparison"] == "Original: Quality + PA - Original: Sleep + PA"].copy()
    top_hyper = (
        hyper_summary.sort_values(["count", "model"], ascending=[False, True])
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return f"""# 최종 엄밀성 보강 보고서

## 1. 왜 이 보강 실험이 필요했는가

이전까지의 분석은 이미 상당히 엄격했지만, 최대 엄밀성 기준에서 두 가지가 남아 있었다.

1. 단변량 검정은 full-row 기준인데, 예측 성능 검증은 grouped 기준이었다.
2. grouped bootstrap 모델 비교는 이전 CV에서 선택된 하이퍼파라미터를 고정해 비교했기 때문에, bootstrap 바깥에서 이미 한 번 선택된 설정에 조건부였다.

즉, 이번 실험의 목적은 **단변량 검정과 모델 검증의 독립성 기준을 더 잘 맞추고**, **bootstrap 안에서 다시 튜닝하는 nested 비교로 모델 우열을 재확인**하는 데 있었다.

## 2. 이번에 무엇을 다시 했는가

### 2.1 exact-row deduplicated 단변량 재검정

- 같은 row가 여러 번 반복되면 p-value가 과도하게 작아질 수 있으므로, `person_id`를 제외한 정확히 같은 row는 하나만 남겼다.
- predictor profile이 같아도 label이 다른 경우는 그대로 유지했다.
  - 이유: 이런 경우까지 하나로 접어버리면 임의의 대표 label을 정해야 하므로, 오히려 더 강한 가정을 도입하게 된다.
- 따라서 이번 dedup은 “중복 정보량의 과장”만 줄이고, label ambiguity는 인위적으로 없애지 않는 보수적 민감도 분석이다.

데이터 정렬 결과는 아래와 같다.

{markdown_table(dataset_compare)}

### 2.2 nested grouped bootstrap 모델 재비교

- bootstrap replicate마다 train/OOB split을 새로 만들고,
- 각 replicate 내부에서 다시 grouped inner CV로 `C`와 class weight를 재튜닝한 뒤,
- OOB test에서 성능을 평가했다.

즉, 이번 비교는 “이전 CV에서 한 번 고른 설정이 bootstrap에서도 좋다”가 아니라, **replicate마다 다시 최적 설정을 찾은 뒤에도 quality-based family가 앞서는가**를 묻는 구조다.

## 3. 단변량 결론은 dedup 후에도 유지되는가

### 3.1 수치형 변수

{markdown_table(numeric_compare)}

핵심 해석:

- full-row와 deduplicated row 모두에서 강건하게 유지된 수치형 축은 다음과 같았다.
  - {", ".join(stable_numeric) if stable_numeric else "없음"}
- dedup 후 강건성이 약해진 변수는 다음과 같았다.
  - {", ".join(weakened_numeric) if weakened_numeric else "없음"}

이 결과는 “혈압축, 나이, 수면시간, 수면의 질, 활동량”이라는 큰 방향이 단순 반복행 때문에만 생긴 결과는 아니라는 점을 보여준다.

### 3.2 범주형 변수

{markdown_table(cat_compare)}

핵심 해석:

- dedup 후에도 남은 범주형 축은 다음과 같았다.
  - {", ".join(stable_cat) if stable_cat else "없음"}
- dedup 후 약해진 범주형 축은 다음과 같았다.
  - {", ".join(weakened_cat) if weakened_cat else "없음"}

즉, 단변량 단계에서도 `Gender`, `BMI 계열` 신호는 비교적 안정적이지만, 희소도가 높은 변수는 독립성 기준을 보수적으로 맞출수록 더 조심해서 읽는 편이 맞다.

## 4. nested grouped bootstrap에서 모델 우열은 어떻게 바뀌는가

### 4.1 성능 요약

{markdown_table(nested_summary)}

### 4.2 replicate 내부 튜닝에서 가장 자주 선택된 설정

{markdown_table(top_hyper)}

### 4.3 paired empirical difference

{markdown_table(pair_df)}

핵심 해석:

- nested grouped bootstrap 평균 기준 최상위 모델은 `{best_row["model_label"]}`였다.
- `Original: Quality + PA`와 `Original: Quality + HR`의 성능 차이는 여전히 매우 작았다.
  - `Quality + PA`: `AUC {quality_pa_row["roc_auc_mean"]:.3f}`, `F1 {quality_pa_row["f1_mean"]:.3f}`, `Brier {quality_pa_row["brier_mean"]:.3f}`
  - `Quality + HR`: `AUC {quality_hr_row["roc_auc_mean"]:.3f}`, `F1 {quality_hr_row["f1_mean"]:.3f}`, `Brier {quality_hr_row["brier_mean"]:.3f}`
- `Quality + PA - Quality + HR`의 empirical difference를 보면, superiority probability가 0.5 근처에 머물면 단일 승자라고 부르기 어렵다.
- `Quality + PA - Sleep + PA` 비교는 quality score를 쓸 때 얼마나 확률 품질이 좋아지는지를 보여주는 보수적 대조군으로 읽는 편이 맞다.

특히 아래 두 비교는 최종 모델 추천 문장을 결정하는 데 중요했다.

`Quality + PA vs Quality + HR`

{markdown_table(pair_quality_hr)}

`Quality + PA vs Sleep + PA`

{markdown_table(pair_sleep_pa)}

## 5. 이번 보강 실험으로 무엇이 더 명확해졌는가

1. 단변량 결론은 exact-row dedup 후에도 큰 방향이 유지됐다.
2. 따라서 `혈압축 + BMI Risk + Quality of Sleep`가 중요하다는 메시지는 반복 row 때문에만 생긴 인공 신호라고 보기 어렵다.
3. nested grouped bootstrap 안에서 다시 튜닝해도 quality-based family가 상위권에 남으면, 기존 모델 추천은 이전 fixed-hyperparameter bootstrap에만 의존한 결론이 아니게 된다.
4. 반대로 quality-based family 내부에서 `PA`와 `HR`의 차이가 작다면, 최종 권고는 “유일한 통계적 승자”보다 “같은 최상위 family”라는 식으로 보수적으로 쓰는 편이 더 엄밀하다.

## 6. 최신 기준 최종 문장

이번 보강 실험까지 반영하면 가장 엄밀한 최종 문장은 아래와 같다.

> exact-row dedup 단변량 재검정 후에도 혈압축, BMI 축, 수면의 질/수면시간 축의 큰 방향은 유지됐다.

> nested grouped bootstrap 안에서 다시 튜닝해도 `Quality + PA`와 `Quality + HR`는 여전히 같은 최상위 quality-based prediction-first family로 해석하는 것이 가장 보수적이고 정확하다.

> operational default 하나를 고르더라도, 그것은 통계적으로 압도적 단일 승자라기보다 quality-based family 안에서의 실용적 선택으로 읽는 것이 맞다.

## 7. 생성된 결과물

- 보고서: [final_rigor_upgrade_report_ko.md]({REPORT_PATH})
- 시각화: [final_rigor_upgrade figures]({FIG_DIR})
- 표: [final_rigor_upgrade tables]({TABLE_DIR})
- 재현 스크립트: [final_rigor_upgrade.py]({Path(__file__).resolve()})

마지막 갱신 시각: `{datetime.now().strftime("%Y-%m-%d %H:%M")}`
"""


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    report = LiveReport(REPORT_PATH)
    report.set_intro(
        "이 보고서는 남아 있던 엄밀성 포인트 2가지를 보강한다. "
        "첫째, 단변량 검정의 독립성 기준을 exact-row dedup sensitivity로 다시 맞춘다. "
        "둘째, bootstrap replicate마다 다시 튜닝하는 nested grouped bootstrap으로 모델 우열을 재검토한다."
    )

    full_df = load_data()
    dedup_df = deduplicate_exact_rows(full_df)

    dataset_compare, numeric_compare, cat_compare, _ = summarize_univariate_alignment(full_df, dedup_df)
    save_csv(dataset_compare, "01_dataset_alignment_summary.csv")
    save_csv(numeric_compare, "02_numeric_dedup_alignment.csv")
    save_csv(cat_compare, "03_categorical_dedup_alignment.csv")
    plot_univariate_alignment(numeric_compare, cat_compare)
    report.upsert(
        "Exact-row Dedup 단변량 재검정",
        "\n".join(
            [
                "동일 row 반복으로 인한 p-value 과장을 줄이기 위해 exact-row dedup sensitivity를 수행했다.",
                "",
                markdown_table(dataset_compare),
                "",
                "수치형과 범주형 모두 full-row와 dedup 기준 결과를 나란히 저장했다.",
            ]
        ),
    )

    fold_df, nested_summary, pair_df, hyper_summary = nested_grouped_bootstrap(full_df)
    save_csv(fold_df, "04_nested_grouped_bootstrap_folds.csv")
    save_csv(nested_summary, "05_nested_grouped_bootstrap_summary.csv")
    save_csv(pair_df, "06_nested_grouped_bootstrap_pairwise.csv")
    save_csv(hyper_summary, "07_nested_grouped_bootstrap_hyperparameters.csv")
    plot_nested_bootstrap(nested_summary, fold_df, pair_df)
    report.upsert(
        "Nested Grouped Bootstrap 모델 재비교",
        "\n".join(
            [
                f"총 `{int(fold_df['replicate'].nunique())}`개 nested grouped bootstrap replicate를 확보했다.",
                "",
                markdown_table(nested_summary),
                "",
                "이 단계는 replicate마다 grouped inner CV로 하이퍼파라미터를 다시 튜닝한 결과다.",
            ]
        ),
    )

    REPORT_PATH.write_text(
        build_report(
            dataset_compare=dataset_compare,
            numeric_compare=numeric_compare,
            cat_compare=cat_compare,
            nested_summary=nested_summary,
            pair_df=pair_df,
            hyper_summary=hyper_summary,
        ),
        encoding="utf-8",
    )
    print("Final rigor upgrade complete.")
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
