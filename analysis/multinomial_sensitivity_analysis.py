from __future__ import annotations

import os
import warnings
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "Sleep_health_and_lifestyle_dataset" / "Sleep_health_and_lifestyle_dataset.csv"
OUT_DIR = ROOT / "results" / "multinomial_sensitivity_study"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "multinomial_sensitivity_report_ko.md"


CLASS_ORDER = ["None", "Insomnia", "Sleep Apnea"]
FEATURES = [
    "age",
    "heart_rate",
    "map_bp",
    "pulse_pressure",
    "sleep_deficit_7h",
    "sleep_stress_balance",
    "male",
    "bmi_risk",
]
DISPLAY = {
    "age": "Age",
    "heart_rate": "Heart Rate",
    "map_bp": "Mean Arterial Pressure",
    "pulse_pressure": "Pulse Pressure",
    "sleep_deficit_7h": "Sleep Deficit (vs 7h)",
    "sleep_stress_balance": "Sleep-Stress Balance",
    "male": "Male",
    "bmi_risk": "BMI Risk",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def pretty(name: str) -> str:
    if name in DISPLAY:
        return DISPLAY[name]
    return name.replace("_", " ").title()


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda x: f"{x:.3f}")
    header = "| " + " | ".join(display_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in display_df.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def load_base_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).rename(
        columns={
            "Person ID": "person_id",
            "Gender": "gender",
            "Age": "age",
            "Sleep Duration": "sleep_duration",
            "Quality of Sleep": "quality_of_sleep",
            "Stress Level": "stress_level",
            "Heart Rate": "heart_rate",
            "BMI Category": "bmi_category",
            "Sleep Disorder": "sleep_disorder",
            "Blood Pressure": "blood_pressure",
        }
    )
    df["sleep_disorder"] = df["sleep_disorder"].fillna("None")
    df["bmi_category"] = df["bmi_category"].replace({"Normal Weight": "Normal"})
    df["male"] = (df["gender"] == "Male").astype(int)
    df["bmi_risk"] = (df["bmi_category"] != "Normal").astype(int)
    bp = df["blood_pressure"].str.split("/", expand=True).astype(int)
    df["systolic_bp"] = bp[0]
    df["diastolic_bp"] = bp[1]
    df["map_bp"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["sleep_deficit_7h"] = (7 - df["sleep_duration"]).clip(lower=0)
    df["sleep_stress_balance"] = df["quality_of_sleep"] - df["stress_level"]
    return df


def dataset_summary(full_df: pd.DataFrame, dedup_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in [("Full", full_df), ("Deduplicated", dedup_df)]:
        counts = df["sleep_disorder"].value_counts().reindex(CLASS_ORDER)
        rows.append(
            {
                "dataset": name,
                "n_rows": int(len(df)),
                "n_none": int(counts["None"]),
                "n_insomnia": int(counts["Insomnia"]),
                "n_sleep_apnea": int(counts["Sleep Apnea"]),
                "positive_share": float((df["sleep_disorder"] != "None").mean()),
            }
        )
    return pd.DataFrame(rows)


def class_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["age", "heart_rate", "map_bp", "pulse_pressure", "sleep_deficit_7h", "sleep_stress_balance"]
    out = df.groupby("sleep_disorder")[cols].mean().reindex(CLASS_ORDER).round(2)
    out.index.name = "sleep_disorder"
    return out.reset_index()


def anova_eta(groups: list[pd.Series]) -> float:
    all_values = pd.concat(groups)
    overall_mean = all_values.mean()
    ss_between = sum(len(group) * (group.mean() - overall_mean) ** 2 for group in groups)
    ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
    return float(ss_between / (ss_between + ss_within))


def anova_sensitivity(full_df: pd.DataFrame, dedup_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in [("Full", full_df), ("Deduplicated", dedup_df)]:
        for feature in ["age", "heart_rate", "map_bp", "pulse_pressure", "sleep_deficit_7h", "sleep_stress_balance"]:
            groups = [df.loc[df["sleep_disorder"] == cls, feature] for cls in CLASS_ORDER]
            f_stat, p_value = stats.f_oneway(*groups)
            rows.append(
                {
                    "dataset": name,
                    "feature": feature,
                    "feature_label": pretty(feature),
                    "f_stat": f_stat,
                    "p_value": p_value,
                    "eta_squared": anova_eta(groups),
                }
            )
    return pd.DataFrame(rows)


def fit_mnlogit(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["sleep_code"] = pd.Categorical(working["sleep_disorder"], categories=CLASS_ORDER).codes
    X = sm.add_constant(working[FEATURES], has_constant="add")
    model = sm.MNLogit(working["sleep_code"], X).fit(method="newton", maxiter=200, disp=False)
    params = model.params
    pvals = model.pvalues
    conf = model.conf_int()
    rows = []
    comparison_map = {0: "Insomnia vs None", 1: "Sleep Apnea vs None"}
    code_map = {0: "1", 1: "2"}
    for column in params.columns:
        for feature in params.index:
            if feature == "const":
                continue
            ci_row = conf.loc[(code_map[column], feature), :]
            coef = float(params.loc[feature, column])
            rows.append(
                {
                    "comparison": comparison_map[column],
                    "feature": feature,
                    "feature_label": pretty(feature),
                    "coefficient": coef,
                    "odds_ratio": float(np.exp(coef)),
                    "ci_low": float(np.exp(ci_row["lower"])),
                    "ci_high": float(np.exp(ci_row["upper"])),
                    "p_value": float(pvals.loc[feature, column]),
                }
            )
    return pd.DataFrame(rows).sort_values(["comparison", "p_value"])


def make_grid() -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=5000, class_weight="balanced")),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid={"clf__C": np.logspace(-4, 4, 21)},
        cv=5,
        scoring="f1_macro",
        n_jobs=None,
    )


def fit_predictive_model(df: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    X = df[FEATURES].copy()
    y = df["sleep_disorder"]
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(make_grid(), X, y, cv=outer_cv, scoring="f1_macro")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    fitted = make_grid()
    fitted.fit(X_train, y_train)
    preds = fitted.predict(X_test)
    probs = fitted.predict_proba(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    coef = pd.DataFrame(
        fitted.best_estimator_.named_steps["clf"].coef_,
        index=fitted.best_estimator_.named_steps["clf"].classes_,
        columns=FEATURES,
    )
    coef = coef.rename(columns=pretty)
    return {
        "dataset": dataset_name,
        "cv_macro_f1_mean": float(cv_scores.mean()),
        "cv_macro_f1_std": float(cv_scores.std()),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_macro_f1": float(f1_score(y_test, preds, average="macro")),
        "test_auc_ovr_macro": float(
            roc_auc_score(y_test, probs, multi_class="ovr", average="macro", labels=fitted.classes_)
        ),
        "best_c": float(fitted.best_params_["clf__C"]),
        "confusion": pd.crosstab(pd.Series(y_test, name="actual"), pd.Series(preds, name="predicted"))
        .reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0),
        "classification_report": pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}),
        "coefficients": coef.reset_index().rename(columns={"index": "class"}),
    }


def coefficient_stability_table(full_coef: pd.DataFrame, dedup_coef: pd.DataFrame) -> pd.DataFrame:
    full_long = full_coef.melt(id_vars="class", var_name="feature_label", value_name="full_coef")
    dedup_long = dedup_coef.melt(id_vars="class", var_name="feature_label", value_name="dedup_coef")
    merged = full_long.merge(dedup_long, on=["class", "feature_label"], how="inner")
    merged["sign_same"] = np.sign(merged["full_coef"]) == np.sign(merged["dedup_coef"])
    merged["abs_delta"] = (merged["full_coef"] - merged["dedup_coef"]).abs()
    return merged.sort_values(["class", "abs_delta"], ascending=[True, False])


def plot_class_balance(summary_df: pd.DataFrame) -> None:
    counts = summary_df.melt(
        id_vars="dataset",
        value_vars=["n_none", "n_insomnia", "n_sleep_apnea"],
        var_name="class",
        value_name="count",
    )
    counts["class"] = counts["class"].map(
        {"n_none": "None", "n_insomnia": "Insomnia", "n_sleep_apnea": "Sleep Apnea"}
    )
    plt.figure(figsize=(8.5, 5))
    sns.barplot(data=counts, x="class", y="count", hue="dataset", palette="Set2")
    plt.title("Class Balance: Full vs Deduplicated Data")
    plt.xlabel("Sleep Disorder")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_class_balance_full_vs_dedup.png", dpi=220)
    plt.close()


def plot_feature_boxplots(df: pd.DataFrame) -> None:
    features = ["age", "heart_rate", "map_bp", "pulse_pressure", "sleep_deficit_7h", "sleep_stress_balance"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, feature in zip(axes.flat, features, strict=False):
        sns.boxplot(
            data=df,
            x="sleep_disorder",
            y=feature,
            order=CLASS_ORDER,
            hue="sleep_disorder",
            palette="Set2",
            legend=False,
            ax=axis,
        )
        axis.set_title(pretty(feature))
        axis.set_xlabel("")
        axis.set_ylabel(pretty(feature))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_selected_features_by_class.png", dpi=220)
    plt.close()


def plot_mnlogit_odds_ratios(or_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, comparison in zip(axes, ["Insomnia vs None", "Sleep Apnea vs None"], strict=False):
        subset = or_df[or_df["comparison"] == comparison].copy().sort_values("odds_ratio")
        axis.errorbar(
            subset["odds_ratio"],
            subset["feature_label"],
            xerr=[subset["odds_ratio"] - subset["ci_low"], subset["ci_high"] - subset["odds_ratio"]],
            fmt="o",
            color="#1f77b4",
            ecolor="#9ec3e6",
            capsize=4,
        )
        axis.axvline(1, color="red", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_title(comparison)
        axis.set_xlabel("Odds Ratio (log scale)")
        axis.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_multinomial_odds_ratios_full.png", dpi=220)
    plt.close()


def plot_effect_size_sensitivity(effect_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=effect_df, x="feature_label", y="eta_squared", hue="dataset", palette="Set2")
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("")
    plt.ylabel("Eta squared")
    plt.title("ANOVA Effect Size Sensitivity: Full vs Deduplicated")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_effect_size_sensitivity.png", dpi=220)
    plt.close()


def plot_performance_sensitivity(metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df.melt(
        id_vars="dataset",
        value_vars=["cv_macro_f1_mean", "test_macro_f1", "test_auc_ovr_macro", "test_accuracy"],
        var_name="metric",
        value_name="value",
    )
    metric_map = {
        "cv_macro_f1_mean": "CV Macro-F1",
        "test_macro_f1": "Test Macro-F1",
        "test_auc_ovr_macro": "Test Macro-AUC",
        "test_accuracy": "Test Accuracy",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_map)
    plt.figure(figsize=(9.5, 5.5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="dataset", palette="Set2")
    plt.ylim(0, 1.0)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("Multinomial Logistic Sensitivity Performance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_multinomial_performance_sensitivity.png", dpi=220)
    plt.close()


def plot_coefficient_stability(full_coef: pd.DataFrame, dedup_coef: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, coef_df, title in zip(
        axes,
        [full_coef.set_index("class"), dedup_coef.set_index("class")],
        ["Full Data Coefficients", "Deduplicated Coefficients"],
        strict=False,
    ):
        sns.heatmap(coef_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_coefficient_stability_heatmap.png", dpi=220)
    plt.close()


def plot_confusion_matrices(full_confusion: pd.DataFrame, dedup_confusion: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, matrix, title in zip(
        axes,
        [full_confusion, dedup_confusion],
        ["Full Data", "Deduplicated Data"],
        strict=False,
    ):
        ConfusionMatrixDisplay(confusion_matrix=matrix.values, display_labels=CLASS_ORDER).plot(
            ax=axis, cmap="Blues", colorbar=False
        )
        axis.set_title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_confusion_matrices_full_vs_dedup.png", dpi=220)
    plt.close()


def save_tables(
    summary_df: pd.DataFrame,
    means_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    full_or: pd.DataFrame,
    dedup_or: pd.DataFrame,
    metrics_df: pd.DataFrame,
    coef_stability: pd.DataFrame,
    full_results: dict[str, object],
    dedup_results: dict[str, object],
) -> None:
    summary_df.to_csv(TABLE_DIR / "dataset_summary.csv", index=False)
    means_df.to_csv(TABLE_DIR / "class_means_full.csv", index=False)
    effect_df.to_csv(TABLE_DIR / "anova_sensitivity.csv", index=False)
    full_or.to_csv(TABLE_DIR / "mnlogit_full_odds_ratios.csv", index=False)
    dedup_or.to_csv(TABLE_DIR / "mnlogit_dedup_odds_ratios.csv", index=False)
    metrics_df.to_csv(TABLE_DIR / "multinomial_model_sensitivity.csv", index=False)
    coef_stability.to_csv(TABLE_DIR / "coefficient_stability.csv", index=False)
    full_results["classification_report"].to_csv(TABLE_DIR / "full_classification_report.csv", index=False)
    dedup_results["classification_report"].to_csv(TABLE_DIR / "dedup_classification_report.csv", index=False)
    full_results["coefficients"].to_csv(TABLE_DIR / "full_multinomial_coefficients.csv", index=False)
    dedup_results["coefficients"].to_csv(TABLE_DIR / "dedup_multinomial_coefficients.csv", index=False)


def build_report(
    summary_df: pd.DataFrame,
    means_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    full_or: pd.DataFrame,
    dedup_or: pd.DataFrame,
    metrics_df: pd.DataFrame,
    coef_stability: pd.DataFrame,
) -> None:
    significant_full = full_or[full_or["p_value"] < 0.05].loc[
        :, ["comparison", "feature_label", "odds_ratio", "ci_low", "ci_high", "p_value"]
    ].copy()
    significant_full["p_value"] = significant_full["p_value"].map(format_p)
    significant_dedup = dedup_or[dedup_or["p_value"] < 0.05].loc[
        :, ["comparison", "feature_label", "odds_ratio", "ci_low", "ci_high", "p_value"]
    ].copy()
    significant_dedup["p_value"] = significant_dedup["p_value"].map(format_p)
    effects_view = effect_df.loc[:, ["dataset", "feature_label", "eta_squared", "p_value"]].copy()
    effects_view["p_value"] = effects_view["p_value"].map(format_p)
    performance_view = metrics_df.copy()
    stable_signs = coef_stability.groupby("class")["sign_same"].mean().reset_index()
    stable_signs["sign_same"] = stable_signs["sign_same"].astype(float)

    report = f"""# 데이터셋 선택 근거, 다항 로지스틱 회귀, 민감도 분석 보고서

## 1. 왜 이 데이터셋을 선택했는가

`dataset` 폴더에는 두 종류의 데이터가 있었다.

1. `Sleep_health_and_lifestyle_dataset`
2. `FitBit Fitness Tracker Data`

통계분석 관점에서 본 분석 데이터로 `Sleep_health_and_lifestyle_dataset`를 선택한 이유는 아래와 같다.

1. **명시적 종속변수 존재**
   - `Sleep Disorder`가 직접 포함되어 있어 분류모형과 집단 비교가 가능하다.
   - 반면 Fitbit 데이터는 활동/수면 로그는 풍부하지만 수면장애 라벨이 없고, 첫 번째 데이터와 개인 단위 연결도 없다.
2. **예측변수 완전성**
   - 주요 예측변수에는 결측이 없고, `Sleep Disorder` 결측은 실제로 `None` 의미로 쓰여 재코딩 가능했다.
3. **연속형 + 범주형 변수가 함께 존재**
   - ANOVA, 카이제곱, 상관분석, 로지스틱 회귀를 한 데이터셋 안에서 일관되게 수행하기 좋다.
4. **이진/다항 분석 모두 가능한 타깃 구조**
   - `None / Insomnia / Sleep Apnea` 3범주가 있어 이진 로지스틱과 다항 로지스틱을 모두 적용할 수 있다.
5. **클래스 불균형이 극단적이지 않음**
   - 전체 데이터 기준 `None=219`, `Insomnia=77`, `Sleep Apnea=78`로, 다항 분류를 시도할 최소 구조를 갖춘다.

데이터 요약:

{markdown_table(summary_df)}

## 2. 왜 이 데이터셋이 로지스틱 회귀에 적합한가

로지스틱 회귀의 특성에 비추어 보면 이 데이터셋은 아래 이유로 적합하다.

1. **종속변수가 범주형**
   - 로지스틱 회귀는 연속형이 아니라 범주형 결과를 설명하는 모델이며, 이 데이터는 `수면장애 여부`와 `수면장애 subtype`을 모두 제공한다.
2. **설명변수가 해석 가능한 임상/생활습관 변수**
   - 나이, 심박수, 혈압, 수면시간, 수면질, 스트레스, BMI는 회귀계수 해석이 직관적이다.
3. **표본 수 대비 변수 수가 과도하지 않음**
   - 특히 압축형 파생변수를 사용하면 공선성을 줄이면서도 모델 규모를 관리할 수 있다.
4. **클래스 균형이 완전히 무너지지 않음**
   - 극단적 희소 클래스가 아니어서 softmax 기반 다항 로지스틱을 적용할 수 있다.

다만 완전한 이상적 조건은 아니다.

- `person_id`를 제외하면 동일 프로파일이 많이 반복되어 독립성 가정이 약해질 수 있다.
- 혈압과 수면 관련 변수들 사이 공선성이 존재한다.

이 한계를 보완하기 위해 이번 분석에서는

1. 혈압군과 수면-스트레스군을 압축한 파생변수 사용
2. 중복 프로파일 제거 전후 민감도 분석

을 함께 수행했다.

## 3. 분석 흐름

1. 다항 로지스틱 회귀에 투입할 **압축형 변수 세트**를 사용했다.
   - `Age`, `Heart Rate`, `Mean Arterial Pressure`, `Pulse Pressure`, `Sleep Deficit (vs 7h)`, `Sleep-Stress Balance`, `Male`, `BMI Risk`
2. 같은 구조를 사용해
   - `Full data`
   - `Deduplicated data`
   를 각각 적합했다.
3. 결과는 두 층위에서 해석했다.
   - `MNLogit`: subtype별 오즈비와 p-value
   - `Multinomial logistic classifier`: 예측 성능과 계수 안정성

## 4. 다항 로지스틱 회귀 결과

### 4.1 Full data에서 유의했던 변수

{markdown_table(significant_full)}

해석:

- `Insomnia vs None`에서는 `Pulse Pressure` 증가, `BMI Risk` 증가, `Sleep-Stress Balance` 저하가 주요 신호였다.
- `Sleep Apnea vs None`에서는 `Mean Arterial Pressure` 증가와 `Heart Rate` 증가가 더 강하게 나타났다.
- 즉, **불면증은 수면-스트레스 균형과 상대적 압력차**, **수면무호흡은 평균 혈압부담과 심혈관 부하** 쪽에 더 가깝게 연결된다.

### 4.2 Deduplicated data에서 유의했던 변수

{markdown_table(significant_dedup)}

해석:

- 표본 수가 줄어 p-value는 전반적으로 약해졌지만,
- `Pulse Pressure`, `Sleep-Stress Balance`, `BMI Risk`의 불면증 방향성,
- `Mean Arterial Pressure`의 수면무호흡 방향성은 유지되었다.

## 5. subtype별 원자료 평균

{markdown_table(means_df)}

원자료 수준 해석:

- `Sleep Apnea`는 평균적으로 가장 나이가 많고, 심박수와 평균동맥압이 가장 높았다.
- `Insomnia`는 `None` 대비 수면-스트레스 균형이 더 나빴다.
- `Sleep Deficit`는 `Insomnia`와 `Sleep Apnea` 모두 `None`보다 높았고, 특히 `Insomnia`에서 더 나쁜 패턴이 보였다.

## 6. 민감도 분석

이번 민감도 분석은 **중복 프로파일 제거 전후** 결과 비교다.

### 6.1 효과크기 민감도

{markdown_table(effects_view)}

해석:

- `Mean Arterial Pressure`, `Pulse Pressure`, `Heart Rate`, `Sleep Deficit`, `Sleep-Stress Balance`는 dedup 후에도 유의했다.
- 즉, 예측 성능은 줄어들더라도 **주요 변수와 수면장애 subtype 간의 통계적 분리 자체는 유지**되었다.

### 6.2 예측 성능 민감도

{markdown_table(performance_view)}

해석:

- `Full data`에서는 macro-F1과 accuracy가 높게 나왔다.
- 하지만 `Deduplicated data`에서는 성능이 뚜렷하게 하락했다.
- 따라서 **중복 프로파일이 모델 성능을 낙관적으로 보이게 했을 가능성**이 크다.

### 6.3 계수 방향 안정성

클래스별 계수 부호 일치 비율:

{markdown_table(stable_signs)}

해석:

- dedup 후에도 핵심 방향성은 꽤 유지되었다.
- 특히 `Sleep Apnea` 쪽의 `MAP` 양의 방향, `Insomnia` 쪽의 `Pulse Pressure` 양의 방향과 `Sleep-Stress Balance` 음의 방향은 비교적 안정적이었다.

## 7. 최종 정리

### 7.1 데이터셋 선택에 대한 결론

이 데이터셋은

1. 명시적 범주형 타깃이 있고
2. 예측변수 결측이 거의 없으며
3. 집단비교와 로지스틱 회귀를 함께 수행할 수 있다는 점에서

통계분석용으로 적합했다.

### 7.2 로지스틱 회귀 적합성에 대한 결론

특히 이 데이터는

1. `이진 로지스틱`으로는 수면장애 유무
2. `다항 로지스틱`으로는 `Insomnia / Sleep Apnea / None`

를 모두 다룰 수 있어 로지스틱 회귀 교육용/해석용 예제로도 적합하다.

### 7.3 다항 로지스틱과 민감도 분석을 통해 얻은 결론

- subtype 차이는 단순히 “수면장애가 있다/없다”보다 더 구체적이다.
- `Sleep Apnea`는 혈압부담(MAP)과 심혈관 패턴이 더 강하고,
- `Insomnia`는 수면-스트레스 불균형과 상대적 압력차(Pulse Pressure)와 더 가깝다.
- 다만 중복 제거 후 성능이 크게 낮아졌으므로, **예측 성능 수치는 보수적으로 해석해야 한다.**
- 반대로 주요 변수의 방향성과 효과크기는 상당 부분 유지되어, **어떤 feature가 중요한가**에 대한 결론은 비교적 견고하다.

## 8. 생성된 결과물

### 8.1 보고서

- `results/multinomial_sensitivity_study/multinomial_sensitivity_report_ko.md`

### 8.2 시각화

1. `results/multinomial_sensitivity_study/figures/01_class_balance_full_vs_dedup.png`
2. `results/multinomial_sensitivity_study/figures/02_selected_features_by_class.png`
3. `results/multinomial_sensitivity_study/figures/03_multinomial_odds_ratios_full.png`
4. `results/multinomial_sensitivity_study/figures/04_effect_size_sensitivity.png`
5. `results/multinomial_sensitivity_study/figures/05_multinomial_performance_sensitivity.png`
6. `results/multinomial_sensitivity_study/figures/06_coefficient_stability_heatmap.png`
7. `results/multinomial_sensitivity_study/figures/07_confusion_matrices_full_vs_dedup.png`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    full_df = load_base_data()
    dedup_df = full_df.drop(columns=["person_id"]).drop_duplicates().copy()

    summary_df = dataset_summary(full_df, dedup_df)
    means_df = class_mean_table(full_df)
    effect_df = anova_sensitivity(full_df, dedup_df)
    full_or = fit_mnlogit(full_df)
    dedup_or = fit_mnlogit(dedup_df)
    full_results = fit_predictive_model(full_df, "Full")
    dedup_results = fit_predictive_model(dedup_df, "Deduplicated")
    metrics_df = pd.DataFrame(
        [
            {
                "dataset": full_results["dataset"],
                "cv_macro_f1_mean": full_results["cv_macro_f1_mean"],
                "cv_macro_f1_std": full_results["cv_macro_f1_std"],
                "test_accuracy": full_results["test_accuracy"],
                "test_macro_f1": full_results["test_macro_f1"],
                "test_auc_ovr_macro": full_results["test_auc_ovr_macro"],
                "best_c": full_results["best_c"],
            },
            {
                "dataset": dedup_results["dataset"],
                "cv_macro_f1_mean": dedup_results["cv_macro_f1_mean"],
                "cv_macro_f1_std": dedup_results["cv_macro_f1_std"],
                "test_accuracy": dedup_results["test_accuracy"],
                "test_macro_f1": dedup_results["test_macro_f1"],
                "test_auc_ovr_macro": dedup_results["test_auc_ovr_macro"],
                "best_c": dedup_results["best_c"],
            },
        ]
    )
    coef_stability = coefficient_stability_table(full_results["coefficients"], dedup_results["coefficients"])

    save_tables(summary_df, means_df, effect_df, full_or, dedup_or, metrics_df, coef_stability, full_results, dedup_results)

    plot_class_balance(summary_df)
    plot_feature_boxplots(full_df)
    plot_mnlogit_odds_ratios(full_or)
    plot_effect_size_sensitivity(effect_df)
    plot_performance_sensitivity(metrics_df)
    plot_coefficient_stability(full_results["coefficients"], dedup_results["coefficients"])
    plot_confusion_matrices(full_results["confusion"], dedup_results["confusion"])

    build_report(summary_df, means_df, effect_df, full_or, dedup_or, metrics_df, coef_stability)

    print("Multinomial sensitivity analysis complete.")
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
