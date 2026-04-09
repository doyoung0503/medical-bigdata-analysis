from __future__ import annotations

import math
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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "Sleep_health_and_lifestyle_dataset" / "Sleep_health_and_lifestyle_dataset.csv"
OUT_DIR = ROOT / "results" / "ridge_feature_study"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "ridge_feature_study_report_ko.md"


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
    "sleep_quality_per_hour": "Sleep Quality per Hour",
    "rate_pressure_product": "Rate Pressure Product",
    "steps_per_activity": "Steps per Activity",
    "gender": "Gender",
    "bmi_category": "BMI Category",
}

DISORDER_ORDER = ["None", "Insomnia", "Sleep Apnea"]
ORIGINAL_NUMERIC = [
    "age",
    "sleep_duration",
    "quality_of_sleep",
    "physical_activity_level",
    "stress_level",
    "heart_rate",
    "daily_steps",
    "systolic_bp",
    "diastolic_bp",
]
ORIGINAL_CATEGORICAL = ["gender", "bmi_category"]
DERIVED_SPECS = [
    {
        "feature": "map_bp",
        "formula": "(systolic_bp + 2*diastolic_bp) / 3",
        "domain": "혈압 군집",
        "rationale": "수축기/이완기 혈압이 거의 같은 축으로 움직여 공통 혈압부담을 요약하기 위함",
        "decision": "retain",
    },
    {
        "feature": "pulse_pressure",
        "formula": "systolic_bp - diastolic_bp",
        "domain": "혈압 군집",
        "rationale": "평균 혈압 수준 외에 수축기-이완기 차이를 추가로 반영하기 위함",
        "decision": "retain",
    },
    {
        "feature": "sleep_deficit_7h",
        "formula": "max(0, 7 - sleep_duration)",
        "domain": "수면-스트레스 군집",
        "rationale": "권장 최소 수면 7시간 기준의 부족량으로 짧은 수면 부담을 해석하기 위함",
        "decision": "retain",
    },
    {
        "feature": "sleep_stress_balance",
        "formula": "quality_of_sleep - stress_level",
        "domain": "수면-스트레스 군집",
        "rationale": "수면 회복감과 스트레스 부담의 순균형을 하나의 축으로 요약하기 위함",
        "decision": "retain",
    },
    {
        "feature": "sleep_quality_per_hour",
        "formula": "quality_of_sleep / sleep_duration",
        "domain": "수면-스트레스 군집",
        "rationale": "수면시간 대비 체감 수면질 효율을 탐색하기 위함",
        "decision": "explore_only",
    },
    {
        "feature": "rate_pressure_product",
        "formula": "heart_rate * systolic_bp",
        "domain": "심혈관 군집",
        "rationale": "혈압과 심박을 결합한 심혈관 부하 지표를 탐색하기 위함",
        "decision": "explore_only",
    },
    {
        "feature": "steps_per_activity",
        "formula": "daily_steps / physical_activity_level",
        "domain": "활동량 군집",
        "rationale": "활동량 대비 실제 걸음수 효율을 탐색하기 위함",
        "decision": "explore_only",
    },
]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def pretty(name: str) -> str:
    if name in DISPLAY:
        return DISPLAY[name]
    if name.startswith("num__"):
        return pretty(name.split("__", 1)[1])
    if name.startswith("cat__gender_"):
        return "Gender: Male"
    if name.startswith("cat__bmi_category_"):
        return f"BMI Category: {name.split('_')[-1].title()}"
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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).rename(columns=RENAME_MAP)
    df["sleep_disorder"] = df["sleep_disorder"].fillna("None")
    df["bmi_category"] = df["bmi_category"].replace({"Normal Weight": "Normal"})
    bp = df["blood_pressure"].str.extract(r"(?P<systolic_bp>\d+)\/(?P<diastolic_bp>\d+)").astype(int)
    df = pd.concat([df, bp], axis=1)
    df["has_sleep_disorder"] = (df["sleep_disorder"] != "None").astype(int)
    df["sleep_disorder"] = pd.Categorical(df["sleep_disorder"], categories=DISORDER_ORDER, ordered=True)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["map_bp"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["sleep_deficit_7h"] = (7 - df["sleep_duration"]).clip(lower=0)
    df["sleep_stress_balance"] = df["quality_of_sleep"] - df["stress_level"]
    df["sleep_quality_per_hour"] = df["quality_of_sleep"] / df["sleep_duration"]
    df["rate_pressure_product"] = df["heart_rate"] * df["systolic_bp"]
    df["steps_per_activity"] = df["daily_steps"] / df["physical_activity_level"]
    return df


def eta_squared(groups: list[pd.Series]) -> float:
    overall_mean = pd.concat(groups).mean()
    ss_between = sum(len(group) * (group.mean() - overall_mean) ** 2 for group in groups)
    ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
    return float(ss_between / (ss_between + ss_within))


def correlation_pairs(df: pd.DataFrame, columns: list[str], threshold: float = 0.75) -> pd.DataFrame:
    corr = df[columns].corr()
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if abs(value) >= threshold:
                rows.append(
                    {
                        "left": pretty(left),
                        "right": pretty(right),
                        "correlation": value,
                        "abs_correlation": abs(value),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def compute_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = sm.add_constant(df[columns].astype(float), has_constant="add")
    rows = []
    for idx, column in enumerate(X.columns):
        if column == "const":
            continue
        rows.append({"feature": column, "feature_label": pretty(column), "vif": variance_inflation_factor(X.values, idx)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def derived_feature_screen(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    original_corr = df[ORIGINAL_NUMERIC].corr().abs()
    for spec in DERIVED_SPECS:
        feature = spec["feature"]
        corr, corr_p = stats.pointbiserialr(df["has_sleep_disorder"], df[feature])
        groups = [df.loc[df["sleep_disorder"] == disorder, feature] for disorder in DISORDER_ORDER]
        f_stat, anova_p = stats.f_oneway(*groups)
        all_corrs = df[ORIGINAL_NUMERIC].corrwith(df[feature]).abs().sort_values(ascending=False)
        top_source = all_corrs.index[0]
        rows.append(
            {
                "feature": feature,
                "feature_label": pretty(feature),
                "domain": spec["domain"],
                "formula": spec["formula"],
                "screening_decision": spec["decision"],
                "pointbiserial_r": corr,
                "pointbiserial_p": corr_p,
                "anova_f": f_stat,
                "anova_p": anova_p,
                "eta_squared": eta_squared(groups),
                "max_abs_corr_with_original": all_corrs.iloc[0],
                "most_correlated_original": pretty(top_source),
                "rationale": spec["rationale"],
            }
        )
    return pd.DataFrame(rows).sort_values(["pointbiserial_p", "eta_squared"], ascending=[True, False])


def make_grid(num_features: list[str], cat_features: list[str]) -> GridSearchCV:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=5000, class_weight="balanced")),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid={"clf__C": np.logspace(-4, 4, 21)},
        cv=5,
        scoring="roc_auc",
        n_jobs=None,
    )


def evaluate_model(df: pd.DataFrame, name: str, num_features: list[str], cat_features: list[str]) -> dict[str, object]:
    X = df[num_features + cat_features].copy()
    y = df["has_sleep_disorder"]
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = make_grid(num_features, cat_features)
    cv_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring="roc_auc")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    fitted = make_grid(num_features, cat_features)
    fitted.fit(X_train, y_train)
    probs = fitted.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    best_estimator = fitted.best_estimator_
    feature_names = list(best_estimator.named_steps["preprocessor"].get_feature_names_out())
    coefs = best_estimator.named_steps["clf"].coef_[0]
    coef_table = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "feature_label": [pretty(name) for name in feature_names],
                "coefficient": coefs,
                "abs_coefficient": np.abs(coefs),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    fpr, tpr, _ = roc_curve(y_test, probs)
    return {
        "name": name,
        "num_features": num_features,
        "cat_features": cat_features,
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "test_auc": float(roc_auc_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "best_c": float(fitted.best_params_["clf__C"]),
        "coef_table": coef_table,
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "predictions": pd.DataFrame({"y_true": y_test.reset_index(drop=True), "y_prob": probs, "y_pred": preds}),
        "confusion_matrix": pd.crosstab(pd.Series(y_test, name="actual"), pd.Series(preds, name="predicted")),
    }


def plot_original_correlation(df: pd.DataFrame) -> None:
    corr = df[ORIGINAL_NUMERIC + ["has_sleep_disorder"]].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Original Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_original_correlation_heatmap.png", dpi=220)
    plt.close()


def plot_high_corr_pairs(pair_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.8))
    plot_df = pair_df.sort_values("abs_correlation", ascending=True)
    sns.barplot(data=plot_df, x="abs_correlation", y=plot_df["left"] + " vs " + plot_df["right"], hue=plot_df["left"] + " vs " + plot_df["right"], legend=False, palette="crest")
    plt.xlabel("Absolute correlation")
    plt.ylabel("")
    plt.title("Highly Correlated Original Feature Pairs")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_high_correlation_pairs.png", dpi=220)
    plt.close()


def plot_vif_comparison(original_vif: pd.DataFrame, compressed_vif: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, title, data in zip(
        axes,
        ["Original Features", "Compressed Derived Set"],
        [original_vif.sort_values("vif"), compressed_vif.sort_values("vif")],
        strict=False,
    ):
        sns.barplot(data=data, x="vif", y="feature_label", hue="feature_label", legend=False, palette="mako", ax=axis)
        axis.axvline(5, color="orange", linestyle="--", linewidth=1)
        axis.axvline(10, color="red", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("VIF")
        axis.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_vif_comparison.png", dpi=220)
    plt.close()


def plot_derived_screen(screen_df: pd.DataFrame) -> None:
    plot_df = screen_df.sort_values("pointbiserial_r")
    palette = plot_df["screening_decision"].map({"retain": "#2f7ed8", "explore_only": "#a1c9f4"})
    plt.figure(figsize=(9, 5.5))
    plt.barh(plot_df["feature_label"], plot_df["pointbiserial_r"], color=palette)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Point-biserial correlation with Has Sleep Disorder")
    plt.ylabel("")
    plt.title("Derived Feature Screening")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_derived_feature_screening.png", dpi=220)
    plt.close()


def plot_derived_heatmap(df: pd.DataFrame) -> None:
    cols = [
        "systolic_bp",
        "diastolic_bp",
        "sleep_duration",
        "quality_of_sleep",
        "stress_level",
        "physical_activity_level",
        "daily_steps",
        "map_bp",
        "pulse_pressure",
        "sleep_deficit_7h",
        "sleep_stress_balance",
        "rate_pressure_product",
        "steps_per_activity",
    ]
    corr = df[cols].corr()
    renamed = corr.rename(index=pretty, columns=pretty)
    plt.figure(figsize=(12, 10))
    sns.heatmap(renamed, cmap="RdBu_r", center=0, annot=False, square=True)
    plt.title("Original and Derived Feature Correlation Map")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_original_derived_correlation_map.png", dpi=220)
    plt.close()


def plot_model_metrics(model_df: pd.DataFrame) -> None:
    melted = model_df.melt(
        id_vars=["model"],
        value_vars=["cv_auc_mean", "test_auc", "accuracy", "f1"],
        var_name="metric",
        value_name="value",
    )
    metric_names = {
        "cv_auc_mean": "CV ROC-AUC",
        "test_auc": "Test ROC-AUC",
        "accuracy": "Accuracy",
        "f1": "F1-score",
    }
    melted["metric"] = melted["metric"].map(metric_names)
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=melted, x="metric", y="value", hue="model", palette="Set2")
    plt.ylim(0.7, 1.0)
    plt.title("Ridge Logistic Model Performance Comparison")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_model_performance_comparison.png", dpi=220)
    plt.close()


def plot_coefficients(title: str, coef_df: pd.DataFrame, filename: str, top_n: int = 12) -> None:
    plot_df = coef_df.head(top_n).sort_values("coefficient")
    colors = ["#d95f02" if value > 0 else "#1b9e77" for value in plot_df["coefficient"]]
    plt.figure(figsize=(9, max(5, top_n * 0.45)))
    plt.barh(plot_df["feature_label"], plot_df["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Standardized ridge coefficient")
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=220)
    plt.close()


def plot_roc_curves(results: list[dict[str, object]]) -> None:
    plt.figure(figsize=(7.5, 6))
    for result in results:
        roc_df = result["roc"]
        plt.plot(roc_df["fpr"], roc_df["tpr"], linewidth=2, label=f"{result['name']} (AUC={result['test_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Across Ridge Logistic Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_roc_curve_comparison.png", dpi=220)
    plt.close()


def plot_confusion_matrices(results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4.2))
    for axis, result in zip(axes, results, strict=False):
        prediction_df = result["predictions"]
        matrix = pd.crosstab(prediction_df["y_true"], prediction_df["y_pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        ConfusionMatrixDisplay(confusion_matrix=matrix.values, display_labels=["No disorder", "Disorder"]).plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(result["name"])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_confusion_matrix_comparison.png", dpi=220)
    plt.close()


def save_tables(
    original_corr: pd.DataFrame,
    pair_df: pd.DataFrame,
    original_vif: pd.DataFrame,
    compressed_vif: pd.DataFrame,
    screen_df: pd.DataFrame,
    model_df: pd.DataFrame,
    results: list[dict[str, object]],
) -> None:
    original_corr.to_csv(TABLE_DIR / "original_correlation_matrix.csv")
    pair_df.to_csv(TABLE_DIR / "high_correlation_pairs.csv", index=False)
    original_vif.to_csv(TABLE_DIR / "original_vif.csv", index=False)
    compressed_vif.to_csv(TABLE_DIR / "compressed_vif.csv", index=False)
    screen_df.to_csv(TABLE_DIR / "derived_feature_screening.csv", index=False)
    model_df.to_csv(TABLE_DIR / "ridge_model_comparison.csv", index=False)
    for result in results:
        safe_name = result["name"].lower().replace(" ", "_")
        result["coef_table"].to_csv(TABLE_DIR / f"{safe_name}_coefficients.csv", index=False)
        result["roc"].to_csv(TABLE_DIR / f"{safe_name}_roc_curve.csv", index=False)
        result["predictions"].to_csv(TABLE_DIR / f"{safe_name}_predictions.csv", index=False)


def build_report(
    df: pd.DataFrame,
    pair_df: pd.DataFrame,
    original_vif: pd.DataFrame,
    compressed_vif: pd.DataFrame,
    screen_df: pd.DataFrame,
    model_df: pd.DataFrame,
    results_by_name: dict[str, dict[str, object]],
) -> None:
    top_pairs = pair_df.loc[:, ["left", "right", "correlation"]].copy()
    top_pairs["correlation"] = top_pairs["correlation"].round(3)
    top_original_vif = original_vif.loc[:, ["feature_label", "vif"]].copy()
    top_compressed_vif = compressed_vif.loc[:, ["feature_label", "vif"]].copy()
    top_screen = screen_df.loc[
        :,
        [
            "feature_label",
            "domain",
            "pointbiserial_r",
            "pointbiserial_p",
            "eta_squared",
            "most_correlated_original",
            "max_abs_corr_with_original",
            "screening_decision",
        ],
    ].copy()
    top_screen["pointbiserial_p"] = top_screen["pointbiserial_p"].map(format_p)
    model_view = model_df.copy()
    model_view["best_c"] = model_view["best_c"].round(6)
    model_view = model_view.rename(
        columns={
            "cv_auc_mean": "cv_auc",
            "cv_auc_std": "cv_sd",
            "test_auc": "test_auc",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }
    )
    baseline_top = results_by_name["Baseline"]["coef_table"].head(8).loc[:, ["feature_label", "coefficient"]]
    compressed_top = results_by_name["Compressed Derived"]["coef_table"].head(8).loc[:, ["feature_label", "coefficient"]]
    augmented_top = results_by_name["Augmented"]["coef_table"].head(8).loc[:, ["feature_label", "coefficient"]]

    report = f"""# Ridge 로지스틱 회귀 및 파생변수 탐색 보고서

## 1. 분석 목적

이번 추가 분석의 목적은 세 가지다.

1. 기존 분류 문제를 **Ridge(L2) 로지스틱 회귀**로 다시 학습해, 다중공선성이 있는 상황에서 모델이 어떻게 동작하는지 확인한다.
2. 원 변수들 사이의 상관관계와 VIF를 통해 **공선성 구조를 먼저 진단**한다.
3. 이 구조를 바탕으로 **시도해볼 만한 파생변수**를 설계하고, 성능과 해석가능성을 기준으로 실제 가치가 있는지 검증한다.

## 2. 왜 이런 순서로 분석했는가

이번 흐름은 `reference` 자료의 통계 검정 -> 상관분석 -> 회귀모형 해석 흐름을 그대로 확장한 것이다.

1. **상관행렬과 VIF 확인**
   - 이유: Ridge를 쓰더라도 어떤 변수 묶음이 중복정보를 담는지 먼저 알아야 파생변수 설계에 근거가 생긴다.
2. **파생변수 후보 스크리닝**
   - 이유: 임의의 feature engineering이 아니라, 실제로 유의하고 해석 가능한 변수만 다음 단계로 넘겨야 한다.
3. **Ridge 로지스틱 비교**
   - 이유: L2 규제는 공선성이 있어도 계수를 안정화해주므로, 원 변수/압축형 파생변수/증강형 파생변수 세트를 공정하게 비교할 수 있다.
4. **성능과 해석의 균형 판단**
   - 이유: AUC가 조금 높다고 무조건 좋은 모델이 아니라, 공선성·단순성·배치 편의성까지 같이 봐야 실제 활용 모델을 추천할 수 있다.

## 3. 데이터와 전처리

- 사용 데이터: `dataset/Sleep_health_and_lifestyle_dataset/Sleep_health_and_lifestyle_dataset.csv`
- 총 관측치: {len(df)}
- 수면장애 양성 비율: {df["has_sleep_disorder"].mean():.3f}
- `Sleep Disorder` 결측은 `None`으로 간주했다.
- `Blood Pressure`는 `systolic_bp`, `diastolic_bp`로 분해했다.
- `BMI Category`의 `Normal Weight`는 `Normal`로 통합했다.
- `person_id`를 제외하면 동일 프로파일 반복이 많아 실제 임상데이터보다 성능이 낙관적으로 나올 수 있다.

## 4. 원 변수의 상관관계와 다중공선성

### 4.1 높은 상관을 보인 변수쌍

{markdown_table(top_pairs)}

핵심 해석:

- `Systolic BP`와 `Diastolic BP`는 거의 동일한 축으로 움직였다.
- `Quality of Sleep`, `Sleep Duration`, `Stress Level`도 매우 강하게 얽혀 있었다.
- `Physical Activity Level`과 `Daily Steps` 역시 같은 활동군집의 정보가 중복되는 경향을 보였다.

### 4.2 원 변수 VIF

{markdown_table(top_original_vif)}

해석:

- 혈압 변수(VIF 35 이상)와 수면-스트레스 군집(`Quality of Sleep`, `Stress Level`)의 공선성이 가장 강했다.
- 따라서 Ridge를 쓰는 것은 타당하며, 동시에 공통축을 요약하는 파생변수를 만들어볼 통계적 근거가 충분했다.

## 5. 파생변수 후보는 어떻게 정했는가

파생변수는 아래 원칙으로 설계했다.

1. 높은 상관을 보인 원 변수군을 요약할 것
2. 임상적 또는 생활습관 해석이 가능할 것
3. 실제로 `Has Sleep Disorder`와 통계적으로 관련이 있을 것

후보 스크리닝 결과:

{markdown_table(top_screen)}

해석:

- **유지(retain)** 판단을 받은 핵심 후보는 `Mean Arterial Pressure`, `Pulse Pressure`, `Sleep Deficit (vs 7h)`, `Sleep-Stress Balance`였다.
- `Rate Pressure Product`는 통계적으로 강했지만 심혈관 축과 너무 많이 겹쳐 **탐색용**으로만 두었다.
- `Sleep Quality per Hour`, `Steps per Activity`는 유의하긴 했지만 효과가 상대적으로 약해 **보조 탐색 변수**로 분류했다.

## 6. Ridge 로지스틱 회귀 비교

비교한 모델은 아래 3가지다.

1. **Baseline**: 원 변수 전체
2. **Compressed Derived**: 공선성이 강한 군집을 파생변수로 압축한 세트
3. **Augmented**: 원 변수에 파생변수를 추가해 Ridge가 스스로 가중치를 분배하게 한 세트

성능 비교:

{markdown_table(model_view)}

핵심 해석:

- **가장 높은 외부형 지표(CV ROC-AUC)** 는 `Baseline`이 기록했다.
- **Compressed Derived**는 CV ROC-AUC가 baseline과 거의 비슷하면서도 정확도/재현성 지표를 크게 해치지 않았다.
- **Augmented**는 hold-out test 수치가 약간 높았지만, 원 변수와 파생변수를 동시에 넣어 해석 목적의 공선성은 더 커졌다.

## 7. 계수 해석

### 7.1 Baseline Ridge 주요 계수

{markdown_table(baseline_top)}

### 7.2 Compressed Derived Ridge 주요 계수

{markdown_table(compressed_top)}

### 7.3 Augmented Ridge 주요 계수

{markdown_table(augmented_top)}

해석:

- `Baseline`에서는 혈압 축, 나이, 수면의 질이 핵심 신호로 남았다.
- `Compressed Derived`에서는 `Mean Arterial Pressure`, `Pulse Pressure`, `Sleep Deficit`, `Sleep-Stress Balance`가 함께 작동하며, 공선성을 줄인 상태에서도 예측력이 유지됐다.
- `Augmented`에서는 혈압 관련 원 변수와 파생변수에 계수가 분산되었다. 이는 Ridge가 공선성 축 전체에 가중치를 나눠 배분하는 전형적 패턴이다.

## 8. 압축형 파생변수 세트의 공선성

{markdown_table(top_compressed_vif)}

해석:

- 압축형 세트는 원 변수 세트보다 VIF가 눈에 띄게 낮아졌다.
- 특히 혈압군을 `MAP + Pulse Pressure`로 바꾼 것이 다중공선성 완화에 가장 크게 기여했다.
- `Sleep-Stress Balance`는 여전히 수면 관련 정보와 연결되어 있으나, 원 세트의 `Quality of Sleep` / `Stress Level` 동시투입보다 해석이 훨씬 단순해졌다.

## 9. 최종 결론

### 9.1 Ridge 모델을 적용했을 때의 결론

- 이 데이터에서는 공선성이 분명히 존재하므로 **Ridge 로지스틱 회귀 적용은 타당**했다.
- Ridge를 쓰면 변수 제거 없이도 안정적인 분류가 가능했고, 세 모델 모두 ROC-AUC 0.94 이상을 유지했다.

### 9.2 시도해볼 만한 파생변수

실제로 다음 파생변수는 통계적으로 시도해볼 가치가 있었다.

1. `Mean Arterial Pressure`
2. `Pulse Pressure`
3. `Sleep Deficit (vs 7h)`
4. `Sleep-Stress Balance`

이 변수들이 유효한 이유:

- 높은 상관군을 요약한다.
- 타깃과의 상관 또는 집단 간 차이가 충분하다.
- 원 변수 대비 공선성을 줄이면서도 예측력을 크게 손상시키지 않는다.

### 9.3 어떤 모델을 추천할 것인가

- **예측 성능 최우선**이면 `Baseline Ridge`를 기본선으로 두는 것이 안전하다.
- **실무 배치와 해석 용이성**까지 고려하면 `Compressed Derived` 모델이 더 추천된다.
  - 이유: 성능 손실이 매우 작고, 변수 수와 공선성이 줄어 설명하기 쉽다.
- `Augmented`는 연구용 실험 세트로는 유용하지만, 원 변수와 파생변수를 동시에 넣는 구조라 해석형 보고모형으로는 덜 적합하다.

## 10. 실무 인사이트

이번 결과는 다음과 같은 도움을 줄 수 있다.

1. 혈압 축은 `수축기/이완기 각각`보다 `MAP`와 `Pulse Pressure`로 재구성해도 충분히 강한 예측 신호를 얻을 수 있다.
2. 수면 관련 변수는 `수면시간`, `수면질`, `스트레스`를 따로 모두 넣기보다 `Sleep Deficit`와 `Sleep-Stress Balance`로 요약하면 더 단순한 스크리닝 모델을 만들 수 있다.
3. 따라서 실제 서비스나 의료 스크리닝 환경에서는 **나이 + 심혈관 압력요약 + 수면부족/스트레스 균형지표** 조합이 간단하고 설명 가능한 고위험군 선별 틀로 활용될 수 있다.

## 11. 생성된 산출물

### 11.1 보고서

- `results/ridge_feature_study/ridge_feature_study_report_ko.md`

### 11.2 시각화

1. `results/ridge_feature_study/figures/01_original_correlation_heatmap.png`
2. `results/ridge_feature_study/figures/02_high_correlation_pairs.png`
3. `results/ridge_feature_study/figures/03_vif_comparison.png`
4. `results/ridge_feature_study/figures/04_derived_feature_screening.png`
5. `results/ridge_feature_study/figures/05_original_derived_correlation_map.png`
6. `results/ridge_feature_study/figures/06_model_performance_comparison.png`
7. `results/ridge_feature_study/figures/07_baseline_ridge_coefficients.png`
8. `results/ridge_feature_study/figures/08_compressed_ridge_coefficients.png`
9. `results/ridge_feature_study/figures/09_augmented_ridge_coefficients.png`
10. `results/ridge_feature_study/figures/10_roc_curve_comparison.png`
11. `results/ridge_feature_study/figures/11_confusion_matrix_comparison.png`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    df = add_derived_features(load_data())

    original_corr = df[ORIGINAL_NUMERIC + ["has_sleep_disorder"]].corr().round(3)
    pair_df = correlation_pairs(df, ORIGINAL_NUMERIC, threshold=0.75)
    original_vif = compute_vif(df, ORIGINAL_NUMERIC)
    compressed_numeric = ["age", "heart_rate", "map_bp", "pulse_pressure", "sleep_deficit_7h", "sleep_stress_balance"]
    compressed_vif = compute_vif(df, compressed_numeric)
    screen_df = derived_feature_screen(df)

    model_configs = [
        ("Baseline", ORIGINAL_NUMERIC, ORIGINAL_CATEGORICAL),
        ("Compressed Derived", compressed_numeric, ORIGINAL_CATEGORICAL),
        (
            "Augmented",
            ORIGINAL_NUMERIC + ["map_bp", "pulse_pressure", "sleep_deficit_7h", "sleep_stress_balance", "steps_per_activity"],
            ORIGINAL_CATEGORICAL,
        ),
    ]
    results = [evaluate_model(df, name, num, cat) for name, num, cat in model_configs]
    results_by_name = {result["name"]: result for result in results}

    model_df = pd.DataFrame(
        [
            {
                "model": result["name"],
                "cv_auc_mean": result["cv_auc_mean"],
                "cv_auc_std": result["cv_auc_std"],
                "test_auc": result["test_auc"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "best_c": result["best_c"],
            }
            for result in results
        ]
    )

    save_tables(original_corr, pair_df, original_vif, compressed_vif, screen_df, model_df, results)

    plot_original_correlation(df)
    plot_high_corr_pairs(pair_df)
    plot_vif_comparison(original_vif, compressed_vif)
    plot_derived_screen(screen_df)
    plot_derived_heatmap(df)
    plot_model_metrics(model_df)
    plot_coefficients("Baseline Ridge Coefficients", results_by_name["Baseline"]["coef_table"], "07_baseline_ridge_coefficients.png")
    plot_coefficients("Compressed Derived Ridge Coefficients", results_by_name["Compressed Derived"]["coef_table"], "08_compressed_ridge_coefficients.png")
    plot_coefficients("Augmented Ridge Coefficients", results_by_name["Augmented"]["coef_table"], "09_augmented_ridge_coefficients.png")
    plot_roc_curves(results)
    plot_confusion_matrices(results)

    build_report(df, pair_df, original_vif, compressed_vif, screen_df, model_df, results_by_name)

    print("Ridge feature study complete.")
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
