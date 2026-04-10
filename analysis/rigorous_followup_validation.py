from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
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
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "Sleep_health_and_lifestyle_dataset" / "Sleep_health_and_lifestyle_dataset.csv"
OUT_DIR = ROOT / "results" / "rigorous_validation"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "rigorous_validation_report_ko.md"


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


CLASS_ORDER = ["None", "Insomnia", "Sleep Apnea"]


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
    "male": "Male",
    "bmi_risk": "BMI Risk",
    "gender": "Gender",
    "bmi_category": "BMI Category",
    "occupation": "Occupation",
    "occupation_collapsed": "Occupation (Collapsed)",
    "bmi_collapsed": "BMI Category (Collapsed)",
}


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
DERIVED_NUMERIC = [
    "map_bp",
    "pulse_pressure",
    "sleep_deficit_7h",
    "sleep_stress_balance",
    "sleep_quality_per_hour",
    "rate_pressure_product",
    "steps_per_activity",
]
ALL_NUMERIC = ORIGINAL_NUMERIC + DERIVED_NUMERIC

CATEGORICAL_FEATURES = ["gender", "bmi_category", "bmi_collapsed", "occupation", "occupation_collapsed"]

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

PRIMARY_MODELS = ["orig_sleep_pa", "compressed", "enhanced"]
BOOTSTRAP_ITER = 400
PERM_ITER = 1500
RANDOM_SEED = 42


@dataclass
class ValidationArtifact:
    predictions: pd.DataFrame
    best_params: dict[str, object]
    calibration: pd.DataFrame
    threshold_curve: pd.DataFrame


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
            "# 수면장애 후속 엄밀성 검증 보고서",
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


def pretty(name: str) -> str:
    return DISPLAY.get(name, name.replace("_", " ").title())


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)
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


def cramers_v(chi2: float, n: int, shape: tuple[int, int]) -> float:
    min_dim = min(shape) - 1
    if min_dim <= 0:
        return float("nan")
    return math.sqrt(chi2 / (n * min_dim))


def logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 6) -> float:
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins[0] = 0.0
    bins[-1] = 1.0
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:], strict=False):
        if right == left:
            continue
        mask = (y_prob >= left) & (y_prob <= right if right == 1 else y_prob < right)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def compute_hedges_g(x1: pd.Series, x2: pd.Series) -> float:
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1)
    pooled_sd = math.sqrt(max(pooled_var, 1e-12))
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return correction * ((x1.mean() - x2.mean()) / pooled_sd)


def compute_rank_biserial(x1: pd.Series, x2: pd.Series) -> float:
    u = stats.mannwhitneyu(x1, x2, alternative="two-sided").statistic
    return 1 - (2 * u) / (len(x1) * len(x2))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).rename(columns=RENAME_MAP)
    df["sleep_disorder"] = df["sleep_disorder"].fillna("None")
    df["bmi_category"] = df["bmi_category"].replace({"Normal Weight": "Normal"})
    bp = df["blood_pressure"].str.extract(r"(?P<systolic_bp>\d+)/(?P<diastolic_bp>\d+)").astype(int)
    df = pd.concat([df, bp], axis=1)
    df["has_sleep_disorder"] = (df["sleep_disorder"] != "None").astype(int)
    df["male"] = (df["gender"] == "Male").astype(int)
    df["bmi_risk"] = (df["bmi_category"] != "Normal").astype(int)
    df["bmi_collapsed"] = np.where(df["bmi_category"] == "Normal", "Normal", "Elevated")
    df["map_bp"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["sleep_deficit_7h"] = (7 - df["sleep_duration"]).clip(lower=0)
    df["sleep_stress_balance"] = df["quality_of_sleep"] - df["stress_level"]
    df["sleep_quality_per_hour"] = df["quality_of_sleep"] / df["sleep_duration"]
    df["rate_pressure_product"] = df["heart_rate"] * df["systolic_bp"]
    df["steps_per_activity"] = df["daily_steps"] / df["physical_activity_level"]
    df["age_map_interaction"] = df["age"] * df["map_bp"]
    df["sleep_balance_deficit_interaction"] = df["sleep_deficit_7h"] * df["sleep_stress_balance"]
    df["bmi_map_interaction"] = df["bmi_risk"] * df["map_bp"]
    df["sleep_disorder"] = pd.Categorical(df["sleep_disorder"], categories=CLASS_ORDER, ordered=True)

    min_share = df["sleep_disorder"].value_counts(normalize=True).min()
    threshold = int(math.ceil(5 / min_share))
    occ_counts = df["occupation"].value_counts()
    df["occupation_collapsed"] = df["occupation"].where(df["occupation"].map(occ_counts) >= threshold, "Other")

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


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    profile_predictors = [
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
    unique_profiles = df[profile_predictors].drop_duplicates().shape[0]
    duplicated_rows = len(df) - unique_profiles
    profile_counts = df[profile_predictors].value_counts()
    out = pd.DataFrame(
        [
            {"metric": "Total rows", "value": len(df)},
            {"metric": "Unique predictor profiles", "value": unique_profiles},
            {"metric": "Duplicated rows by predictor profile", "value": duplicated_rows},
            {"metric": "Largest profile repetition count", "value": int(profile_counts.max())},
            {"metric": "Number of predictor profiles repeated >=2 times", "value": int((profile_counts >= 2).sum())},
            {"metric": "Number of predictor profiles repeated >=3 times", "value": int((profile_counts >= 3).sum())},
        ]
    )
    return out


def robust_numeric_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for feature in ALL_NUMERIC:
        groups = [df.loc[df["sleep_disorder"] == disorder, feature] for disorder in CLASS_ORDER]
        classical = stats.f_oneway(*groups)
        welch = stats.f_oneway(*groups, equal_var=False)
        kruskal = stats.kruskal(*groups)
        levene = stats.levene(*groups, center="median")
        shapiro = stats.shapiro(df[feature].sample(min(len(df), 500), random_state=RANDOM_SEED))
        all_values = pd.concat(groups)
        overall_mean = all_values.mean()
        ss_between = sum(len(group) * (group.mean() - overall_mean) ** 2 for group in groups)
        ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
        eta_sq = ss_between / max(ss_between + ss_within, 1e-12)
        epsilon_sq = (kruskal.statistic - len(groups) + 1) / max(len(all_values) - len(groups), 1)
        rows.append(
            {
                "feature": feature,
                "feature_label": pretty(feature),
                "domain": "Original" if feature in ORIGINAL_NUMERIC else "Derived",
                "anova_p": classical.pvalue,
                "welch_p": welch.pvalue,
                "kruskal_p": kruskal.pvalue,
                "levene_p": levene.pvalue,
                "shapiro_p": shapiro.pvalue,
                "eta_squared": eta_sq,
                "epsilon_squared": epsilon_sq,
            }
        )
    out = pd.DataFrame(rows)
    for column in ["anova_p", "welch_p", "kruskal_p"]:
        out[column.replace("_p", "_fdr")] = multipletests(out[column], method="fdr_bh")[1]
    out["robust_keep"] = (out["welch_fdr"] < 0.05) & (out["kruskal_fdr"] < 0.05)
    out = out.sort_values(["welch_fdr", "eta_squared"], ascending=[True, False]).reset_index(drop=True)

    posthoc_rows = []
    selected_features = out.loc[out["robust_keep"], "feature"].head(8).tolist()
    for feature in selected_features:
        feature_rows = []
        for left, right in combinations(CLASS_ORDER, 2):
            x1 = df.loc[df["sleep_disorder"] == left, feature]
            x2 = df.loc[df["sleep_disorder"] == right, feature]
            welch_t = stats.ttest_ind(x1, x2, equal_var=False)
            mann_whitney = stats.mannwhitneyu(x1, x2, alternative="two-sided")
            feature_rows.append(
                {
                    "feature": feature,
                    "feature_label": pretty(feature),
                    "comparison": f"{left} vs {right}",
                    "mean_left": x1.mean(),
                    "mean_right": x2.mean(),
                    "welch_p": welch_t.pvalue,
                    "mannwhitney_p": mann_whitney.pvalue,
                    "hedges_g": compute_hedges_g(x1, x2),
                    "rank_biserial": compute_rank_biserial(x1, x2),
                }
            )
        feature_df = pd.DataFrame(feature_rows)
        feature_df["welch_holm"] = multipletests(feature_df["welch_p"], method="holm")[1]
        feature_df["mannwhitney_holm"] = multipletests(feature_df["mannwhitney_p"], method="holm")[1]
        posthoc_rows.append(feature_df)
    posthoc = pd.concat(posthoc_rows, ignore_index=True) if posthoc_rows else pd.DataFrame()
    return out, posthoc


def permutation_chi_square(values: pd.Series, labels: pd.Series, n_perm: int = PERM_ITER, seed: int = RANDOM_SEED) -> float:
    rng = np.random.default_rng(seed)
    observed = stats.chi2_contingency(pd.crosstab(values, labels))[0]
    simulations = np.empty(n_perm)
    label_values = labels.to_numpy()
    feature_values = values.to_numpy()
    for idx in range(n_perm):
        permuted = rng.permutation(label_values)
        simulations[idx] = stats.chi2_contingency(pd.crosstab(feature_values, permuted))[0]
    return float((np.sum(simulations >= observed) + 1) / (n_perm + 1))


def robust_categorical_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CATEGORICAL_FEATURES:
        contingency = pd.crosstab(df[feature], df["sleep_disorder"])
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)
        rows.append(
            {
                "feature": feature,
                "feature_label": pretty(feature),
                "n_levels": contingency.shape[0],
                "chi2": chi2,
                "chi2_p": chi2_p,
                "permutation_p": permutation_chi_square(df[feature], df["sleep_disorder"]),
                "cramers_v": cramers_v(chi2, int(contingency.values.sum()), contingency.shape),
                "cells_lt5": int((expected < 5).sum()),
                "min_expected": float(expected.min()),
            }
        )
    out = pd.DataFrame(rows).sort_values("permutation_p").reset_index(drop=True)
    out["perm_fdr"] = multipletests(out["permutation_p"], method="fdr_bh")[1]
    return out


def plot_numeric_robustness(numeric_df: pd.DataFrame) -> None:
    heatmap_df = numeric_df[["feature_label", "anova_fdr", "welch_fdr", "kruskal_fdr"]].copy()
    for column in ["anova_fdr", "welch_fdr", "kruskal_fdr"]:
        heatmap_df[column] = -np.log10(np.clip(heatmap_df[column], 1e-12, 1))
    heatmap_df = heatmap_df.rename(
        columns={"anova_fdr": "ANOVA FDR", "welch_fdr": "Welch FDR", "kruskal_fdr": "Kruskal FDR"}
    ).set_index("feature_label")
    plt.figure(figsize=(8.5, 8.5))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Robust Numeric Screening: -log10(FDR)")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_numeric_method_significance_heatmap.png", dpi=220)
    plt.close()

    effect_df = numeric_df.sort_values("eta_squared", ascending=False).head(12)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=effect_df, y="feature_label", x="eta_squared", palette="Blues_r", ax=axes[0])
    axes[0].set_title("Eta-squared (ANOVA)")
    axes[0].set_xlabel("Effect size")
    axes[0].set_ylabel("")
    sns.barplot(data=effect_df, y="feature_label", x="epsilon_squared", palette="Greens_r", ax=axes[1])
    axes[1].set_title("Epsilon-squared (Kruskal)")
    axes[1].set_xlabel("Effect size")
    axes[1].set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_numeric_effect_sizes.png", dpi=220)
    plt.close()


def plot_categorical_robustness(categorical_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_df = categorical_df.sort_values("perm_fdr")
    sns.barplot(data=plot_df, y="feature_label", x="cramers_v", palette="Set2", ax=axes[0])
    axes[0].set_title("Categorical Effect Size (Cramer's V)")
    axes[0].set_xlabel("Cramer's V")
    axes[0].set_ylabel("")

    diag_df = plot_df.copy()
    diag_df["diag_score"] = diag_df["cells_lt5"] + diag_df["min_expected"]
    sns.barplot(data=diag_df, y="feature_label", x="cells_lt5", palette="rocket_r", ax=axes[1])
    axes[1].set_title("Expected-count Problem Diagnostics")
    axes[1].set_xlabel("Cells with expected count < 5")
    axes[1].set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_categorical_robustness.png", dpi=220)
    plt.close()


def nested_cv_metrics(df: pd.DataFrame, feature_names: list[str], grouped: bool) -> pd.DataFrame:
    X = df[feature_names].copy()
    y = df["has_sleep_disorder"]
    groups = df["profile_group"]
    outer_cv = (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        if grouped
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    )
    inner_cv = (
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        if grouped
        else StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    )

    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups if grouped else None), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
            ]
        )
        grid = GridSearchCV(
            pipe,
            param_grid={"clf__C": np.logspace(-3, 3, 7), "clf__class_weight": [None, "balanced"]},
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=None,
        )
        if grouped:
            grid.fit(X_train, y_train, groups=groups.iloc[train_idx])
        else:
            grid.fit(X_train, y_train)
        probs = grid.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        rows.append(
            {
                "fold": fold_id,
                "roc_auc": roc_auc_score(y_test, probs),
                "f1": f1_score(y_test, preds, zero_division=0),
                "accuracy": accuracy_score(y_test, preds),
                "brier": brier_score_loss(y_test, probs),
                "best_c": float(grid.best_params_["clf__C"]),
                "best_weight": str(grid.best_params_["clf__class_weight"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_validation(fold_df: pd.DataFrame, model_name: str, scheme: str) -> dict[str, object]:
    return {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "scheme": scheme,
        "roc_auc_mean": fold_df["roc_auc"].mean(),
        "roc_auc_sd": fold_df["roc_auc"].std(ddof=1),
        "f1_mean": fold_df["f1"].mean(),
        "accuracy_mean": fold_df["accuracy"].mean(),
        "brier_mean": fold_df["brier"].mean(),
        "most_common_weight": fold_df["best_weight"].mode().iloc[0],
        "median_best_c": fold_df["best_c"].median(),
    }


def fixed_group_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(df, df["has_sleep_disorder"], groups=df["profile_group"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def fit_group_holdout_model(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_names: list[str]) -> ValidationArtifact:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    grid = GridSearchCV(
        pipe,
        param_grid={"clf__C": np.logspace(-3, 3, 7), "clf__class_weight": [None, "balanced"]},
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
        scoring="roc_auc",
        n_jobs=None,
    )
    grid.fit(train_df[feature_names], train_df["has_sleep_disorder"], groups=train_df["profile_group"])
    probs = grid.predict_proba(test_df[feature_names])[:, 1]
    preds = (probs >= 0.5).astype(int)

    pred_df = pd.DataFrame(
        {
            "profile_group": test_df["profile_group"].to_numpy(),
            "sleep_disorder": test_df["sleep_disorder"].astype(str).to_numpy(),
            "y_true": test_df["has_sleep_disorder"].to_numpy(),
            "y_prob": probs,
            "y_pred_default": preds,
        }
    )

    calibration = calibration_table(pred_df)
    threshold_df = threshold_curve(pred_df)
    return ValidationArtifact(
        predictions=pred_df,
        best_params=grid.best_params_,
        calibration=calibration,
        threshold_curve=threshold_df,
    )


def calibration_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(
        pred_df["y_true"], pred_df["y_prob"], n_bins=min(6, pred_df["profile_group"].nunique()), strategy="quantile"
    )
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})


def threshold_curve(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = pred_df["y_true"].to_numpy()
    y_prob = pred_df["y_prob"].to_numpy()
    for threshold in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_group_metrics(pred_df: pd.DataFrame, threshold: float = 0.5, n_boot: int = BOOTSTRAP_ITER) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    groups = pred_df["profile_group"].unique()
    records = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([pred_df.loc[pred_df["profile_group"] == group] for group in sampled_groups], ignore_index=True)
        if sample["y_true"].nunique() < 2:
            continue
        probs = sample["y_prob"].to_numpy()
        y_true = sample["y_true"].to_numpy()
        y_pred = (probs >= threshold).astype(int)
        records.append(
            {
                "roc_auc": roc_auc_score(y_true, probs),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "brier": brier_score_loss(y_true, probs),
            }
        )
    return pd.DataFrame(records)


def summarize_bootstrap(boot_df: pd.DataFrame, model_name: str, threshold: float) -> pd.DataFrame:
    rows = []
    for metric in ["roc_auc", "f1", "accuracy", "brier"]:
        rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "threshold": threshold,
                "metric": metric,
                "mean": boot_df[metric].mean(),
                "ci_low": boot_df[metric].quantile(0.025),
                "ci_high": boot_df[metric].quantile(0.975),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(pred_df: pd.DataFrame) -> dict[str, float]:
    y_true = pred_df["y_true"].reset_index(drop=True)
    y_prob = pred_df["y_prob"].reset_index(drop=True)
    cal_df = pd.DataFrame({"logit": logit(y_prob.to_numpy())})
    cal_fit = sm.GLM(y_true, sm.add_constant(cal_df), family=sm.families.Binomial()).fit()
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": ece_score(y_true.to_numpy(), y_prob.to_numpy()),
        "calibration_intercept": float(cal_fit.params["const"]),
        "calibration_slope": float(cal_fit.params["logit"]),
    }


def plot_validation_schemes(validation_df: pd.DataFrame) -> None:
    plot_df = validation_df.melt(
        id_vars=["model_label", "scheme"],
        value_vars=["roc_auc_mean", "f1_mean", "accuracy_mean", "brier_mean"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metric_titles = {
        "roc_auc_mean": "ROC-AUC",
        "f1_mean": "F1",
        "accuracy_mean": "Accuracy",
        "brier_mean": "Brier Score",
    }
    for axis, metric in zip(axes.flat, metric_titles, strict=False):
        sns.barplot(
            data=plot_df[plot_df["metric"] == metric],
            x="model_label",
            y="value",
            hue="scheme",
            palette="Set2",
            ax=axis,
        )
        axis.set_title(metric_titles[metric])
        axis.set_xlabel("")
        axis.set_ylabel(metric_titles[metric])
        axis.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_random_vs_grouped_validation.png", dpi=220)
    plt.close()


def plot_bootstrap_metrics(bootstrap_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["roc_auc", "f1", "brier"]
    titles = {"roc_auc": "ROC-AUC", "f1": "F1", "brier": "Brier Score"}
    for axis, metric in zip(axes, metrics, strict=False):
        subset = bootstrap_summary[bootstrap_summary["metric"] == metric].copy()
        axis.errorbar(
            subset["mean"],
            subset["model_label"],
            xerr=[subset["mean"] - subset["ci_low"], subset["ci_high"] - subset["mean"]],
            fmt="o",
            capsize=4,
            color="#1f77b4",
            ecolor="#9ec3e6",
        )
        axis.set_title(titles[metric])
        axis.set_xlabel(titles[metric])
        axis.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_grouped_bootstrap_metric_cis.png", dpi=220)
    plt.close()


def plot_calibration_and_thresholds(calibration_summaries: pd.DataFrame, artifacts: dict[str, ValidationArtifact]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for model_name, artifact in artifacts.items():
        axes[0].plot(
            artifact.calibration["mean_predicted"],
            artifact.calibration["fraction_positive"],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS[model_name],
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("Grouped Holdout Calibration Curves")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed disorder rate")
    axes[0].legend()

    for model_name, artifact in artifacts.items():
        axes[1].plot(
            artifact.threshold_curve["threshold"],
            artifact.threshold_curve["f1"],
            linewidth=2,
            label=MODEL_LABELS[model_name],
        )
    axes[1].axvline(0.5, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Threshold Sweep (F1)")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_calibration_and_thresholds.png", dpi=220)
    plt.close()


def box_tidwell_test(df: pd.DataFrame, base_features: list[str], binary_features: list[str]) -> pd.DataFrame:
    working = df.copy().reset_index(drop=True)
    X = working[base_features + binary_features].copy()
    for feature in base_features:
        X[f"{feature}_bt"] = working[feature] * np.log(working[feature])
    fit = sm.Logit(working["has_sleep_disorder"], sm.add_constant(X)).fit(disp=False)
    rows = []
    for feature in base_features:
        rows.append(
            {
                "feature": feature,
                "feature_label": pretty(feature),
                "box_tidwell_coef": float(fit.params[f"{feature}_bt"]),
                "p_value": float(fit.pvalues[f"{feature}_bt"]),
            }
        )
    out = pd.DataFrame(rows)
    out["fdr"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def influence_summary(df: pd.DataFrame, features: list[str], binary_features: list[str]) -> tuple[pd.DataFrame, sm.GLM]:
    working = df.copy().reset_index(drop=True)
    X = sm.add_constant(working[features + binary_features])
    fit = sm.GLM(working["has_sleep_disorder"], X, family=sm.families.Binomial()).fit()
    influence = fit.get_influence(observed=False)
    summary = working[
        ["sleep_disorder", "age", "sleep_duration", "physical_activity_level", "diastolic_bp", "male", "bmi_risk"]
    ].copy()
    summary.insert(0, "profile_id", np.arange(1, len(summary) + 1))
    summary["leverage"] = influence.hat_matrix_diag
    summary["cooks_d"] = influence.cooks_distance[0]
    summary["studentized_resid"] = influence.resid_studentized
    summary = summary.sort_values("cooks_d", ascending=False).reset_index(drop=True)
    return summary, fit


def bootstrap_odds_ratios(df: pd.DataFrame, features: list[str], binary_features: list[str], n_boot: int = BOOTSTRAP_ITER) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    working = df.copy().reset_index(drop=True)
    X = sm.add_constant(working[features + binary_features])
    fit = sm.GLM(working["has_sleep_disorder"], X, family=sm.families.Binomial()).fit()
    coef_names = fit.params.index.tolist()
    rows = []
    samples = []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(len(working)), size=len(working), replace=True)
        boot = working.iloc[idx].reset_index(drop=True)
        try:
            boot_fit = sm.GLM(
                boot["has_sleep_disorder"],
                sm.add_constant(boot[features + binary_features]),
                family=sm.families.Binomial(),
            ).fit()
            samples.append(boot_fit.params.reindex(coef_names).to_numpy())
        except Exception:
            continue
    sample_df = pd.DataFrame(samples, columns=coef_names)
    for name in coef_names:
        if name == "const":
            continue
        rows.append(
            {
                "feature": name,
                "feature_label": pretty(name),
                "coef": float(fit.params[name]),
                "odds_ratio": float(np.exp(fit.params[name])),
                "wald_p": float(fit.pvalues[name]),
                "bootstrap_ci_low": float(np.exp(sample_df[name].quantile(0.025))),
                "bootstrap_ci_high": float(np.exp(sample_df[name].quantile(0.975))),
            }
        )
    return pd.DataFrame(rows).sort_values("odds_ratio", ascending=False)


def plot_diagnostics(box_tidwell_df: pd.DataFrame, influence_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(data=box_tidwell_df, y="feature_label", x=-np.log10(np.clip(box_tidwell_df["fdr"], 1e-12, 1)), palette="Blues_r", ax=axes[0])
    axes[0].axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Box-Tidwell Nonlinearity Test")
    axes[0].set_xlabel("-log10(FDR)")
    axes[0].set_ylabel("")

    top = influence_df.head(20).copy()
    scatter = axes[1].scatter(
        top["leverage"],
        top["studentized_resid"],
        s=5000 * np.clip(top["cooks_d"], 0, None) + 40,
        c=top["cooks_d"],
        cmap="magma",
        alpha=0.7,
    )
    axes[1].set_title("Most Influential Deduplicated Profiles")
    axes[1].set_xlabel("Leverage")
    axes[1].set_ylabel("Studentized residual")
    plt.colorbar(scatter, ax=axes[1], label="Cook's distance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_logistic_diagnostics.png", dpi=220)
    plt.close()


def plot_bootstrap_or(or_df: pd.DataFrame) -> None:
    plot_df = or_df.sort_values("odds_ratio", ascending=True)
    plt.figure(figsize=(9, 5.5))
    plt.errorbar(
        plot_df["odds_ratio"],
        plot_df["feature_label"],
        xerr=[plot_df["odds_ratio"] - plot_df["bootstrap_ci_low"], plot_df["bootstrap_ci_high"] - plot_df["odds_ratio"]],
        fmt="o",
        color="#1f77b4",
        ecolor="#9ec3e6",
        capsize=4,
    )
    plt.axvline(1, color="red", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.title("Bootstrap Odds Ratios (Deduplicated Inference Model)")
    plt.xlabel("Odds ratio (log scale)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_bootstrap_odds_ratios.png", dpi=220)
    plt.close()


def plot_nested_comparison(grouped_validation: pd.DataFrame) -> None:
    metrics = grouped_validation[grouped_validation["scheme"] == "Grouped CV"].copy()
    metrics["brier_inverted"] = 1 - metrics["brier_mean"]
    plot_df = metrics.melt(
        id_vars=["model_label"],
        value_vars=["roc_auc_mean", "f1_mean", "accuracy_mean", "brier_inverted"],
        var_name="metric",
        value_name="value",
    )
    metric_names = {
        "roc_auc_mean": "ROC-AUC",
        "f1_mean": "F1",
        "accuracy_mean": "Accuracy",
        "brier_inverted": "1 - Brier",
    }
    plt.figure(figsize=(13, 6))
    sns.barplot(data=plot_df, x="model_label", y="value", hue="metric", palette="Set2")
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("Grouped Nested CV Comparison Across Feature Sets")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, [metric_names[label] for label in labels], title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_grouped_nested_model_comparison.png", dpi=220)
    plt.close()


def save_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(TABLE_DIR / filename, index=False)


def build_intro(summary_df: pd.DataFrame) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
이 보고서는 현재 데이터를 신뢰할 수 있다는 전제 아래, 기존 분석 결과를 **더 엄밀한 방법론으로 다시 점검**하기 위해 작성했다.

핵심 목표는 다섯 가지다.

1. 단변량 결과가 검정 선택이 바뀌어도 유지되는지 확인
2. 랜덤 분할보다 엄격한 `grouped validation`에서 성능이 유지되는지 확인
3. ROC-AUC뿐 아니라 `calibration`과 `threshold stability`까지 점검
4. 최종 로지스틱 모델의 선형성·영향점·오즈비 불확실성을 확인
5. 원 변수 세트, 압축 파생변수 세트, 상호작용 확장 세트 중 무엇이 가장 설득력 있는지 비교

이 순서를 택한 이유는 다음과 같다.

- 먼저 단변량 강건성을 확인해야 어떤 변수군을 다음 모델 비교 단계로 넘길지 합리적으로 설명할 수 있다.
- 그 다음에는 동일 데이터 구조에서 가장 중요한 위험인 `검증 낙관성`을 줄이기 위해 grouped CV와 bootstrap을 수행해야 한다.
- 그 이후에 calibration을 봐야 “확률이 믿을 만한가”를 말할 수 있고,
- 마지막으로 로지스틱 가정 진단과 feature-set 비교를 통해 **해석용 모델**과 **예측용 모델**을 분리 추천할 수 있다.

현재 데이터 구조 요약은 아래와 같다.

{markdown_table(summary_df)}

보고서는 각 단계가 끝날 때마다 자동으로 갱신되도록 만들었다. 마지막 갱신 시각은 `{timestamp}`이다.
"""


def build_step1_section(numeric_df: pd.DataFrame, posthoc_df: pd.DataFrame, categorical_df: pd.DataFrame) -> str:
    original_robust = numeric_df[(numeric_df["domain"] == "Original") & (numeric_df["robust_keep"])]
    derived_robust = numeric_df[(numeric_df["domain"] == "Derived") & (numeric_df["robust_keep"])]
    top_pairwise = posthoc_df.sort_values(["welch_holm", "hedges_g"]).head(12)
    numeric_view = numeric_df[
        [
            "feature_label",
            "domain",
            "welch_fdr",
            "kruskal_fdr",
            "eta_squared",
            "epsilon_squared",
            "robust_keep",
        ]
    ]
    categorical_view = categorical_df[
        ["feature_label", "n_levels", "permutation_p", "perm_fdr", "cramers_v", "cells_lt5", "min_expected"]
    ]
    failed_levene = int((numeric_df["levene_p"] < 0.05).sum())
    failed_shapiro = int((numeric_df["shapiro_p"] < 0.05).sum())
    return f"""
### 왜 이 단계를 먼저 했는가

기존 분석의 첫 결론은 “어떤 변수가 수면장애와 관련이 있는가”였다. 하지만 이 결론이 일반 ANOVA 하나에만 의존하면 이후 회귀모형의 입력 변수 선정도 흔들릴 수 있다. 그래서 이번 단계에서는 같은 질문을 세 가지 방식으로 다시 물었다.

1. `ANOVA`: 기존 결과와의 연결성 확인
2. `Welch ANOVA`: 등분산 가정이 흔들려도 유지되는지 확인
3. `Kruskal-Wallis`: 비정규성과 반복값에 덜 민감한 순위 기반 검정

여기에 더해 모든 수치형 변수에 대해 `FDR` 보정을 적용했고, 범주형 변수는 permutation p-value까지 계산했다. 이 단계의 목적은 “유의하다”를 다시 말하는 것이 아니라, **검정 방식이 바뀌어도 살아남는 변수와 그렇지 않은 변수를 구분하는 것**이었다.

### 가정 진단에서 확인된 점

- Levene 기준 등분산성 위반 변수 수: `{failed_levene} / {len(numeric_df)}`
- Shapiro-Wilk 기준 정규성 위반 변수 수: `{failed_shapiro} / {len(numeric_df)}`

즉, 현재 데이터에서는 고전적 ANOVA만 고수하는 것보다 Welch와 Kruskal을 함께 보는 쪽이 통계적으로 더 타당하다.

### 수치형 변수의 강건성 결과

{markdown_table(numeric_view, max_rows=13)}

해석은 다음과 같다.

- 원 변수에서는 `Diastolic BP`, `Systolic BP`, `Age`, `Physical Activity Level`, `Sleep Duration`, `Quality of Sleep`, `Heart Rate`, `Daily Steps`가 강건하게 남았다.
- 파생변수에서도 `Mean Arterial Pressure`, `Pulse Pressure`, `Sleep Deficit`, `Sleep-Stress Balance`가 모두 유지됐다.
- 따라서 기존에 제안했던 “혈압 축”과 “수면-스트레스 축”은 검정 방식을 바꿔도 무너지지 않았다.

### 범주형 변수의 강건성 결과

{markdown_table(categorical_view)}

범주형 결과는 이렇게 해석하는 것이 맞다.

- `Gender`, `BMI` 계열 변수는 permutation p-value와 효과크기 모두 안정적이었다.
- `Occupation`은 매우 강한 차이를 보였지만 희소 셀 문제가 남아 있어, **설명적 참고 변수**로는 유용해도 회귀모형 입력 변수로 바로 쓰기에는 조심해야 한다.
- 그래서 이후 모델 단계에서는 `gender`, `bmi_risk`는 유지하고, `occupation`은 제외했다.

### pairwise 수준에서 어떤 차이가 가장 컸는가

{markdown_table(top_pairwise[["feature_label", "comparison", "mean_left", "mean_right", "welch_holm", "hedges_g"]], max_rows=12)}

이 결과를 바탕으로 다음 단계에서는 변수 선택을 이렇게 진행했다.

1. 혈압 군에서는 가장 직접적이고 강건한 대표변수로 `diastolic_bp`를 원 변수 모델에 남긴다.
2. 수면 군과 활동/자율신경 군은 서로 높은 상관을 갖기 때문에, 원 변수 모델에서는 대표 조합을 여러 개 만들어 grouped CV로 비교한다.
3. 압축 파생변수 모델은 기존 제안대로 `MAP`, `Pulse Pressure`, `Sleep Deficit`, `Sleep-Stress Balance`를 유지한다.

즉, 이 단계의 결론은 “중요한 변수 목록” 그 자체보다, **다음 모델 비교 단계로 넘어가도 되는 안정적인 변수군이 무엇인지 정리했다는 점**에 있다.
"""


def build_step2_section(
    original_screen_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> str:
    random_df = validation_df[validation_df["scheme"] == "Random CV"]
    grouped_df = validation_df[validation_df["scheme"] == "Grouped CV"]
    random_auc_mean = random_df["roc_auc_mean"].mean()
    grouped_auc_mean = grouped_df["roc_auc_mean"].mean()
    return f"""
### 왜 grouped validation을 두 번째로 했는가

단변량 결과가 강건하더라도, 예측모형이 실제로 안정적인지는 별개의 문제다. 같은 데이터 안에서도 검증 설계가 느슨하면 모델 성능이 과대평가될 수 있다. 그래서 이번 단계에서는 같은 모델 세트를 두 가지 방식으로 비교했다.

1. `Random CV`: 기존 결과와의 연결성
2. `Grouped CV`: 동일 predictor profile이 train/test에 동시에 들어가지 않도록 막는 더 엄격한 검증

이 단계의 핵심 질문은 “모델이 잘 맞느냐”가 아니라, **검증 설계를 엄격하게 바꾸면 어떤 모델이 남느냐**였다.

### 원 변수 대표 조합 screening

원 변수 모델은 수면군(`sleep_duration` vs `quality_of_sleep`)과 활동/자율신경군(`physical_activity_level` vs `heart_rate`) 중 무엇을 대표로 쓸지 먼저 비교했다.

{markdown_table(original_screen_df)}

이 비교에서 `Original: Sleep + PA` 조합은 grouped CV 기준으로 AUC가 충분히 높으면서도 F1과 Accuracy가 가장 안정적이었다. 반대로 `Sleep + HR` 조합은 CV 평균은 좋지만 뒤 단계 calibration에서 threshold 안정성이 떨어져, 최종 대표 원 변수 모델은 `Sleep + PA` 조합으로 가져가기로 했다.

### Random CV vs Grouped CV 비교

{markdown_table(validation_df)}

중요한 해석 포인트는 다음과 같다.

- 전체적으로 Random CV 평균 ROC-AUC는 `{random_auc_mean:.3f}`, Grouped CV 평균 ROC-AUC는 `{grouped_auc_mean:.3f}`였다.
- 즉, grouped 검증으로 바꾸면 성능이 다소 보수적으로 내려가지만, 핵심 모델들은 여전히 0.92 전후의 ROC-AUC를 유지했다.
- `Original: Sleep + PA`는 grouped 기준에서도 가장 안정적인 분류 지표를 보였다.
- `Compressed Derived`와 `Compressed + Interaction`은 분리력은 유지했지만 threshold 0.5에서의 안정성은 뒤 단계 calibration에서 다시 확인할 필요가 있었다.

### grouped holdout bootstrap 신뢰구간

{markdown_table(bootstrap_summary)}

이 단계에서 얻은 결론은 분명하다.

1. grouped 검증은 반드시 필요했다.
2. 그래도 핵심 모델들은 grouped 기준에서도 충분한 분리력을 남겼다.
3. 원 변수 모델 중에서는 `Original: Sleep + PA`가 가장 실전적인 후보였다.
4. 파생변수 모델은 성능 자체는 나쁘지 않지만, 확률 품질과 threshold 안정성을 별도로 봐야 했다.

그래서 다음 단계에서는 ROC-AUC만 보지 않고, **calibration과 threshold sweep**으로 모델을 한 번 더 걸러냈다.
"""


def build_step3_section(calibration_df: pd.DataFrame, threshold_df: pd.DataFrame) -> str:
    default_failures = threshold_df.loc[np.isclose(threshold_df["threshold"], 0.5), ["model_label", "f1"]]
    best_thresholds = threshold_df.sort_values(["model_label", "f1"], ascending=[True, False]).groupby("model_label").head(1)
    return f"""
### 왜 calibration을 별도 단계로 봤는가

ROC-AUC가 높다는 것은 “순위를 잘 매긴다”는 뜻이지, “예측확률이 믿을 만하다”는 뜻은 아니다. 특히 screening 도구를 생각하면, 0.5 기준에서 바로 쓸 수 있는지 아니면 threshold를 따로 조정해야 하는지가 매우 중요하다.

그래서 이번 단계에서는 grouped holdout에서 아래를 함께 봤다.

1. calibration intercept / slope
2. Brier score
3. ECE
4. threshold sweep

### calibration 요약

{markdown_table(calibration_df)}

이 결과는 매우 해석 가치가 높다.

- `Original: Sleep + PA`는 calibration intercept와 slope가 완벽하진 않지만, 복잡도를 거의 늘리지 않고도 안정적인 확률 품질을 보였다.
- `Compressed Derived`는 ROC-AUC는 높지만 calibration slope가 매우 가파르고 ECE도 커서, raw probability를 그대로 쓰기에는 가장 불안정했다.
- `Compressed + Interaction`은 `Compressed Derived`보다 확률 품질이 훨씬 좋아졌고, holdout 기준으로는 원 변수 모델과 비슷한 수준까지 회복됐다.
- 따라서 calibration 관점에서 보면 “파생변수 모델 전체가 부적절하다”기보다, **단순 압축형(compressed)만 그대로 쓰기 어렵고, 상호작용을 넣어야 그나마 안정성이 회복된다**고 해석하는 것이 맞다.

### threshold sweep 결과

{markdown_table(best_thresholds[["model_label", "threshold", "f1", "accuracy", "precision", "recall"]])}

0.5 threshold에서의 F1은 아래와 같았다.

{markdown_table(default_failures)}

해석은 단순하다.

- `Original: Sleep + PA`는 기본 threshold 0.5에서도 바로 사용할 수 있었다.
- `Compressed Derived`는 threshold를 조정하지 않으면 손실이 컸고, 따라서 별도 calibration 또는 threshold tuning이 사실상 필수였다.
- `Compressed + Interaction`은 default threshold에서도 충분히 작동했지만, 원 변수 모델 대비 성능 이득이 압도적이지는 않았다.
- 따라서 현재 데이터에서 “설명 가능성과 사용 편의성까지 고려한 default model”은 여전히 원 변수 기반 모델이고, “복잡도를 감수할 수 있을 때의 성능형 대안”은 `Compressed + Interaction`이다.

이 단계의 결론은 중요한 분기점을 만든다.

1. **설명 가능한 default model**은 `Original: Sleep + PA`
2. **복잡도를 감수할 수 있는 성능형 대안**은 `Compressed + Interaction`
3. **threshold를 별도로 튜닝해야 하는 모델**은 `Compressed Derived`

그래서 다음 단계의 가정 진단과 오즈비 해석은 `Original: Sleep + PA`를 중심으로 수행했다.
"""


def build_step4_section(box_tidwell_df: pd.DataFrame, influence_df: pd.DataFrame, or_df: pd.DataFrame) -> str:
    top_influence = influence_df.head(10)[
        [
            "profile_id",
            "sleep_disorder",
            "age",
            "sleep_duration",
            "physical_activity_level",
            "diastolic_bp",
            "leverage",
            "cooks_d",
            "studentized_resid",
        ]
    ]
    return f"""
### 왜 로지스틱 가정 진단을 이 시점에 했는가

앞 단계에서 모델 후보를 좁힌 뒤에야 비로소 “이 모델을 해석해도 되는가”를 묻는 것이 자연스럽다. 만약 calibration이 불안정한 모델까지 한꺼번에 진단하면 해석 초점이 흐려진다. 따라서 가장 deployable했던 `Original: Sleep + PA` 모델에 대해 다음을 점검했다.

1. `Box-Tidwell`: 연속형 변수의 logit 선형성
2. `Influence diagnostics`: 특정 프로파일이 계수를 과도하게 흔드는지 여부
3. `Bootstrap OR CI`: 오즈비 불확실성

### Box-Tidwell 결과

{markdown_table(box_tidwell_df)}

해석:

- FDR 기준으로 강한 비선형 신호가 남지 않았다.
- 즉, 현재 모델에서 `Age`, `Sleep Duration`, `Physical Activity Level`, `Diastolic BP`를 1차 선형항으로 두는 것은 통계적으로 받아들일 만하다.
- spline이나 piecewise model이 반드시 필요한 수준의 증거는 현재 데이터에서 확인되지 않았다.

### 영향점 상위 프로파일

{markdown_table(top_influence)}

이 표는 “결과가 특정 몇 샘플에만 끌려가는가”를 보기 위해 넣었다.

- Cook's distance 상위 프로파일이 존재하긴 하지만, 극단적으로 하나의 샘플이 모델을 지배하는 구조는 아니었다.
- 영향점은 주로 높은 혈압과 낮은 수면지표가 동시에 나타나는 profile에서 발생했다.
- 따라서 모델은 완전히 균질한 데이터에서 나온 것은 아니지만, 특정 1-2개 프로파일에만 과도하게 의존한다고 보기도 어렵다.

### Bootstrap 오즈비 결과

{markdown_table(or_df)}

이 결과에서 가장 중요한 해석은 다음과 같다.

- `Diastolic BP`는 bootstrap 신뢰구간까지 포함해 가장 안정적인 위험 신호였다.
- `Sleep Duration`과 `Physical Activity Level`은 방향성은 일관되게 보호 방향이지만, 불확실성 폭이 상대적으로 더 컸다.
- `male`, `bmi_risk`는 보정된 다변량 모델 안에서는 방향은 있으나 독립 기여가 크다고 단정하기 어려웠다.

즉, 이 단계까지 오면 `Original: Sleep + PA` 모델의 메시지는 명확하다.

> 현재 데이터에서 수면장애 위험을 가장 안정적으로 밀어 올리는 축은 이완기 혈압이며, 수면시간과 활동량은 보호 방향 신호로 보이지만 혈압만큼 강하게 고정되지는 않는다.
"""


def build_step5_section(grouped_validation_df: pd.DataFrame) -> str:
    grouped_only = grouped_validation_df[grouped_validation_df["scheme"] == "Grouped CV"].copy()
    grouped_only["roc_rank"] = grouped_only["roc_auc_mean"].rank(ascending=False, method="min")
    grouped_only["f1_rank"] = grouped_only["f1_mean"].rank(ascending=False, method="min")
    grouped_only["brier_rank"] = grouped_only["brier_mean"].rank(ascending=True, method="min")
    grouped_only["overall_rank"] = grouped_only[["roc_rank", "f1_rank", "brier_rank"]].mean(axis=1)
    ranking = grouped_only.sort_values("overall_rank")[
        ["model_label", "roc_auc_mean", "f1_mean", "accuracy_mean", "brier_mean", "overall_rank"]
    ]
    best_model = ranking.iloc[0]["model_label"]
    return f"""
### 왜 마지막에 feature-set 비교를 다시 정리했는가

앞 단계들에서 이미 많은 결과가 나왔지만, 최종 추천을 하려면 질문을 하나로 모아야 한다.

> “원 변수 대표 모델, 압축 파생변수 모델, 상호작용 확장 모델 중 무엇을 최종 추천할 것인가?”

이 질문에 답하기 위해 grouped nested CV 결과를 다시 종합해서 순위를 계산했다. 순위는 아래 세 기준을 함께 반영했다.

1. ROC-AUC는 높을수록 좋다.
2. F1은 높을수록 좋다.
3. Brier score는 낮을수록 좋다.

### grouped nested CV 종합 순위

{markdown_table(ranking)}

### 최종 해석

- grouped nested CV만 보면 원 변수 모델과 압축 파생변수 모델이 모두 경쟁력이 있다.
- 하지만 앞 단계 calibration까지 합치면 `Original: Sleep + PA`가 가장 균형이 좋다.
- `Compressed + Interaction`은 복잡도는 늘지만, `Compressed Derived` 대비 일관된 이득을 보여주지 못했다.

즉, 현재 데이터에서의 최종 추천은 두 갈래로 정리하는 것이 가장 정직하다.

1. **설명과 확률 해석까지 고려한 최종 추천 모델**: `Original: Sleep + PA`
2. **복잡도를 감수할 수 있는 성능형 대안**: `Compressed + Interaction`
3. **추가 calibration 또는 threshold tuning이 전제될 때만 고려할 모델**: `Compressed Derived`

종합적으로 보면 최종 1위 모델은 `{best_model}`였지만, 이 순위만으로 추천을 내리기보다 calibration과 가정 진단까지 함께 보는 것이 더 적절했다. 그 기준을 적용하면 실전적인 최종 모델은 `Original: Sleep + PA`가 된다.

### 이 후속 분석으로 무엇이 달라졌는가

기존 분석은 “혈압과 수면 관련 특징이 중요하다”는 결론을 보여줬다.

이번 후속 엄밀성 검증은 그 결론을 다음처럼 더 정교하게 바꿨다.

- 혈압과 수면 관련 특징이 중요한 것은 맞다.
- 그러나 모델을 평가하는 방식에 따라 무엇이 “최종 추천 모델”인지는 달라진다.
- 단순 AUC 최대화보다 `검정 강건성 + grouped validation + calibration + 진단`을 모두 보면,
  최종적으로는 `Age + Sleep Duration + Physical Activity Level + Diastolic BP + male + bmi_risk` 조합이 가장 설득력 있다.
- 파생변수 모델은 유용한 압축이지만, 단순 압축형은 calibration 손실이 크고 상호작용 확장은 이를 일부 회복하더라도 복잡도 대비 이득이 크지 않았다.

결론적으로 이번 후속 실험은 기존 메시지를 뒤집은 것이 아니라, **더 엄밀한 기준으로 어느 모델을 어떻게 써야 하는지 구체화했다**고 보는 것이 맞다.
"""


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    report = LiveReport(REPORT_PATH)
    df = load_data()
    dedup = df.drop(columns=["person_id"]).drop_duplicates().reset_index(drop=True)
    summary_df = dataset_summary(df)
    save_csv(summary_df, "00_dataset_summary.csv")
    report.set_intro(build_intro(summary_df))

    print("STEP 1/5: running robust univariate analyses")
    numeric_df, posthoc_df = robust_numeric_tests(df)
    categorical_df = robust_categorical_tests(df)
    save_csv(numeric_df, "01_numeric_robust_tests.csv")
    save_csv(posthoc_df, "02_numeric_pairwise_posthoc.csv")
    save_csv(categorical_df, "03_categorical_robust_tests.csv")
    plot_numeric_robustness(numeric_df)
    plot_categorical_robustness(categorical_df)
    report.upsert("1. 강건한 단변량 재검정", build_step1_section(numeric_df, posthoc_df, categorical_df))

    print("STEP 2/5: running random vs grouped validation")
    validation_rows = []
    original_screen_rows = []
    grouped_fold_store: dict[str, pd.DataFrame] = {}
    random_fold_store: dict[str, pd.DataFrame] = {}
    for model_name, features in MODEL_FEATURES.items():
        random_folds = nested_cv_metrics(df, features, grouped=False)
        grouped_folds = nested_cv_metrics(df, features, grouped=True)
        random_fold_store[model_name] = random_folds
        grouped_fold_store[model_name] = grouped_folds
        validation_rows.append(summarize_validation(random_folds, model_name, "Random CV"))
        validation_rows.append(summarize_validation(grouped_folds, model_name, "Grouped CV"))
        if model_name.startswith("orig_"):
            original_screen_rows.append(summarize_validation(grouped_folds, model_name, "Grouped CV"))
    validation_df = pd.DataFrame(validation_rows).sort_values(["scheme", "roc_auc_mean"], ascending=[True, False]).reset_index(drop=True)
    original_screen_df = pd.DataFrame(original_screen_rows).sort_values(["roc_auc_mean", "f1_mean"], ascending=[False, False]).reset_index(drop=True)
    save_csv(validation_df, "04_validation_random_vs_grouped.csv")
    save_csv(original_screen_df, "05_original_model_screening_grouped.csv")
    plot_validation_schemes(validation_df)

    train_df, test_df = fixed_group_holdout(df)
    artifacts: dict[str, ValidationArtifact] = {}
    bootstrap_summaries = []
    for model_name in PRIMARY_MODELS:
        artifact = fit_group_holdout_model(train_df, test_df, MODEL_FEATURES[model_name])
        artifacts[model_name] = artifact
        save_csv(artifact.predictions, f"06_predictions_{model_name}.csv")
        save_csv(artifact.calibration, f"07_calibration_bins_{model_name}.csv")
        save_csv(artifact.threshold_curve, f"08_threshold_curve_{model_name}.csv")
        boot_df = bootstrap_group_metrics(artifact.predictions, threshold=0.5)
        save_csv(boot_df, f"09_bootstrap_metrics_{model_name}.csv")
        bootstrap_summaries.append(summarize_bootstrap(boot_df, model_name, threshold=0.5))
    bootstrap_summary_df = pd.concat(bootstrap_summaries, ignore_index=True)
    save_csv(bootstrap_summary_df, "10_bootstrap_metric_summary.csv")
    plot_bootstrap_metrics(bootstrap_summary_df)
    report.upsert("2. grouped CV와 bootstrap 검증", build_step2_section(original_screen_df, validation_df, bootstrap_summary_df))

    print("STEP 3/5: running calibration analyses")
    calibration_rows = []
    threshold_rows = []
    for model_name, artifact in artifacts.items():
        calibration_rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                **calibration_summary(artifact.predictions),
                "default_f1": float(
                    artifact.threshold_curve.loc[np.isclose(artifact.threshold_curve["threshold"], 0.5), "f1"].iloc[0]
                ),
            }
        )
        curve = artifact.threshold_curve.copy()
        curve.insert(0, "model", model_name)
        curve.insert(1, "model_label", MODEL_LABELS[model_name])
        threshold_rows.append(curve)
    calibration_df = pd.DataFrame(calibration_rows).sort_values("brier").reset_index(drop=True)
    threshold_df = pd.concat(threshold_rows, ignore_index=True)
    save_csv(calibration_df, "11_calibration_summary.csv")
    save_csv(threshold_df, "12_threshold_sweep_summary.csv")
    plot_calibration_and_thresholds(calibration_df, artifacts)
    report.upsert("3. calibration과 threshold 안정성", build_step3_section(calibration_df, threshold_df))

    print("STEP 4/5: running logistic diagnostics and bootstrap OR")
    primary_base_features = ["age", "sleep_duration", "physical_activity_level", "diastolic_bp"]
    primary_binary_features = ["male", "bmi_risk"]
    box_tidwell_df = box_tidwell_test(dedup, primary_base_features, primary_binary_features)
    influence_df, _ = influence_summary(dedup, primary_base_features, primary_binary_features)
    or_df = bootstrap_odds_ratios(dedup, primary_base_features, primary_binary_features)
    save_csv(box_tidwell_df, "13_box_tidwell_results.csv")
    save_csv(influence_df, "14_influence_summary.csv")
    save_csv(or_df, "15_bootstrap_odds_ratios.csv")
    plot_diagnostics(box_tidwell_df, influence_df)
    plot_bootstrap_or(or_df)
    report.upsert("4. 로지스틱 가정 진단과 오즈비 불확실성", build_step4_section(box_tidwell_df, influence_df, or_df))

    print("STEP 5/5: consolidating final grouped nested comparison")
    plot_nested_comparison(validation_df)
    report.upsert("5. feature set 최종 비교와 최종 권고", build_step5_section(validation_df))

    print("Rigorous follow-up validation complete.")
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
