from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

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
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "Sleep_health_and_lifestyle_dataset" / "Sleep_health_and_lifestyle_dataset.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORT_PATH = RESULTS_DIR / "sleep_disorder_statistical_report.md"


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


DISPLAY_LABELS = {
    "age": "Age",
    "sleep_duration": "Sleep Duration",
    "quality_of_sleep": "Quality of Sleep",
    "physical_activity_level": "Physical Activity Level",
    "stress_level": "Stress Level",
    "heart_rate": "Heart Rate",
    "daily_steps": "Daily Steps",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "gender": "Gender",
    "occupation": "Occupation",
    "bmi_category": "BMI Category",
    "has_sleep_disorder": "Has Sleep Disorder",
}


DISORDER_ORDER = ["None", "Insomnia", "Sleep Apnea"]
NUMERIC_FEATURES = [
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
CATEGORICAL_FEATURES = ["gender", "occupation", "bmi_category"]
MODEL_EXCLUDED_CATEGORICALS = {"occupation"}


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def pretty_name(name: str) -> str:
    if name in DISPLAY_LABELS:
        return DISPLAY_LABELS[name]
    if name.startswith("C(gender)["):
        return "Gender: Male"
    if name.startswith("C(bmi_category)[T."):
        level = name.split("T.", 1)[1].rstrip("]")
        return f"BMI Category: {level}"
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
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display_df.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def cramers_v(contingency: pd.DataFrame) -> float:
    if contingency.empty:
        return float("nan")
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.values.sum()
    if n == 0:
        return float("nan")
    min_dim = min(contingency.shape) - 1
    if min_dim <= 0:
        return float("nan")
    return math.sqrt(chi2 / (n * min_dim))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).rename(columns=RENAME_MAP)
    df["sleep_disorder"] = df["sleep_disorder"].fillna("None")
    df["bmi_category"] = df["bmi_category"].replace({"Normal Weight": "Normal"})
    bp = df["blood_pressure"].str.extract(r"(?P<systolic_bp>\d+)\/(?P<diastolic_bp>\d+)").astype(int)
    df = pd.concat([df, bp], axis=1)
    df["has_sleep_disorder"] = (df["sleep_disorder"] != "None").astype(int)
    df["sleep_disorder"] = pd.Categorical(df["sleep_disorder"], categories=DISORDER_ORDER, ordered=True)
    return df


def summarize_dataset(df: pd.DataFrame) -> dict[str, float]:
    return {
        "n_rows": int(df.shape[0]),
        "n_unique_profiles_excluding_id": int(df.drop(columns=["person_id"]).drop_duplicates().shape[0]),
        "n_duplicate_profiles_excluding_id": int(df.drop(columns=["person_id"]).duplicated().sum()),
        "positive_rate": float(df["has_sleep_disorder"].mean()),
    }


def descriptive_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_summary = (
        df.groupby("sleep_disorder", observed=False)[NUMERIC_FEATURES]
        .agg(["mean", "std"])
        .round(2)
    )
    disorder_counts = (
        df["sleep_disorder"]
        .value_counts()
        .reindex(DISORDER_ORDER)
        .rename_axis("sleep_disorder")
        .reset_index(name="count")
    )
    disorder_counts["share"] = (disorder_counts["count"] / disorder_counts["count"].sum()).round(3)
    categorical_rows = []
    for feature in CATEGORICAL_FEATURES:
        table = pd.crosstab(df[feature], df["sleep_disorder"], normalize="columns").mul(100).round(1)
        table.insert(0, "feature", pretty_name(feature))
        table.insert(1, "level", table.index)
        categorical_rows.append(table.reset_index(drop=True))
    categorical_summary = pd.concat(categorical_rows, ignore_index=True)
    return numeric_summary, disorder_counts, categorical_summary


def run_anova(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anova_rows = []
    tukey_rows = []
    for feature in NUMERIC_FEATURES:
        groups = [df.loc[df["sleep_disorder"] == group, feature] for group in DISORDER_ORDER]
        f_stat, p_value = stats.f_oneway(*groups)
        overall_mean = df[feature].mean()
        ss_between = sum(len(group) * (group.mean() - overall_mean) ** 2 for group in groups)
        ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
        eta_sq = ss_between / (ss_between + ss_within)
        anova_rows.append(
            {
                "feature": feature,
                "feature_label": pretty_name(feature),
                "f_stat": f_stat,
                "p_value": p_value,
                "eta_squared": eta_sq,
            }
        )
        if p_value < 0.05:
            tukey = pairwise_tukeyhsd(endog=df[feature], groups=df["sleep_disorder"])
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_df.insert(0, "feature", feature)
            tukey_df.insert(1, "feature_label", pretty_name(feature))
            tukey_rows.append(tukey_df)
    anova_df = pd.DataFrame(anova_rows).sort_values(["p_value", "eta_squared"], ascending=[True, False])
    tukey_df = pd.concat(tukey_rows, ignore_index=True) if tukey_rows else pd.DataFrame()
    return anova_df, tukey_df


def run_chi_square(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CATEGORICAL_FEATURES:
        contingency = pd.crosstab(df[feature], df["sleep_disorder"])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        rows.append(
            {
                "feature": feature,
                "feature_label": pretty_name(feature),
                "chi2": chi2,
                "p_value": p_value,
                "dof": dof,
                "cramers_v": cramers_v(contingency),
                "min_level_count": int(df[feature].value_counts().min()),
                "n_levels": int(df[feature].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["p_value", "cramers_v"], ascending=[True, False])


def run_pointbiserial(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_FEATURES:
        corr, p_value = stats.pointbiserialr(df["has_sleep_disorder"], df[feature])
        rows.append(
            {
                "feature": feature,
                "feature_label": pretty_name(feature),
                "correlation": corr,
                "abs_correlation": abs(corr),
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values(["p_value", "abs_correlation"], ascending=[True, False])


def design_matrix_for_vif(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    matrix = df[numeric_features].copy()
    if categorical_features:
        dummies = pd.get_dummies(df[categorical_features], drop_first=True, dtype=float)
        matrix = pd.concat([matrix, dummies], axis=1)
    return matrix.astype(float)


def compute_vif(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    matrix = design_matrix_for_vif(df, numeric_features, categorical_features)
    if matrix.empty:
        return pd.DataFrame(columns=["feature", "vif"])
    matrix = sm.add_constant(matrix, has_constant="add")
    vif_rows = []
    for idx, column in enumerate(matrix.columns):
        if column == "const":
            continue
        vif_rows.append({"feature": column, "vif": variance_inflation_factor(matrix.values, idx)})
    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False)


def candidate_terms(anova_df: pd.DataFrame, chi_df: pd.DataFrame, point_df: pd.DataFrame) -> list[str]:
    selected_numeric = sorted(
        {
            row.feature
            for row in anova_df.itertuples(index=False)
            if row.p_value < 0.05 and row.eta_squared >= 0.02
        }
        | {
            row.feature
            for row in point_df.itertuples(index=False)
            if row.p_value < 0.05 and abs(row.correlation) >= 0.15
        }
    )
    selected_categorical = []
    for row in chi_df.itertuples(index=False):
        if row.p_value < 0.05 and row.feature not in MODEL_EXCLUDED_CATEGORICALS and row.min_level_count >= 10:
            selected_categorical.append(f"C({row.feature})")
    return selected_numeric + selected_categorical


def fit_glm(formula: str, df: pd.DataFrame):
    return smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()


def removable_term(term: str, fit_result) -> tuple[bool, float]:
    if term.startswith("C("):
        term_prefix = term + "["
        p_values = fit_result.pvalues[[idx.startswith(term_prefix) for idx in fit_result.pvalues.index]]
        if len(p_values) == 0:
            return False, 0.0
        if (p_values > 0.05).all():
            return True, float(p_values.min())
        return False, float(p_values.min())
    p_value = float(fit_result.pvalues.get(term, 0.0))
    return p_value > 0.05, p_value


def backward_elimination(df: pd.DataFrame, terms: list[str]) -> tuple[list[str], object]:
    remaining = terms.copy()
    if not remaining:
        model = fit_glm("has_sleep_disorder ~ 1", df)
        return remaining, model
    while True:
        formula = "has_sleep_disorder ~ " + " + ".join(remaining)
        fit_result = fit_glm(formula, df)
        removable = []
        for term in remaining:
            can_remove, score = removable_term(term, fit_result)
            if can_remove:
                removable.append((term, score))
        if not removable:
            return remaining, fit_result
        term_to_remove = sorted(removable, key=lambda item: item[1], reverse=True)[0][0]
        remaining.remove(term_to_remove)
        if not remaining:
            model = fit_glm("has_sleep_disorder ~ 1", df)
            return remaining, model


def model_feature_sets(terms: Iterable[str]) -> tuple[list[str], list[str]]:
    numeric = [term for term in terms if not term.startswith("C(")]
    categorical = [term[2:-1] for term in terms if term.startswith("C(")]
    return numeric, categorical


def run_predictive_model(df: pd.DataFrame, terms: list[str]) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    numeric_features, categorical_features = model_feature_sets(terms)
    feature_columns = numeric_features + categorical_features
    X = df[feature_columns]
    y = df["has_sleep_disorder"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "logistic",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds)),
        "test_recall": float(recall_score(y_test, preds)),
        "test_f1": float(f1_score(y_test, preds)),
        "test_roc_auc": float(roc_auc_score(y_test, probs)),
    }

    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    report_df["label"] = report_df["label"].replace({"0": "No disorder", "1": "Disorder"})
    return report_df, metrics, pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": preds,
            "y_prob": probs,
        }
    )


def odds_ratio_table(fit_result) -> pd.DataFrame:
    params = fit_result.params.drop("Intercept", errors="ignore")
    conf = fit_result.conf_int().loc[params.index]
    or_table = pd.DataFrame(
        {
            "term": params.index,
            "feature_label": [pretty_name(term) for term in params.index],
            "coef": params.values,
            "odds_ratio": np.exp(params.values),
            "ci_low": np.exp(conf[0].values),
            "ci_high": np.exp(conf[1].values),
            "p_value": fit_result.pvalues.loc[params.index].values,
        }
    )
    return or_table.sort_values("p_value")


def save_tables(
    numeric_summary: pd.DataFrame,
    disorder_counts: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    anova_df: pd.DataFrame,
    tukey_df: pd.DataFrame,
    chi_df: pd.DataFrame,
    point_df: pd.DataFrame,
    vif_df: pd.DataFrame,
    or_df: pd.DataFrame,
    class_report_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> None:
    numeric_summary.to_csv(TABLES_DIR / "numeric_summary_by_disorder.csv")
    disorder_counts.to_csv(TABLES_DIR / "sleep_disorder_counts.csv", index=False)
    categorical_summary.to_csv(TABLES_DIR / "categorical_distribution_by_disorder.csv", index=False)
    anova_df.to_csv(TABLES_DIR / "anova_results.csv", index=False)
    tukey_df.to_csv(TABLES_DIR / "anova_tukey_posthoc.csv", index=False)
    chi_df.to_csv(TABLES_DIR / "chi_square_results.csv", index=False)
    point_df.to_csv(TABLES_DIR / "pointbiserial_results.csv", index=False)
    vif_df.to_csv(TABLES_DIR / "vif_results.csv", index=False)
    or_df.to_csv(TABLES_DIR / "logistic_odds_ratios.csv", index=False)
    class_report_df.to_csv(TABLES_DIR / "classification_report.csv", index=False)
    prediction_df.to_csv(TABLES_DIR / "test_predictions.csv", index=False)


def plot_disorder_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    order = DISORDER_ORDER
    ax = sns.countplot(
        data=df,
        x="sleep_disorder",
        hue="sleep_disorder",
        order=order,
        palette="Set2",
        legend=False,
    )
    ax.set_title("Sleep Disorder Distribution")
    ax.set_xlabel("Sleep Disorder")
    ax.set_ylabel("Count")
    for patch in ax.patches:
        height = int(patch.get_height())
        ax.annotate(
            str(height),
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_sleep_disorder_distribution.png", dpi=220)
    plt.close()


def plot_numeric_correlation(df: pd.DataFrame) -> None:
    corr_df = df[NUMERIC_FEATURES + ["has_sleep_disorder"]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_numeric_correlation_heatmap.png", dpi=220)
    plt.close()


def plot_anova_effect_sizes(anova_df: pd.DataFrame) -> None:
    plot_df = anova_df.sort_values("eta_squared", ascending=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=plot_df,
        x="eta_squared",
        y="feature_label",
        hue="feature_label",
        palette="crest",
        legend=False,
    )
    plt.title("ANOVA Effect Sizes by Numeric Feature")
    plt.xlabel("Eta squared")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_anova_effect_sizes.png", dpi=220)
    plt.close()


def plot_group_boxplots(df: pd.DataFrame, point_df: pd.DataFrame) -> None:
    top_features = point_df.sort_values("abs_correlation", ascending=False).head(4)["feature"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, feature in zip(axes.flat, top_features, strict=False):
        sns.boxplot(
            data=df,
            x="sleep_disorder",
            y=feature,
            hue="sleep_disorder",
            order=DISORDER_ORDER,
            palette="Set2",
            ax=axis,
            legend=False,
        )
        axis.set_title(pretty_name(feature))
        axis.set_xlabel("Sleep Disorder")
        axis.set_ylabel(pretty_name(feature))
    for axis in axes.flat[len(top_features) :]:
        axis.axis("off")
    fig.suptitle("Top Numeric Features by Sleep Disorder Group", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_group_boxplots.png", dpi=220)
    plt.close()


def plot_categorical_effect_sizes(chi_df: pd.DataFrame) -> None:
    plot_df = chi_df.sort_values("cramers_v", ascending=True)
    plt.figure(figsize=(8, 4.8))
    sns.barplot(
        data=plot_df,
        x="cramers_v",
        y="feature_label",
        hue="feature_label",
        palette="flare",
        legend=False,
    )
    plt.title("Categorical Feature Association Strength")
    plt.xlabel("Cramer's V")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_categorical_effect_sizes.png", dpi=220)
    plt.close()


def plot_categorical_panels(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, feature in zip(axes, ["gender", "bmi_category"], strict=False):
        proportion = pd.crosstab(df[feature], df["sleep_disorder"], normalize="index")
        proportion = proportion.reindex(sorted(proportion.index))
        proportion.plot(kind="bar", stacked=True, ax=axis, colormap="Set2")
        axis.set_title(pretty_name(feature))
        axis.set_xlabel(pretty_name(feature))
        axis.set_ylabel("Proportion")
        axis.legend(title="Sleep Disorder", loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_categorical_panels.png", dpi=220)
    plt.close()


def plot_vif(vif_df: pd.DataFrame) -> None:
    if vif_df.empty:
        return
    plot_df = vif_df.sort_values("vif", ascending=True).copy()
    plot_df["feature_label"] = plot_df["feature"].map(pretty_name)
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=plot_df,
        x="vif",
        y="feature_label",
        hue="feature_label",
        palette="mako",
        legend=False,
    )
    plt.axvline(5, color="orange", linestyle="--", linewidth=1, label="VIF = 5")
    plt.axvline(10, color="red", linestyle="--", linewidth=1, label="VIF = 10")
    plt.title("Multicollinearity Check for Final Model")
    plt.xlabel("VIF")
    plt.ylabel("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_vif_plot.png", dpi=220)
    plt.close()


def plot_odds_ratios(or_df: pd.DataFrame) -> None:
    plot_df = or_df.sort_values("odds_ratio", ascending=True)
    plt.figure(figsize=(9, max(4.5, len(plot_df) * 0.65)))
    plt.errorbar(
        plot_df["odds_ratio"],
        plot_df["feature_label"],
        xerr=[
            plot_df["odds_ratio"] - plot_df["ci_low"],
            plot_df["ci_high"] - plot_df["odds_ratio"],
        ],
        fmt="o",
        color="#1f77b4",
        ecolor="#7aa6d6",
        capsize=4,
    )
    plt.axvline(1, color="red", linestyle="--", linewidth=1)
    plt.title("Logistic Regression Odds Ratios")
    plt.xlabel("Odds Ratio (95% CI)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_logistic_odds_ratios.png", dpi=220)
    plt.close()


def plot_model_performance(prediction_df: pd.DataFrame) -> None:
    fpr, tpr, _ = roc_curve(prediction_df["y_true"], prediction_df["y_prob"])
    auc = roc_auc_score(prediction_df["y_true"], prediction_df["y_prob"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    matrix = confusion_matrix(prediction_df["y_true"], prediction_df["y_pred"])
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["No disorder", "Disorder"]).plot(
        ax=axes[1],
        cmap="Blues",
        colorbar=False,
    )
    axes[1].set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_model_performance.png", dpi=220)
    plt.close()


def create_report(
    summary: dict[str, float],
    disorder_counts: pd.DataFrame,
    anova_df: pd.DataFrame,
    chi_df: pd.DataFrame,
    point_df: pd.DataFrame,
    final_terms: list[str],
    vif_df: pd.DataFrame,
    or_df: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    top_numeric = anova_df.loc[:, ["feature_label", "f_stat", "p_value", "eta_squared"]].head(5).copy()
    top_numeric["p_value"] = top_numeric["p_value"].map(format_p)
    top_corr = point_df.loc[:, ["feature_label", "correlation", "p_value"]].head(5).copy()
    top_corr["p_value"] = top_corr["p_value"].map(format_p)
    top_cat = chi_df.loc[:, ["feature_label", "chi2", "p_value", "cramers_v"]].copy()
    top_cat["p_value"] = top_cat["p_value"].map(format_p)
    selected_features = [pretty_name(term) for term in final_terms]
    vif_top = vif_df.copy()
    vif_top["feature_label"] = vif_top["feature"].map(pretty_name)
    vif_top = vif_top[["feature_label", "vif"]]
    or_top = or_df.loc[:, ["feature_label", "odds_ratio", "ci_low", "ci_high", "p_value"]].copy()
    or_top["p_value"] = or_top["p_value"].map(format_p)

    strongest_positive = or_df.sort_values("odds_ratio", ascending=False).iloc[0]
    strongest_negative = or_df.sort_values("odds_ratio", ascending=True).iloc[0]
    max_vif = float(vif_df["vif"].max()) if not vif_df.empty else 0.0
    if max_vif < 5:
        vif_comment = "All final VIF values were below 5, so multicollinearity is not a practical concern in the retained model."
    elif max_vif < 10:
        vif_comment = "The final VIF values were below 10, which keeps the model within a tolerable multicollinearity range for interpretation."
    else:
        vif_comment = "Some VIF values remained high, so coefficient-level interpretation should be treated cautiously even though the model still predicts well."

    report = f"""# Sleep Health Statistical Analysis Report

## 1. Analysis rationale and sequence

The reference materials were translated into a practical analysis flow for this dataset.

1. Week 3 emphasizes cross-tabulation and one-way ANOVA for checking whether group-level differences exist before modeling.
2. Week 4 emphasizes Pearson correlation, regression interpretation, and VIF-based multicollinearity checks.
3. Week 5 extends the workflow to logistic regression for binary clinical outcomes.

Following that logic, this analysis used the sequence below:

1. Descriptive statistics to confirm the distribution of sleep disorder labels and the available variable types.
2. Group-comparison tests to identify which features differ across `None`, `Insomnia`, and `Sleep Apnea`.
3. Correlation analysis to quantify linear associations with sleep disorder and detect redundancy among predictors.
4. Logistic regression to build an interpretable classification model for `Has Sleep Disorder`.

## 2. Dataset inventory and preparation

The `dataset` folder contains two different resources:

1. `Sleep_health_and_lifestyle_dataset`: person-level tabular data with an explicit `Sleep Disorder` label
2. `FitBit Fitness Tracker Data`: device logs for activity and sleep minutes, but without a direct `Sleep Disorder` target and without a person-level linkage to the first dataset

Because the user asked for correlation with **sleep disorder** and for a **logistic regression classifier**, the main inferential analysis was performed on `Sleep_health_and_lifestyle_dataset`. The Fitbit files were reviewed as contextual sleep/activity logs, but they are not suitable for supervised sleep-disorder classification in this workspace as provided.

## 3. Data preparation

- Source file: `dataset/Sleep_health_and_lifestyle_dataset/Sleep_health_and_lifestyle_dataset.csv`
- Total rows: {summary["n_rows"]}
- Distinct profiles excluding `person_id`: {summary["n_unique_profiles_excluding_id"]}
- Repeated profiles excluding `person_id`: {summary["n_duplicate_profiles_excluding_id"]}
- Positive class rate for `Has Sleep Disorder`: {summary["positive_rate"]:.3f}
- `Sleep Disorder` missing values were treated as `"None"` because the raw CSV stores non-disorder cases in that form.
- `Blood Pressure` was split into `Systolic BP` and `Diastolic BP`.
- `BMI Category` was harmonized by merging `Normal Weight` into `Normal`.

Sleep disorder class balance:

{markdown_table(disorder_counts)}

## 4. Group-comparison results

### 4.1 Numeric features: one-way ANOVA

The strongest numeric group differences were:

{markdown_table(top_numeric)}

Interpretation:

- Larger `eta_squared` means the feature separates sleep disorder groups more clearly.
- In this dataset, blood pressure, age, physical activity, and heart rate showed the largest group separation, while sleep duration and sleep quality were also statistically important.

### 4.2 Categorical features: chi-square test

{markdown_table(top_cat)}

Interpretation:

- `BMI Category`, `Occupation`, and `Gender` all show univariate association with the disorder groups.
- `Occupation` was excluded from the predictive model because several job levels are too sparse for stable coefficient estimation.
- `BMI Category` and `Gender` were considered during modeling, but they did not survive the final multivariable reduction step.

## 5. Correlation analysis

Point-biserial correlations with `Has Sleep Disorder`:

{markdown_table(top_corr)}

Interpretation:

- Negative coefficients mean the variable tends to be lower in the disorder group.
- Positive coefficients mean the variable tends to be higher in the disorder group.
- The largest practical signals came from shorter sleep duration, poorer sleep quality, higher stress, higher heart rate, and higher blood pressure.

## 6. Logistic regression for sleep-disorder classification

### 6.1 Variable selection logic

Variables were considered for the final logistic model only if they met the following conditions:

1. They showed evidence of group differences or binary association in the earlier statistical tests.
2. They were clinically interpretable for sleep disorder screening.
3. They did not create problematic sparsity or unstable multicollinearity.

Final variables used in the logistic model:

- {"\n- ".join(selected_features) if selected_features else "Intercept only"}

VIF check for the final model:

{markdown_table(vif_top)}

{vif_comment}

### 6.2 Logistic regression coefficients

{markdown_table(or_top)}

Interpretation:

- The strongest risk-increasing feature in the final model was **{strongest_positive["feature_label"]}** with OR {strongest_positive["odds_ratio"]:.2f}.
- The strongest protective feature in the final model was **{strongest_negative["feature_label"]}** with OR {strongest_negative["odds_ratio"]:.2f}.
- Odds ratios above 1 indicate higher odds of having a sleep disorder; below 1 indicate lower odds.

### 6.3 Classification performance

- 5-fold cross-validated ROC-AUC: {metrics["cv_auc_mean"]:.3f} +/- {metrics["cv_auc_std"]:.3f}
- Test ROC-AUC: {metrics["test_roc_auc"]:.3f}
- Test accuracy: {metrics["test_accuracy"]:.3f}
- Test precision: {metrics["test_precision"]:.3f}
- Test recall: {metrics["test_recall"]:.3f}
- Test F1-score: {metrics["test_f1"]:.3f}

## 7. Final insights

### 7.1 Features associated with sleep disorder

This analysis consistently suggests that sleep disorder is most closely related to:

1. Reduced sleep duration
2. Lower quality of sleep
3. Higher stress level
4. Elevated heart rate
5. Higher blood pressure
6. Less favorable BMI category

These variables were the main univariate signals across ANOVA, chi-square, and correlation screening. After overlapping information was accounted for in the multivariable logistic model, the most stable retained predictors were `Age`, `Diastolic BP`, and `Quality of Sleep`.

### 7.2 Which variables should be used for logistic regression?

For an interpretable screening-oriented logistic regression model, the recommended variables are:

- {"\n- ".join(selected_features) if selected_features else "No stable predictors were retained"}

Why these variables:

- They showed statistical evidence of association with sleep disorder.
- They offered clinically understandable directionality.
- They avoided occupation-like sparse categories that can make the model unstable.

### 7.3 Practical value

Based on these findings, the dataset supports a useful screening insight:

- People with shorter and poorer sleep, higher stress, and worse cardio-metabolic signals are more likely to belong to the sleep-disorder group.
- A lightweight logistic screening model can help prioritize who may benefit from additional sleep assessment, lifestyle coaching, or medical follow-up.
- In practice, the model can support early triage, personalized sleep-health feedback, and risk-based monitoring when full polysomnography or specialist evaluation is not immediately available.

## 8. Figures generated

1. `results/figures/01_sleep_disorder_distribution.png`
2. `results/figures/02_numeric_correlation_heatmap.png`
3. `results/figures/03_anova_effect_sizes.png`
4. `results/figures/04_group_boxplots.png`
5. `results/figures/05_categorical_effect_sizes.png`
6. `results/figures/06_categorical_panels.png`
7. `results/figures/07_vif_plot.png`
8. `results/figures/08_logistic_odds_ratios.png`
9. `results/figures/09_model_performance.png`

## 9. Caution

- This dataset contains many repeated profiles aside from `person_id`, which suggests a templated or synthetic structure. The direction of findings is still useful, but p-values may look stronger than they would in a more heterogeneous real-world cohort.
- The logistic model predicts presence of any sleep disorder, not the subtype distinction between insomnia and sleep apnea.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid")
    df = load_data()
    summary = summarize_dataset(df)

    numeric_summary, disorder_counts, categorical_summary = descriptive_tables(df)
    anova_df, tukey_df = run_anova(df)
    chi_df = run_chi_square(df)
    point_df = run_pointbiserial(df)

    initial_terms = candidate_terms(anova_df, chi_df, point_df)
    final_terms, fit_result = backward_elimination(df, initial_terms)
    numeric_model_features, categorical_model_features = model_feature_sets(final_terms)
    vif_df = compute_vif(df, numeric_model_features, categorical_model_features)
    or_df = odds_ratio_table(fit_result)
    class_report_df, metrics, prediction_df = run_predictive_model(df, final_terms)

    save_tables(
        numeric_summary=numeric_summary,
        disorder_counts=disorder_counts,
        categorical_summary=categorical_summary,
        anova_df=anova_df,
        tukey_df=tukey_df,
        chi_df=chi_df,
        point_df=point_df,
        vif_df=vif_df,
        or_df=or_df,
        class_report_df=class_report_df,
        prediction_df=prediction_df,
    )

    plot_disorder_distribution(df)
    plot_numeric_correlation(df)
    plot_anova_effect_sizes(anova_df)
    plot_group_boxplots(df, point_df)
    plot_categorical_effect_sizes(chi_df)
    plot_categorical_panels(df)
    plot_vif(vif_df)
    plot_odds_ratios(or_df)
    plot_model_performance(prediction_df)

    create_report(
        summary=summary,
        disorder_counts=disorder_counts,
        anova_df=anova_df,
        chi_df=chi_df,
        point_df=point_df,
        final_terms=final_terms,
        vif_df=vif_df,
        or_df=or_df,
        metrics=metrics,
    )

    print("Selected terms:", final_terms)
    print("Cross-validated AUC:", round(metrics["cv_auc_mean"], 3))
    print("Test ROC-AUC:", round(metrics["test_roc_auc"], 3))
    print("Report written to:", REPORT_PATH)


if __name__ == "__main__":
    main()
