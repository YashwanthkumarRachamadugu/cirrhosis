import io
import pickle
import warnings
from dataclasses import dataclass
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


warnings.filterwarnings("ignore")

DEFAULT_DATASET_PATH = "cirrhosis.csv"
RANDOM_STATE = 46

st.set_page_config(
    page_title="Cirrhosis ML Studio",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)
sns.set_theme(style="whitegrid")


CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1280px;
    }
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e5e9f0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    .section-note {
        color: #526071;
        font-size: 0.95rem;
        margin-top: -0.4rem;
        margin-bottom: 0.6rem;
    }
    .best-model {
        border-left: 4px solid #247a6b;
        background: #eef8f5;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0 1rem 0;
    }
</style>
"""


@dataclass
class TrainedResult:
    name: str
    estimator: Any
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_f1_mean: float | None
    cv_f1_std: float | None
    confusion: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    y_test: np.ndarray
    report: pd.DataFrame


def inject_style() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file_bytes: bytes | None, use_default: bool) -> pd.DataFrame:
    if uploaded_file_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes))
    if use_default:
        return pd.read_csv(DEFAULT_DATASET_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_pickle_model(uploaded_model_bytes: bytes | None):
    if uploaded_model_bytes is None:
        return None
    try:
        return joblib.load(io.BytesIO(uploaded_model_bytes))
    except Exception:
        return pickle.loads(uploaded_model_bytes)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=False),
        }
    )


def profile_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.assign(
        skew=numeric.skew(numeric_only=True),
        kurtosis=numeric.kurtosis(numeric_only=True),
    ).round(3)


def prepare_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    working = clean_columns(df)
    if target_column not in working.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    working = working.drop(columns=[col for col in working.columns if col.lower() == "id"], errors="ignore")
    working = working.dropna(subset=[target_column])
    y_raw = working[target_column].astype(str).str.strip()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    X = working.drop(columns=[target_column], errors="ignore")
    return X, y, encoder


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return numeric_features, categorical_features


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, scaler_name: str) -> ColumnTransformer:
    numeric_features, categorical_features = split_feature_types(X)
    scaler = RobustScaler() if scaler_name == "RobustScaler" else StandardScaler()
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def with_preprocessor(name: str, estimator: Any, X: pd.DataFrame, scaler_name: str) -> tuple[str, Pipeline]:
    return (
        name,
        Pipeline(
            steps=[
                ("transformer", build_preprocessor(X, scaler_name)),
                ("model", estimator),
            ]
        ),
    )


def sidebar_hyperparameters() -> dict[str, Any]:
    st.sidebar.header("Training Controls")
    test_size = st.sidebar.slider("Test size", 0.15, 0.40, 0.25, 0.05)
    cv_folds = st.sidebar.slider("Cross-validation folds", 3, 10, 5, 1)
    scaler_name = st.sidebar.selectbox("Numeric transformer", ["RobustScaler", "StandardScaler"])
    max_models = st.sidebar.slider("Maximum models to evaluate", 4, 18, 12, 1)

    with st.sidebar.expander("Model Hyperparameters", expanded=False):
        rf_estimators = st.slider("RF trees", 50, 500, 180, 10)
        rf_depth = st.slider("RF max depth", 2, 30, 10, 1)
        gb_estimators = st.slider("GB trees", 50, 400, 160, 10)
        learning_rate = st.slider("Boosting learning rate", 0.01, 0.30, 0.05, 0.01)
        svm_c = st.slider("SVM C", 0.1, 10.0, 2.0, 0.1)
        knn_neighbors = st.slider("KNN neighbors", 2, 25, 5, 1)

    return {
        "test_size": test_size,
        "cv_folds": cv_folds,
        "scaler_name": scaler_name,
        "max_models": max_models,
        "rf_estimators": rf_estimators,
        "rf_depth": rf_depth,
        "gb_estimators": gb_estimators,
        "learning_rate": learning_rate,
        "svm_c": svm_c,
        "knn_neighbors": knn_neighbors,
    }


def build_model_catalog(X: pd.DataFrame, controls: dict[str, Any], n_classes: int) -> list[tuple[str, Pipeline]]:
    scaler_name = controls["scaler_name"]
    models: list[tuple[str, Pipeline]] = [
        with_preprocessor("Decision Tree (Baseline)", DecisionTreeClassifier(random_state=RANDOM_STATE), X, scaler_name),
        with_preprocessor(
            "Decision Tree (Tuned)",
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                criterion="entropy",
                max_depth=8,
                min_samples_split=8,
                min_samples_leaf=3,
                class_weight="balanced",
            ),
            X,
            scaler_name,
        ),
        with_preprocessor(
            "Random Forest (Baseline)",
            RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, n_jobs=1),
            X,
            scaler_name,
        ),
        with_preprocessor(
            "Random Forest (Tuned)",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=controls["rf_estimators"],
                max_depth=controls["rf_depth"],
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
            ),
            X,
            scaler_name,
        ),
        with_preprocessor(
            "Logistic Regression (Tuned)",
            LogisticRegression(
                max_iter=2500,
                random_state=RANDOM_STATE,
                C=0.8,
                solver="lbfgs",
                class_weight="balanced",
            ),
            X,
            scaler_name,
        ),
        with_preprocessor(
            "SVM (Tuned)",
            SVC(
                random_state=RANDOM_STATE,
                C=controls["svm_c"],
                gamma="scale",
                kernel="rbf",
                class_weight="balanced",
                probability=True,
            ),
            X,
            scaler_name,
        ),
        with_preprocessor(
            "KNN (Tuned)",
            KNeighborsClassifier(n_neighbors=controls["knn_neighbors"], weights="distance", metric="manhattan"),
            X,
            scaler_name,
        ),
        with_preprocessor("Naive Bayes", GaussianNB(var_smoothing=1e-9), X, scaler_name),
        with_preprocessor(
            "AdaBoost (Tuned)",
            AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=120, learning_rate=controls["learning_rate"]),
            X,
            scaler_name,
        ),
        with_preprocessor(
            "Gradient Boosting (Tuned)",
            GradientBoostingClassifier(
                random_state=RANDOM_STATE,
                n_estimators=controls["gb_estimators"],
                learning_rate=controls["learning_rate"],
                max_depth=3,
                subsample=0.85,
            ),
            X,
            scaler_name,
        ),
        with_preprocessor(
            "Hist Gradient Boosting",
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                learning_rate=controls["learning_rate"],
                max_iter=100,
                max_depth=5,
            ),
            X,
            scaler_name,
        ),
    ]

    if XGBClassifier is not None:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        models.append(
            with_preprocessor(
                "XGBoost (Baseline)",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.1,
                    objective=objective,
                    eval_metric="logloss" if n_classes == 2 else "mlogloss",
                ),
                X,
                scaler_name,
            )
        )
        models.append(
            with_preprocessor(
                "XGBoost (Tuned)",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=220,
                    max_depth=4,
                    learning_rate=controls["learning_rate"],
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.2,
                    reg_alpha=0.05,
                    objective=objective,
                    eval_metric="logloss" if n_classes == 2 else "mlogloss",
                ),
                X,
                scaler_name,
            )
        )

    if LGBMClassifier is not None:
        models.append(
            with_preprocessor(
                "LightGBM (Tuned)",
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=220,
                    learning_rate=controls["learning_rate"],
                    num_leaves=15,
                    min_child_samples=12,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    verbose=-1,
                ),
                X,
                scaler_name,
            )
        )

    if CatBoostClassifier is not None:
        models.append(
            with_preprocessor(
                "CatBoost (Tuned)",
                CatBoostClassifier(
                    random_state=RANDOM_STATE,
                    iterations=260,
                    depth=4,
                    learning_rate=controls["learning_rate"],
                    l2_leaf_reg=4,
                    verbose=False,
                ),
                X,
                scaler_name,
            )
        )

    models.append(
        with_preprocessor(
            "Stacking Classifier",
            StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=120, max_depth=8, n_jobs=1)),
                    ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")),
                    ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=80)),
                ],
                final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                cv=3,
            ),
            X,
            scaler_name,
        )
    )
    return models[: controls["max_models"]]


def safe_stratify(y: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(y, return_counts=True)
    return y if len(counts) > 1 and counts.min() >= 2 else None


def evaluate_estimator(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    cv_folds: int,
) -> TrainedResult:
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    try:
        probabilities = estimator.predict_proba(X_test)
    except Exception:
        probabilities = None

    cv_mean = None
    cv_std = None
    _, train_counts = np.unique(y_train, return_counts=True)
    usable_folds = min(cv_folds, int(train_counts.min())) if len(train_counts) else 0
    if usable_folds >= 2:
        cv = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="f1_weighted")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

    class_indexes = list(range(len(labels)))
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=class_indexes,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).T

    return TrainedResult(
        name=name,
        estimator=estimator,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
        recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
        f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
        cv_f1_mean=cv_mean,
        cv_f1_std=cv_std,
        confusion=confusion_matrix(y_test, y_pred, labels=class_indexes),
        predictions=y_pred,
        probabilities=probabilities,
        y_test=np.asarray(y_test),
        report=report_df,
    )


def roc_curve_data(y_test: np.ndarray, probabilities: np.ndarray | None, n_classes: int):
    if probabilities is None:
        return None
    try:
        if n_classes == 2 and probabilities.ndim == 2 and probabilities.shape[1] > 1:
            if len(np.unique(y_test)) < 2:
                return None
            fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
            return [("binary", fpr, tpr, auc(fpr, tpr))]
        if n_classes > 2 and probabilities.ndim == 2 and probabilities.shape[1] == n_classes:
            y_bin = label_binarize(y_test, classes=list(range(n_classes)))
            fpr, tpr, _ = roc_curve(y_bin.ravel(), probabilities.ravel())
            return [("micro-average", fpr, tpr, auc(fpr, tpr))]
    except ValueError:
        return None
    return None


def plot_confusion(result: TrainedResult, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ConfusionMatrixDisplay(result.confusion, display_labels=labels).plot(
        ax=ax,
        cmap="YlGnBu",
        colorbar=False,
        values_format="d",
    )
    ax.set_title(result.name)
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, clear_figure=True)


def plot_roc(results: list[TrainedResult], n_classes: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = False
    for result in results:
        data = roc_curve_data(result.y_test, result.probabilities, n_classes)
        if data is None:
            continue
        for label, fpr, tpr, auc_score in data:
            ax.plot(fpr, tpr, label=f"{result.name} ({label}, AUC={auc_score:.2f})")
            plotted = True
    if not plotted:
        st.info("ROC curves are unavailable for the evaluated models.")
        return
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8792a2", alpha=0.8)
    ax.set_title("ROC Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize="x-small")
    st.pyplot(fig, clear_figure=True)


def render_eda(df: pd.DataFrame, target_column: str) -> None:
    st.header("Exploratory Data Analysis")
    st.markdown('<div class="section-note">Data quality, distributions, target balance, and feature relationships.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    with st.expander("Dataset preview and column profile", expanded=True):
        st.dataframe(df.head(15), use_container_width=True)
        st.dataframe(summarize_dataframe(df), use_container_width=True)

    target_counts = df[target_column].astype(str).value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax, palette="viridis")
    ax.set_title(f"{target_column} class balance")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        st.subheader("Numeric Profile")
        st.dataframe(profile_numeric(df), use_container_width=True)

        corr = df[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 6.5))
        sns.heatmap(corr, cmap="vlag", center=0, annot=True, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig, clear_figure=True)

        selected_numeric = st.multiselect("Numeric distributions", numeric_columns, default=numeric_columns[:4])
        if selected_numeric:
            fig, axes = plt.subplots(len(selected_numeric), 2, figsize=(11, 3.2 * len(selected_numeric)))
            axes = np.atleast_2d(axes)
            for row, col in enumerate(selected_numeric):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[row, 0], color="#247a6b")
                sns.boxplot(x=df[target_column].astype(str), y=df[col], ax=axes[row, 1], color="#8fb9aa")
                axes[row, 0].set_title(f"{col} distribution")
                axes[row, 1].set_title(f"{col} by {target_column}")
                axes[row, 1].tick_params(axis="x", rotation=20)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

    categorical_columns = [col for col in df.select_dtypes(exclude=[np.number]).columns if col != target_column]
    if categorical_columns:
        st.subheader("Categorical Relationships")
        selected_cats = st.multiselect("Categorical count plots", categorical_columns, default=categorical_columns[:3])
        for col in selected_cats:
            crosstab = pd.crosstab(df[col].astype(str), df[target_column].astype(str), normalize="index")
            fig, ax = plt.subplots(figsize=(8, 3.8))
            crosstab.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
            ax.set_title(f"{col} vs {target_column}")
            ax.set_ylabel("Class share")
            ax.legend(title=target_column, bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)


def render_modeling(
    X: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    controls: dict[str, Any],
    uploaded_model: Any,
) -> tuple[list[TrainedResult], TrainedResult | None]:
    st.header("Model Comparison & Evaluation")
    st.markdown('<div class="section-note">Pipelines include imputation, scaling, one-hot encoding, model training, test metrics, cross-validation, and confusion matrices.</div>', unsafe_allow_html=True)

    if len(np.unique(y)) < 2:
        st.warning("The target needs at least two classes for classification.")
        return [], None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=controls["test_size"],
        random_state=RANDOM_STATE,
        stratify=safe_stratify(y),
    )

    catalog = build_model_catalog(X, controls, len(labels))
    if uploaded_model is not None and hasattr(uploaded_model, "fit"):
        catalog.append(("Uploaded Model", uploaded_model))

    results: list[TrainedResult] = []
    progress = st.progress(0, text="Training models...")
    for index, (name, estimator) in enumerate(catalog, start=1):
        try:
            result = evaluate_estimator(name, estimator, X_train, X_test, y_train, y_test, labels, controls["cv_folds"])
            results.append(result)
        except Exception as exc:
            st.warning(f"{name} could not be evaluated: {exc}")
        progress.progress(index / len(catalog), text=f"Evaluated {index} of {len(catalog)} models")
    progress.empty()

    if not results:
        st.error("No models could be trained with the current dataset.")
        return [], None

    summary = pd.DataFrame(
        [
            {
                "model": item.name,
                "accuracy": item.accuracy,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
                "cv_f1_mean": item.cv_f1_mean,
                "cv_f1_std": item.cv_f1_std,
            }
            for item in results
        ]
    ).sort_values(["cv_f1_mean", "f1"], ascending=False, na_position="last")

    best_name = summary.iloc[0]["model"]
    best_result = next(item for item in results if item.name == best_name)
    st.markdown(f'<div class="best-model"><strong>Best model:</strong> {best_name} with weighted F1 {best_result.f1:.3f}</div>', unsafe_allow_html=True)

    st.dataframe(
        summary.style.format(
            {
                "accuracy": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
                "cv_f1_mean": "{:.3f}",
                "cv_f1_std": "{:.3f}",
            }
        ).apply(lambda row: ["background-color: #eaf6f1" if row["model"] == best_name else "" for _ in row], axis=1),
        use_container_width=True,
    )

    st.subheader("Confusion Matrix and Classification Report")
    selected_result_name = st.selectbox("Inspect model", [item.name for item in results], index=[item.name for item in results].index(best_name))
    selected_result = next(item for item in results if item.name == selected_result_name)
    left, right = st.columns([1, 1])
    with left:
        plot_confusion(selected_result, labels)
    with right:
        st.dataframe(selected_result.report.round(3), use_container_width=True)

    st.subheader("ROC Curves")
    plot_roc(results, len(labels))

    st.subheader("Model Hyperparameters")
    params = selected_result.estimator.get_params() if hasattr(selected_result.estimator, "get_params") else {}
    params_df = pd.DataFrame({"parameter": list(params.keys()), "value": [str(value) for value in params.values()]})
    st.dataframe(params_df, use_container_width=True, height=280)
    return results, best_result


def raw_manual_inputs(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    columns = st.columns(3)
    for idx, column in enumerate(feature_columns):
        target = columns[idx % 3]
        series = df[column]
        with target:
            if pd.api.types.is_numeric_dtype(series):
                clean = pd.to_numeric(series, errors="coerce")
                default = float(clean.median()) if not clean.dropna().empty else 0.0
                min_value = float(clean.min()) if not clean.dropna().empty else 0.0
                max_value = float(clean.max()) if not clean.dropna().empty else 1.0
                if min_value == max_value:
                    max_value = min_value + 1.0
                step = max((max_value - min_value) / 100, 0.01)
                inputs[column] = st.number_input(column, min_value=min_value, max_value=max_value, value=default, step=step)
            else:
                choices = sorted(series.dropna().astype(str).unique().tolist()) or [""]
                inputs[column] = st.selectbox(column, choices)
    return inputs


def render_prediction(
    df: pd.DataFrame,
    X: pd.DataFrame,
    target_column: str,
    label_encoder: LabelEncoder,
    results: list[TrainedResult],
    best_result: TrainedResult | None,
    uploaded_model: Any,
) -> None:
    st.header("Prediction")
    st.markdown('<div class="section-note">Predict one patient record using the best trained pipeline or an uploaded compatible model.</div>', unsafe_allow_html=True)

    candidates: dict[str, Any] = {}
    if best_result is not None:
        candidates[f"Best trained model: {best_result.name}"] = best_result.estimator
    if uploaded_model is not None and hasattr(uploaded_model, "predict"):
        candidates["Uploaded model"] = uploaded_model
    for result in results:
        candidates[result.name] = result.estimator

    if not candidates:
        st.warning("Train or upload a model before prediction.")
        return

    selected_model_name = st.selectbox("Prediction model", list(candidates.keys()))
    selected_model = candidates[selected_model_name]
    raw_input = raw_manual_inputs(clean_columns(df), X.columns.tolist())

    if st.button("Predict", type="primary"):
        one_row = pd.DataFrame([raw_input], columns=X.columns)
        try:
            prediction = selected_model.predict(one_row)
            decoded = label_encoder.inverse_transform(np.asarray(prediction, dtype=int))
            st.success(f"Predicted {target_column}: {decoded[0]}")
            try:
                probabilities = selected_model.predict_proba(one_row)
                prob_df = pd.DataFrame(probabilities, columns=label_encoder.classes_)
                st.dataframe(prob_df.T.rename(columns={0: "probability"}).style.format("{:.3f}"), use_container_width=True)
            except Exception:
                st.info("This model does not expose prediction probabilities.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def main() -> None:
    inject_style()
    st.sidebar.title("Cirrhosis ML Studio")
    uploaded_csv = st.sidebar.file_uploader("Upload dataset (.csv)", type=["csv"])
    uploaded_model_file = st.sidebar.file_uploader("Upload pre-trained model", type=["pkl", "joblib", "pickle"])
    use_default = st.sidebar.checkbox("Use default local dataset", value=True)

    try:
        df = load_csv(uploaded_csv.getvalue() if uploaded_csv is not None else None, use_default)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    if df.empty:
        st.warning("Upload a dataset or enable the default dataset.")
        return

    df = clean_columns(df)
    target_candidates = [col for col in df.columns if col.lower() in {"status", "stage"}]
    if not target_candidates:
        target_candidates = df.columns.tolist()
    target_column = st.sidebar.selectbox("Prediction target", target_candidates, index=0)

    controls = sidebar_hyperparameters()

    uploaded_model = None
    if uploaded_model_file is not None:
        try:
            uploaded_model = load_pickle_model(uploaded_model_file.getvalue())
            st.sidebar.success("Uploaded model loaded.")
        except Exception as exc:
            st.sidebar.error(f"Could not load uploaded model: {exc}")

    st.title("Cirrhosis Machine Learning Studio")
    st.caption("EDA, transformer-based preprocessing, model comparison, cross-validation, and manual prediction for Status or Stage.")

    try:
        X, y, label_encoder = prepare_target(df, target_column)
    except ValueError as exc:
        st.error(str(exc))
        return

    tab_eda, tab_modeling, tab_prediction = st.tabs(["EDA", "Modeling", "Prediction"])
    with tab_eda:
        render_eda(df, target_column)
    with tab_modeling:
        results, best_result = render_modeling(X, y, label_encoder.classes_.tolist(), controls, uploaded_model)
    with tab_prediction:
        render_prediction(df, X, target_column, label_encoder, results, best_result, uploaded_model)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Random state: {RANDOM_STATE}. XGBoost baseline also uses 46.")


if __name__ == "__main__":
    main()
