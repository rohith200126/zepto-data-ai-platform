from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

                                                                       
                  
                                                                       

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_classification_preprocessor():
    numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

def classification_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
    }

def save_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar=True,
        xticklabels=[0, 1],
        yticklabels=[0, 1],
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

                                                                       
                                      
                                                                       

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} was not found.\n"
        "Run 01_eda.py first so analytics/titanic.csv exists."
    )

df = pd.read_csv(DATA_PATH)
print("MODULE 2 - TASK 2: PREDICTIVE MODELING")
print(f"Loaded cleaned dataset: {DATA_PATH}")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

if "survived" not in df.columns:
    raise ValueError("The cleaned dataset must contain the 'survived' column.")

                                                                       
                                           
                                                                       

classification_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "sex",
    "embarked",
]

missing_columns = [
    col for col in classification_features
    if col not in df.columns
]
if missing_columns:
    raise ValueError(
        f"Missing classification columns in titanic.csv: {missing_columns}"
    )

X_cls = df[classification_features].copy()
y_cls = df["survived"].astype(int).copy()

class_counts = y_cls.value_counts().sort_index()
class_percentages = (
    y_cls.value_counts(normalize=True).sort_index() * 100
).round(2)

class_balance = pd.DataFrame(
    {
        "Count": class_counts,
        "Percentage": class_percentages,
    }
)

print("\nClass balance:")
print(class_balance)

class_balance.to_csv(
    OUTPUT_DIR / "class_balance.csv"
)

print(
    "\nWhy stratification is used:\n"
    "The Titanic target is not perfectly balanced. A stratified split "
    "preserves approximately the same survived/not-survived proportion "
    "in both training and test data."
)

            
                                                       
X_train, X_test, y_train, y_test = train_test_split(
    X_cls,
    y_cls,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_cls,
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")

print("\nTraining target proportions:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTesting target proportions:")
print(y_test.value_counts(normalize=True).sort_index())

                                                                       
                               
                                                                       

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

classification_results = []
roc_data = {}
fitted_pipelines = {}

for model_name, estimator in models.items():

    print("\n" + "-" * 90)
    print(model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_classification_preprocessor()),
            ("model", estimator),
        ]
    )

                                                      
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(
        y_test,
        y_pred,
        y_prob,
    )

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    for metric_name, value in metrics.items():
        print(f"{metric_name:10s}: {value:.4f}")

    classification_results.append(
        {
            "Model": model_name,
            **metrics,
        }
    )

    fitted_pipelines[model_name] = pipeline

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": metrics["AUC"],
    }

    save_confusion_matrix(
        cm,
        f"{model_name} Confusion Matrix",
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png",
    )

classification_df = pd.DataFrame(classification_results)

classification_df.to_csv(
    OUTPUT_DIR / "classification_model_comparison.csv",
    index=False,
)

print("\nClassification comparison:")
print(classification_df.round(4).to_string(index=False))

                                                                       
                                
                                                                       

tree_pipeline = fitted_pipelines["Decision Tree"]
tree_preprocessor = tree_pipeline.named_steps["preprocessor"]
tree_model = tree_pipeline.named_steps["model"]

try:
    tree_feature_names = tree_preprocessor.get_feature_names_out()
except Exception:
    tree_feature_names = [
        f"feature_{i}"
        for i in range(tree_model.n_features_in_)
    ]

plt.figure(figsize=(24, 14))

plot_tree(
    tree_model,
    feature_names=tree_feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=7,
)

plt.title("Decision Tree")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "decision_tree.png",
    dpi=180,
    bbox_inches="tight",
)
plt.show()
plt.close()

                                                                       
                         
                                                                       

plt.figure(figsize=(10, 7))

for model_name, data in roc_data.items():
    plt.plot(
        data["fpr"],
        data["tpr"],
        label=f"{model_name} AUC = {data['auc']:.3f}",
    )

plt.plot(
    [0, 1],
    [0, 1],
    "r--",
    label="Random classifier",
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "roc_curve_comparison.png",
    dpi=160,
    bbox_inches="tight",
)
plt.show()
plt.close()

                                                                       
                                  
                
                               
             
 
                                                                      
                                     
                                                                       

print("\n" + "=" * 90)
print("IMBALANCE HANDLING COMPARISON")

imbalance_results = []

                                                                       
             
                                                                       

baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", make_classification_preprocessor()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

baseline_pipeline.fit(X_train, y_train)

baseline_pred = baseline_pipeline.predict(X_test)

imbalance_results.append(
    {
        "Strategy": "Baseline",
        "Precision": precision_score(
            y_test,
            baseline_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            baseline_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            baseline_pred,
            zero_division=0,
        ),
    }
)

                                                                       
                          
                                                                       

balanced_pipeline = Pipeline(
    steps=[
        ("preprocessor", make_classification_preprocessor()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

balanced_pipeline.fit(X_train, y_train)

balanced_pred = balanced_pipeline.predict(X_test)

imbalance_results.append(
    {
        "Strategy": "Class Weight Balanced",
        "Precision": precision_score(
            y_test,
            balanced_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            balanced_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            balanced_pred,
            zero_division=0,
        ),
    }
)

                                                                       
          
                                                                       

try:
    from imblearn.over_sampling import SMOTE
except ImportError as exc:
    raise ImportError(
        "imbalanced-learn is required for SMOTE.\n"
        "Run: pip install imbalanced-learn"
    ) from exc

smote_preprocessor = make_classification_preprocessor()

                                
X_train_transformed = smote_preprocessor.fit_transform(X_train)

                                                                       
X_test_transformed = smote_preprocessor.transform(X_test)

smote = SMOTE(
    random_state=RANDOM_STATE
)

                                         
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_transformed,
    y_train,
)

print("\nSMOTE class-count check:")
print(
    "Before SMOTE:",
    y_train.value_counts().sort_index().to_dict(),
)
print(
    "After SMOTE :",
    pd.Series(y_train_smote).value_counts().sort_index().to_dict(),
)

smote_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)

smote_model.fit(
    X_train_smote,
    y_train_smote,
)

smote_pred = smote_model.predict(
    X_test_transformed
)

imbalance_results.append(
    {
        "Strategy": "SMOTE",
        "Precision": precision_score(
            y_test,
            smote_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            smote_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            smote_pred,
            zero_division=0,
        ),
    }
)

imbalance_df = pd.DataFrame(imbalance_results)

imbalance_df.to_csv(
    OUTPUT_DIR / "imbalance_comparison.csv",
    index=False,
)

print("\nImbalance comparison:")
print(imbalance_df.round(4).to_string(index=False))

plt.figure(figsize=(10, 6))

imbalance_df.set_index("Strategy")[
    ["Precision", "Recall", "F1"]
].plot(
    kind="bar",
    ax=plt.gca(),
)

plt.title("Imbalance Handling Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "imbalance_comparison.png",
    dpi=160,
    bbox_inches="tight",
)
plt.show()
plt.close()

best_imbalance_row = imbalance_df.loc[
    imbalance_df["F1"].idxmax()
]

best_imbalance_strategy = best_imbalance_row["Strategy"]
best_imbalance_f1 = best_imbalance_row["F1"]

print(
    f"\nImbalance conclusion: {best_imbalance_strategy} "
    f"produced the highest F1 score "
    f"({best_imbalance_f1:.4f}) among the tested strategies."
)

                                                                       
                                        
                                                                       

print("\n" + "=" * 90)
print("RANDOM FOREST HYPERPARAMETER TUNING")

rf_grid_pipeline = Pipeline(
    steps=[
        ("preprocessor", make_classification_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    estimator=rf_grid_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True,
)

grid_search.fit(
    X_train,
    y_train,
)

tuned_rf_pipeline = grid_search.best_estimator_
tuned_rf_model = tuned_rf_pipeline.named_steps["model"]

print("Best parameters:")
print(grid_search.best_params_)

print(
    f"Best CV F1: {grid_search.best_score_:.4f}"
)

print(
    f"OOB score: {tuned_rf_model.oob_score_:.4f}"
)

grid_search_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_search_results.to_csv(
    OUTPUT_DIR / "random_forest_gridsearch_results.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "random_forest_best_params.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "best_parameters": grid_search.best_params_,
            "best_cv_f1": float(grid_search.best_score_),
            "oob_score": float(tuned_rf_model.oob_score_),
        },
        file,
        indent=2,
    )

                                                                       
                                          
                                                                       

tuned_rf_pred = tuned_rf_pipeline.predict(X_test)
tuned_rf_prob = tuned_rf_pipeline.predict_proba(X_test)[:, 1]

tuned_rf_metrics = classification_metrics(
    y_test,
    tuned_rf_pred,
    tuned_rf_prob,
)

print("\nTuned Random Forest test metrics:")
for metric_name, value in tuned_rf_metrics.items():
    print(f"{metric_name:10s}: {value:.4f}")

tuned_rf_cm = confusion_matrix(
    y_test,
    tuned_rf_pred,
)

save_confusion_matrix(
    tuned_rf_cm,
    "Tuned Random Forest Confusion Matrix",
    "tuned_random_forest_confusion_matrix.png",
)

                                                                       
                                       
                                                                       

print("\n" + "=" * 90)
print("REGRESSION SIDE TASK: FARE PREDICTION")

regression_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked",
]

missing_regression_columns = [
    col for col in regression_features
    if col not in df.columns
]

if missing_regression_columns:
    raise ValueError(
        "Missing regression columns: "
        f"{missing_regression_columns}"
    )

X_reg = df[regression_features].copy()
y_reg = pd.to_numeric(
    df["fare"],
    errors="coerce",
)

valid_rows = y_reg.notna()

X_reg = X_reg.loc[valid_rows].copy()
y_reg = y_reg.loc[valid_rows].copy()

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

reg_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
]

reg_categorical_features = [
    "sex",
    "embarked",
]

def make_regression_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                reg_numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                reg_categorical_features,
            ),
        ],
        remainder="drop",
    )

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            make_regression_preprocessor(),
        ),
        (
            "model",
            LinearRegression(),
        ),
    ]
)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train,
)

y_reg_pred = regression_pipeline.predict(
    X_reg_test
)

residuals = (
    y_reg_test.to_numpy()
    - y_reg_pred
)

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred,
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred,
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred,
)

n = len(y_reg_test)

try:
    p = len(
        regression_pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )
except Exception:
    p = len(regression_features)

if n > p + 1:
    adjusted_r2 = (
        1
        - (1 - r2)
        * (n - 1)
        / (n - p - 1)
    )
else:
    adjusted_r2 = np.nan

print(f"MAE        : {mae:.6f}")
print(f"RMSE       : {rmse:.6f}")
print(f"R2         : {r2:.6f}")
print(f"Adjusted R2: {adjusted_r2:.6f}")

                                                                       
               
                                                                       

plt.figure(figsize=(10, 7))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.75,
)

plt.axhline(
    0,
    linestyle="--",
)

plt.title("Regression Residual Plot")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.grid(alpha=0.20)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "regression_residual_plot.png",
    dpi=160,
    bbox_inches="tight",
)

plt.show()
plt.close()

                                                                       
                                       
                                                                       

if np.std(y_reg_pred) == 0:
    abs_residual_corr = np.nan
else:
    abs_residual_corr = np.corrcoef(
        y_reg_pred,
        np.abs(residuals),
    )[0, 1]

if np.isnan(abs_residual_corr):

    heteroscedasticity_conclusion = (
        "The residual spread could not be assessed quantitatively "
        "because the absolute-residual correlation was undefined."
    )

elif abs(abs_residual_corr) >= 0.30:

    heteroscedasticity_conclusion = (
        "The residual plot shows evidence consistent with "
        "heteroscedasticity: the spread of residual magnitudes "
        "changes as predicted fare increases. The correlation "
        f"between predicted fare and absolute residual is "
        f"{abs_residual_corr:.3f}. This is an interpretation aid, "
        "not a formal heteroscedasticity test."
    )

else:

    heteroscedasticity_conclusion = (
        "The residual plot does not show strong evidence of "
        "heteroscedasticity from this spread check. The correlation "
        f"between predicted fare and absolute residual is "
        f"{abs_residual_corr:.3f}. This is an interpretation aid, "
        "not a formal heteroscedasticity test."
    )

print("\nHeteroscedasticity conclusion:")
print(heteroscedasticity_conclusion)

regression_results = pd.DataFrame(
    [
        {
            "Model": "Multivariate Linear Regression",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2,
            "Predicted_AbsResidual_Correlation": abs_residual_corr,
        }
    ]
)

regression_results.to_csv(
    OUTPUT_DIR / "regression_results.csv",
    index=False,
)

                                                                       
                           
                                                                   
                                                                       

final_rows = []

for row in classification_results:

    final_rows.append(
        {
            "Model Type": "Classification",
            "Model": row["Model"],
            "Accuracy": row["Accuracy"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "F1": row["F1"],
            "AUC": row["AUC"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Adjusted_R2": np.nan,
        }
    )

                                                         
final_rows.append(
    {
        "Model Type": "Classification - Tuned",
        "Model": "Tuned Random Forest",
        "Accuracy": tuned_rf_metrics["Accuracy"],
        "Precision": tuned_rf_metrics["Precision"],
        "Recall": tuned_rf_metrics["Recall"],
        "F1": tuned_rf_metrics["F1"],
        "AUC": tuned_rf_metrics["AUC"],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Adjusted_R2": np.nan,
    }
)

final_rows.append(
    {
        "Model Type": "Regression",
        "Model": "Multivariate Linear Regression",
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "AUC": np.nan,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
    }
)

final_comparison_df = pd.DataFrame(
    final_rows
)

final_comparison_df.to_csv(
    OUTPUT_DIR / "final_model_comparison.csv",
    index=False,
)

print("\n" + "=" * 90)
print("FINAL MODEL COMPARISON")
print(
    final_comparison_df.round(4).to_string(
        index=False
    )
)

                                                                       
                                                  
 
                                                    
                                                                  
                                                                       

classifier_candidates = []

for model_name, pipeline in fitted_pipelines.items():

    row = classification_df[
        classification_df["Model"] == model_name
    ].iloc[0]

    classifier_candidates.append(
        {
            "Model": model_name,
            "Pipeline": pipeline,
            "F1": row["F1"],
            "AUC": row["AUC"],
            "Accuracy": row["Accuracy"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
        }
    )

                              
classifier_candidates.append(
    {
        "Model": "Tuned Random Forest",
        "Pipeline": tuned_rf_pipeline,
        "F1": tuned_rf_metrics["F1"],
        "AUC": tuned_rf_metrics["AUC"],
        "Accuracy": tuned_rf_metrics["Accuracy"],
        "Precision": tuned_rf_metrics["Precision"],
        "Recall": tuned_rf_metrics["Recall"],
    }
)

best_classifier = sorted(
    classifier_candidates,
    key=lambda item: (
        item["F1"],
        item["AUC"],
        item["Accuracy"],
    ),
    reverse=True,
)[0]

best_classifier_name = best_classifier["Model"]
best_classifier_pipeline = best_classifier["Pipeline"]

                                                                       
                                     
                                                                       

pipeline_path = (
    OUTPUT_DIR
    / "best_model_pipeline.joblib"
)

joblib.dump(
    best_classifier_pipeline,
    pipeline_path,
)

print("\nComplete pipeline saved successfully:")
print(pipeline_path)

                                                                       
                                            
                                                                       

reloaded_pipeline = joblib.load(
    pipeline_path
)

raw_sample = X_test.iloc[[0]].copy()

reloaded_prediction = int(
    reloaded_pipeline.predict(
        raw_sample
    )[0]
)

reloaded_probability = float(
    reloaded_pipeline.predict_proba(
        raw_sample
    )[0, 1]
)

print("\nReloaded pipeline test:")
print(
    f"Predicted survival: {reloaded_prediction}"
)
print(
    f"Survival probability: "
    f"{reloaded_probability:.6f}"
)

                                                                       
                             
                                                                       

report_path = (
    BASE_DIR
    / "modeling_report.md"
)

classification_table = (
    classification_df.round(4)
    .to_string(index=False)
)

imbalance_table = (
    imbalance_df.round(4)
    .to_string(index=False)
)

final_table = (
    final_comparison_df.round(4)
    .to_string(index=False)
)

best_classifier_recommendation = (
    f"For the Titanic survival classification task, "
    f"the final pipeline uses {best_classifier_name}. "
    f"On the held-out test set it achieved "
    f"accuracy={best_classifier['Accuracy']:.3f}, "
    f"precision={best_classifier['Precision']:.3f}, "
    f"recall={best_classifier['Recall']:.3f}, "
    f"F1={best_classifier['F1']:.3f}, and "
    f"AUC={best_classifier['AUC']:.3f}. "
    f"The selection uses the classification metrics together, "
    f"with F1 as the primary comparison and AUC/accuracy used "
    f"as tie-breaking measures. Classification and regression "
    f"metrics are not treated as directly comparable numbers "
    f"because they measure different predictive tasks."
)

report = f"""# Module 2 — Task 2: Predictive Modeling Report

## 1. Dataset and train/test split

Task 2 reads the cleaned `analytics/titanic.csv` produced by Task 1.
It does not call `sns.load_dataset()` again.

The survival classification task uses a stratified 80/20 train/test split.
Stratification preserves approximately the same survived/not-survived
class proportion in the training and test sets.

## 2. Preprocessing

Classification numeric features (`pclass`, `age`, `sibsp`, `parch`, `fare`)
use median imputation followed by `StandardScaler`.

Categorical features (`sex`, `embarked`) use most-frequent imputation
followed by one-hot encoding.

The preprocessing is inside a `ColumnTransformer` and `Pipeline`.
It is fitted only on the training split and then used to transform
the test split. This prevents test-set information leakage.

## 3. Three required classifiers

```text
{classification_table}
```

All three classifiers use the same train/test split and are evaluated with:
confusion matrix, accuracy, precision, recall, F1 score and ROC-AUC.

## 4. Decision Tree

The Decision Tree is visualized in:

`outputs/decision_tree.png`

The tree includes transformed feature names and the class names
`Not Survived` and `Survived`.

## 5. ROC/AUC comparison

The ROC comparison is saved as:

`outputs/roc_curve_comparison.png`

AUC measures the model's ability to separate the two survival classes
over classification thresholds.

## 6. Imbalance handling

```text
{imbalance_table}
```

The tested strategy with the highest F1 score was:

**{best_imbalance_strategy} — F1 = {best_imbalance_f1:.4f}**

SMOTE was applied only to the training fold after the training-only
preprocessing step. The test fold was never oversampled.

## 7. Random Forest hyperparameter tuning

`GridSearchCV` tuned:

- `n_estimators`
- `max_depth`
- `max_features`

The Random Forest was constructed with `oob_score=True`.

Best parameters:

```text
{json.dumps(grid_search.best_params_, indent=2)}
```

Best cross-validation F1:

**{grid_search.best_score_:.4f}**

OOB score:

**{tuned_rf_model.oob_score_:.4f}**

Tuned Random Forest test metrics:

- Accuracy: **{tuned_rf_metrics["Accuracy"]:.4f}**
- Precision: **{tuned_rf_metrics["Precision"]:.4f}**
- Recall: **{tuned_rf_metrics["Recall"]:.4f}**
- F1: **{tuned_rf_metrics["F1"]:.4f}**
- AUC: **{tuned_rf_metrics["AUC"]:.4f}**

## 8. Regression side-task

Fare was predicted from the other selected cleaned Titanic features
using multivariate linear regression.

- MAE: **{mae:.4f}**
- RMSE: **{rmse:.4f}**
- R²: **{r2:.4f}**
- Adjusted R²: **{adjusted_r2:.4f}**

The residual plot is saved as:

`outputs/regression_residual_plot.png`

### Heteroscedasticity conclusion

{heteroscedasticity_conclusion}

This is a residual-spread interpretation aid, not a formal
heteroscedasticity statistical test.

## 9. Final model comparison

```text
{final_table}
```

Classification metrics and regression metrics are intentionally
kept as separate metric groups because the two tasks measure
different things and their metric values are not directly comparable.

## 10. Final written recommendation

{best_classifier_recommendation}

## 11. Complete saved pipeline

The final selected pipeline is:

**{best_classifier_name}**

Saved to:

`outputs/best_model_pipeline.joblib`

The saved object contains the preprocessing steps and final estimator
together. It was reloaded with `joblib.load()` and tested using raw,
unpreprocessed feature values.

Reload test:

- Predicted survival: **{reloaded_prediction}**
- Survival probability: **{reloaded_probability:.6f}**

## 12. Required generated artifacts

- `outputs/class_balance.csv`
- `outputs/classification_model_comparison.csv`
- `outputs/logistic_regression_confusion_matrix.png`
- `outputs/decision_tree_confusion_matrix.png`
- `outputs/random_forest_confusion_matrix.png`
- `outputs/decision_tree.png`
- `outputs/roc_curve_comparison.png`
- `outputs/imbalance_comparison.csv`
- `outputs/imbalance_comparison.png`
- `outputs/random_forest_gridsearch_results.csv`
- `outputs/random_forest_best_params.json`
- `outputs/tuned_random_forest_confusion_matrix.png`
- `outputs/regression_results.csv`
- `outputs/regression_residual_plot.png`
- `outputs/final_model_comparison.csv`
- `outputs/best_model_pipeline.joblib`
- `modeling_report.md`
"""

report_path.write_text(
    report,
    encoding="utf-8",
)

                                                                       
                                    
                                                                       

checks = {
    "Cleaned titanic.csv exists": DATA_PATH.exists(),
    "Stratified split performed": True,
    "Preprocessing fitted only on training data": True,
    "Logistic Regression trained": True,
    "Decision Tree trained": True,
    "Random Forest trained": True,
    "All 3 confusion matrices generated": all(
        (
            OUTPUT_DIR
            / (
                model_name.lower().replace(" ", "_")
                + "_confusion_matrix.png"
            )
        ).exists()
        for model_name in models
    ),
    "Accuracy/Precision/Recall/F1/AUC reported": True,
    "Decision Tree visualization generated": (
        OUTPUT_DIR / "decision_tree.png"
    ).exists(),
    "ROC/AUC comparison generated": (
        OUTPUT_DIR / "roc_curve_comparison.png"
    ).exists(),
    "Baseline/class_weight/SMOTE comparison": len(
        imbalance_df
    ) == 3,
    "SMOTE applied to training fold only": True,
    "GridSearchCV executed": True,
    "Best RF parameters reported": bool(
        grid_search.best_params_
    ),
    "RF OOB score available": hasattr(
        tuned_rf_model,
        "oob_score_",
    ),
    "Regression MAE/RMSE/R2/Adjusted R2": True,
    "Regression residual plot generated": (
        OUTPUT_DIR / "regression_residual_plot.png"
    ).exists(),
    "Heteroscedasticity conclusion written": bool(
        heteroscedasticity_conclusion
    ),
    "Final model comparison generated": (
        OUTPUT_DIR / "final_model_comparison.csv"
    ).exists(),
    "Complete pipeline saved": pipeline_path.exists(),
    "Saved pipeline reloaded successfully": (
        reloaded_pipeline is not None
    ),
    "Raw-input prediction test completed": True,
    "Written modeling report generated": report_path.exists(),
}

print("\n" + "=" * 90)
print("MODULE 2 TASK 2 ACCEPTANCE CHECKLIST")

all_passed = True

for requirement, passed in checks.items():

    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] {requirement}"
    )

    all_passed = (
        all_passed
        and passed
    )

print("\n" + "=" * 90)

if all_passed:
    print("MODULE 2 TASK 2 COMPLETED SUCCESSFULLY.")
else:
    print(
        "MODULE 2 TASK 2 FINISHED WITH ONE OR MORE FAILED CHECKS."
    )
