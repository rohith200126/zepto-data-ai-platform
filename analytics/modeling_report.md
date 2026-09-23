# Module 2 — Task 2: Predictive Modeling Report

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
              Model  Accuracy  Precision  Recall     F1    AUC
Logistic Regression    0.8045     0.7931  0.6667 0.7244 0.8445
      Decision Tree    0.7654     0.7547  0.5797 0.6557 0.7971
      Random Forest    0.8212     0.8136  0.6957 0.7500 0.8300
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
             Strategy  Precision  Recall     F1
             Baseline     0.7931  0.6667 0.7244
Class Weight Balanced     0.7297  0.7826 0.7552
                SMOTE     0.7397  0.7826 0.7606
```

The tested strategy with the highest F1 score was:

**SMOTE — F1 = 0.7606**

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
{
  "model__max_depth": 10,
  "model__max_features": "sqrt",
  "model__n_estimators": 100
}
```

Best cross-validation F1:

**0.7452**

OOB score:

**0.8062**

Tuned Random Forest test metrics:

- Accuracy: **0.8212**
- Precision: **0.8246**
- Recall: **0.6812**
- F1: **0.7460**
- AUC: **0.8424**

## 8. Regression side-task

Fare was predicted from the other selected cleaned Titanic features
using multivariate linear regression.

- MAE: **20.8977**
- RMSE: **30.5328**
- R²: **0.3975**
- Adjusted R²: **0.3617**

The residual plot is saved as:

`outputs/regression_residual_plot.png`

### Heteroscedasticity conclusion

The residual plot shows evidence consistent with heteroscedasticity: the spread of residual magnitudes changes as predicted fare increases. The correlation between predicted fare and absolute residual is 0.546. This is an interpretation aid, not a formal heteroscedasticity test.

This is a residual-spread interpretation aid, not a formal
heteroscedasticity statistical test.

## 9. Final model comparison

```text
            Model Type                          Model  Accuracy  Precision  Recall     F1    AUC     MAE    RMSE     R2  Adjusted_R2
        Classification            Logistic Regression    0.8045     0.7931  0.6667 0.7244 0.8445     NaN     NaN    NaN          NaN
        Classification                  Decision Tree    0.7654     0.7547  0.5797 0.6557 0.7971     NaN     NaN    NaN          NaN
        Classification                  Random Forest    0.8212     0.8136  0.6957 0.7500 0.8300     NaN     NaN    NaN          NaN
Classification - Tuned            Tuned Random Forest    0.8212     0.8246  0.6812 0.7460 0.8424     NaN     NaN    NaN          NaN
            Regression Multivariate Linear Regression       NaN        NaN     NaN    NaN    NaN 20.8977 30.5328 0.3975       0.3617
```

Classification metrics and regression metrics are intentionally
kept as separate metric groups because the two tasks measure
different things and their metric values are not directly comparable.

## 10. Final written recommendation

For the Titanic survival classification task, the final pipeline uses Random Forest. On the held-out test set it achieved accuracy=0.821, precision=0.814, recall=0.696, F1=0.750, and AUC=0.830. The selection uses the classification metrics together, with F1 as the primary comparison and AUC/accuracy used as tie-breaking measures. Classification and regression metrics are not treated as directly comparable numbers because they measure different predictive tasks.

## 11. Complete saved pipeline

The final selected pipeline is:

**Random Forest**

Saved to:

`outputs/best_model_pipeline.joblib`

The saved object contains the preprocessing steps and final estimator
together. It was reloaded with `joblib.load()` and tested using raw,
unpreprocessed feature values.

Reload test:

- Predicted survival: **0**
- Survival probability: **0.178889**

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
