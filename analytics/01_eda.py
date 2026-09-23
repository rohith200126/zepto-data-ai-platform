import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


output_folder = Path("analytics")
output_folder.mkdir(exist_ok=True)

df = sns.load_dataset("titanic")

print("Dataset shape:", df.shape)

print("\nDataset information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
missing = df.isnull().sum()
missing_percentage = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Missing Percentage": missing_percentage
})

print(missing_table)

print("\nColumns with more than 50% missing values:")
print(missing_table[missing_table["Missing Percentage"] > 50])

if "deck" in df.columns:
    df = df.drop(columns=["deck"])

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

age_q1 = df["age"].quantile(0.25)
age_q3 = df["age"].quantile(0.75)
age_iqr = age_q3 - age_q1

age_lower = age_q1 - 1.5 * age_iqr
age_upper = age_q3 + 1.5 * age_iqr

age_outliers = df[
    (df["age"] < age_lower) |
    (df["age"] > age_upper)
]

fare_q1 = df["fare"].quantile(0.25)
fare_q3 = df["fare"].quantile(0.75)
fare_iqr = fare_q3 - fare_q1

fare_lower = fare_q1 - 1.5 * fare_iqr
fare_upper = fare_q3 + 1.5 * fare_iqr

fare_outliers = df[
    (df["fare"] < fare_lower) |
    (df["fare"] > fare_upper)
]

print("\nAge outlier count:", len(age_outliers))
print("Fare outlier count:", len(fare_outliers))

print("\nFare statistics:")
print("Mean:", df["fare"].mean())
print("Median:", df["fare"].median())
print("Mode:", df["fare"].mode()[0])
print("Skewness:", df["fare"].skew())

plt.figure(figsize=(10, 6))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(output_folder / "age_distribution.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df["fare"], bins=20, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(output_folder / "fare_distribution.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(y=df["age"])
plt.title("Age Boxplot")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(output_folder / "age_boxplot.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(y=df["fare"])
plt.title("Fare Boxplot")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(output_folder / "fare_boxplot.png")
plt.show()

survival_by_sex = df.groupby("sex")["survived"].mean()

print("\nSurvival rate by sex:")
print(survival_by_sex)

plt.figure(figsize=(8, 6))
sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values)
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(output_folder / "survival_by_sex.png")
plt.show()

survival_by_class = df.groupby("pclass")["survived"].mean()

print("\nSurvival rate by passenger class:")
print(survival_by_class)

plt.figure(figsize=(8, 6))
sns.barplot(x=survival_by_class.index, y=survival_by_class.values)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(output_folder / "survival_by_class.png")
plt.show()

survival_by_sex_class = df.groupby(
    ["pclass", "sex"]
)["survived"].mean().reset_index()

print("\nSurvival rate by sex and passenger class:")
print(survival_by_sex_class)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=survival_by_sex_class,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(output_folder / "survival_by_sex_class.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="survived", y="age")
plt.title("Age Distribution by Survival")
plt.xlabel("Survived")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(output_folder / "age_by_survival.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="pclass", y="fare")
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(output_folder / "fare_by_class.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="age",
    hue="sex",
    bins=20,
    kde=True,
    element="step"
)
plt.title("Age Distribution by Sex")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(output_folder / "age_by_sex.png")
plt.show()

survival_count = df["survived"].value_counts().sort_index()

plt.figure(figsize=(8, 6))
sns.barplot(
    x=survival_count.index,
    y=survival_count.values
)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(output_folder / "survival_count.png")
plt.show()

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation matrix:")
print(correlation_matrix)

correlation_matrix.to_csv(
    output_folder / "correlation_matrix.csv"
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="magma"
)
plt.title("Titanic Correlation Heatmap")
plt.tight_layout()
plt.savefig(output_folder / "correlation_heatmap.png")
plt.show()

correlation_pairs = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

correlation_pairs = correlation_pairs.stack()
correlation_pairs = correlation_pairs.abs().sort_values(
    ascending=False
)

print("\nStrongest two correlations:")
print(correlation_pairs.head(2))

age_mean = df["age"].mean()
age_std = df["age"].std()

fare_mean = df["fare"].mean()
fare_std = df["fare"].std()

df["standardized_age"] = (
    (df["age"] - age_mean) / age_std
)

df["standardized_fare"] = (
    (df["fare"] - fare_mean) / fare_std
)

standardized_check = df[
    ["standardized_age", "standardized_fare"]
].agg(["mean", "std"])

print("\nStandardization check:")
print(standardized_check)

df[
    ["standardized_age", "standardized_fare"]
].to_csv(
    output_folder / "standardized_age_fare.csv",
    index=False
)

df.to_csv(
    output_folder / "titanic.csv",
    index=False
)

print("\nCleaned Titanic dataset saved to analytics/titanic.csv")