"""
California Housing Price Prediction
===================================
A data science project: exploratory analysis and regression modelling of
median house prices across California districts.

Dataset : sklearn.datasets.fetch_california_housing (public, BSD-3-Clause)
Script  : runs the full analysis and exports the visualizations used in the
          portfolio. Run `python california_housing_analysis.py` to reproduce.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
OUT_DIR = "visualizations"
plt.rcParams["figure.dpi"] = 120

# ----------------------------------------------------------------------
# 1. Load the data
# ----------------------------------------------------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
print("Columns:", list(df.columns))
print("Missing values:", int(df.isna().sum().sum()))

# ----------------------------------------------------------------------
# 2. Exploratory data analysis
# ----------------------------------------------------------------------
print("\n--- Target (MedHouseVal) summary ---")
print(df["MedHouseVal"].describe().round(2).to_string())

# Fig 1: distribution of the target
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["MedHouseVal"], bins=50, kde=True, ax=ax, color="#2a6f97")
ax.set_title("Distribution of Median House Values (California, 1990 Census)")
ax.set_xlabel("Median house value ($100,000s)")
ax.set_ylabel("Number of districts")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_price_distribution.png")
plt.close(fig)

# Fig 2: correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix of Features")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_correlation_heatmap.png")
plt.close(fig)

# Fig 3: strongest feature vs target
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x="MedInc", y="MedHouseVal", data=df, alpha=0.25, ax=ax,
                color="#e07a5f")
sns.regplot(x="MedInc", y="MedHouseVal", data=df, scatter=False,
            color="black", ax=ax, line_kws={"lw": 2})
ax.set_title("Median Income vs. Median House Value")
ax.set_xlabel("Median income ($10,000s)")
ax.set_ylabel("Median house value ($100,000s)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_income_vs_price.png")
plt.close(fig)

print("\nSaved 3 exploratory figures to", OUT_DIR)

# ----------------------------------------------------------------------
# 3. Modelling
# ----------------------------------------------------------------------
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}
print("\n--- Model evaluation (test set) ---")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results[name] = (model, pred, rmse, r2)
    print(f"{name:<18} RMSE={rmse:.3f}  R2={r2:.3f}")

# Fig 4: actual vs predicted
model, pred, rmse, r2 = results["Random Forest"]
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, pred, alpha=0.3, color="#2a6f97")
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
ax.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual median house value")
ax.set_ylabel("Predicted median house value")
ax.set_title(f"Random Forest: Actual vs Predicted (R\u00b2={r2:.3f})")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_actual_vs_predicted.png")
plt.close(fig)

# Feature importance
rf = results["Random Forest"][0]
fig, ax = plt.subplots(figsize=(8, 5))
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.plot.barh(ax=ax, color="#2a6f97")
ax.set_title("Random Forest Feature Importances")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_feature_importance.png")
plt.close(fig)

print("\nAll figures generated. Analysis complete.")
