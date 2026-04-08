import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("AeroData-Plant1-5.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['plant_no', 'date']).reset_index(drop=True)

# =========================
# CUSTOM IMPUTATION
# =========================
def custom_impute(group):
    group = group.sort_values('date')
    for col in group.columns:
        if group[col].dtype != 'object':
            for i in range(len(group)):
                if pd.isna(group.iloc[i][col]):
                    if i > 0 and i < len(group)-1:
                        prev_val = group.iloc[i-1][col]
                        next_val = group.iloc[i+1][col]
                        if not pd.isna(prev_val) and not pd.isna(next_val):
                            group.iloc[i, group.columns.get_loc(col)] = (prev_val + next_val) / 2
    return group

df = df.groupby('plant_no').apply(custom_impute).reset_index(drop=True)

# =========================
# FEATURES & TARGET
# =========================
features = ['ave_ph', 'ave_do', 'ave_tds', 'ave_temp', 'ave_humidity']
target = 'leaves'

# =========================
# TRAIN TEST SPLIT
# =========================
train_df = df[df['plant_no'].isin([1,2,3])]
test_df = df[df['plant_no'].isin([4,5])]

X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[target]
y_test = test_df[target]

# =========================
# SCALER FOR SVR
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# METRICS FUNCTION
# =========================
def evaluate(y_true, y_pred):
    return r2_score(y_true, y_pred)

# =========================
# HYPERPARAMETER GRIDS
# =========================
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

param_grid_svr = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# =========================
# BASE MODELS
# =========================
base_models = {
    "MLR": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "SVR": SVR()
}

# =========================
# TRAIN & EVALUATE BASE MODELS
# =========================
print(f"\n=== TARGET: {target.upper()} ===\n")

for name, model in base_models.items():
    Xtr, Xte = (X_train_scaled, X_test_scaled) if name=="SVR" else (X_train, X_test)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    r2 = evaluate(y_test, y_pred)
    print(f"{name} Base Model R2: {r2:.4f}")

# =========================
# TUNED MODELS
# =========================
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='r2')
rf_grid.fit(X_train, y_train)

svr_grid = GridSearchCV(SVR(), param_grid_svr, cv=3, scoring='r2')
svr_grid.fit(X_train_scaled, y_train)

tuned_models = {
    "MLR": base_models["MLR"],
    "RandomForest": rf_grid.best_estimator_,
    "SVR": svr_grid.best_estimator_
}

print("\n=== TUNED MODELS ===")
tuned_results = {}
tuned_r2_scores = {}

for name, model in tuned_models.items():
    Xtr, Xte = (X_train_scaled, X_test_scaled) if name=="SVR" else (X_train, X_test)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    tuned_results[name] = y_pred
    r2 = evaluate(y_test, y_pred)
    tuned_r2_scores[name] = r2
    print(f"{name} Tuned Model R2: {r2:.4f}")

# =========================
# BAR GRAPH: TUNED MODELS R2
# =========================
r2_df = pd.DataFrame(list(tuned_r2_scores.items()), columns=['Model', 'R2 Score'])

plt.figure(figsize=(7,5))
sns.barplot(x='Model', y='R2 Score', data=r2_df)

plt.title("Comparison of Tuned Models (R² Score)")
plt.xlabel("Model")
plt.ylabel("R² Score")

# Display values (4 decimal places)
for index, row in r2_df.iterrows():
    plt.text(index, row['R2 Score'], f"{row['R2 Score']:.4f}", 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# =========================
# FEATURE IMPORTANCE (Random Forest)
# =========================
rf_model = tuned_models["RandomForest"]
if hasattr(rf_model, "feature_importances_"):
    print("\nFeature Importance (RandomForest):")
    for f, imp in zip(features, rf_model.feature_importances_):
        print(f"{f}: {imp:.4f}")
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=features, y=rf_model.feature_importances_)
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

# =========================
# ACTUAL VS PREDICTED PLOT
# =========================
plt.figure(figsize=(8,6))
for name, y_pred in tuned_results.items():
    plt.scatter(y_test, y_pred, label=f"{name} Tuned")

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Leaves")
plt.ylabel("Predicted Leaves")
plt.title("Actual vs Predicted Leaves")
plt.legend()
plt.tight_layout()
plt.show()