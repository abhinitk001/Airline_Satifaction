import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Training shape:", train.shape)
print("Test shape:", test.shape)

TARGET = "satisfaction"
ID = "id"

# Preserve test IDs for the submission
test_ids = test[ID].copy()

# Separate features and target
X = train.drop(columns=[TARGET, ID])
y = train[TARGET].astype(int)

X_test = test.drop(columns=[ID])

# Ensure training and test columns match
X_test = X_test.reindex(columns=X.columns)

# ------------------------------------------------------------
# 2. IDENTIFY CATEGORICAL FEATURES
# ------------------------------------------------------------

cat_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# CatBoost handles numeric missing values natively.
# Convert categorical values to strings and handle missing values.
for col in cat_features:
    X[col] = X[col].fillna("__MISSING__").astype(str)
    X_test[col] = X_test[col].fillna("__MISSING__").astype(str)

print("Categorical features:", cat_features)
print("Missing values in target:", y.isna().sum())

# ------------------------------------------------------------
# 3. STRATIFIED VALIDATION SPLIT
# ------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ------------------------------------------------------------
# 4. TRAIN CATBOOST MODEL
# ------------------------------------------------------------

model = CatBoostClassifier(
    iterations=4000,
    learning_rate=0.02,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    thread_count=-1
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=150,
    verbose=200
)

# ------------------------------------------------------------
# 5. EVALUATE MODEL
# ------------------------------------------------------------

val_probabilities = model.predict_proba(X_val)[:, 1]
val_predictions = (val_probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_val, val_predictions)
auc = roc_auc_score(y_val, val_probabilities)

print("\n--- VALIDATION RESULTS ---")
print(f"Accuracy: {accuracy:.5f}")
print(f"ROC-AUC:  {auc:.5f}")
print("Best iteration:", model.best_iteration_)

# ------------------------------------------------------------
# 6. RETRAIN ON THE FULL TRAINING DATA
# ------------------------------------------------------------

best_iterations = model.best_iteration_

if best_iterations is None or best_iterations < 0:
    best_iterations = 1000
else:
    best_iterations += 1

final_model = CatBoostClassifier(
    iterations=best_iterations,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    thread_count=-1
)

final_model.fit(
    X,
    y,
    cat_features=cat_features,
    verbose=200
)

# ------------------------------------------------------------
# 7. PREDICT THE KAGGLE TEST DATA
# ------------------------------------------------------------

test_probabilities = final_model.predict_proba(X_test)[:, 1]

# Binary class predictions
test_predictions = (test_probabilities >= 0.5)

# ------------------------------------------------------------
# 8. CREATE SUBMISSION FILES
# ------------------------------------------------------------

# For competitions expecting True/False class labels
submission = pd.DataFrame({
    "id": test_ids,
    "satisfaction": test_predictions
})

submission.to_csv("submission.csv", index=False)

# Alternative file for competitions evaluated using ROC-AUC
# Use this only if the competition expects prediction probabilities.
submission_prob = pd.DataFrame({
    "id": test_ids,
    "satisfaction": test_probabilities
})

submission_prob.to_csv("submission_probabilities.csv", index=False)

print("\nSubmission files created successfully!")
print("submission.csv")
print("submission_probabilities.csv")

print("\nSubmission preview:")
print(submission.head())

print("\nSubmission shape:", submission.shape)
print("Expected test rows:", len(test))
print("Files contain matching row counts:", len(submission) == len(test))
