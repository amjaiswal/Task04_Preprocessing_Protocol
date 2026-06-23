import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from preprocess import create_preprocessor

# Load dataset
df = pd.read_csv("data/employee_attrition.csv")

# Features and Target
X = df[
    [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "Gender",
        "MaritalStatus",
        "Department"
    ]
]

y = df["Attrition"]

# ----------------------------------------------------
# Add a few missing values (for learning/demo purposes)
# ----------------------------------------------------
X.loc[5, "Age"] = None
X.loc[10, "MonthlyIncome"] = None
X.loc[15, "Gender"] = None
X.loc[20, "Department"] = None

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create preprocessing pipeline
preprocessor = create_preprocessor()

# Fit ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

# Save fitted preprocessor
os.makedirs("saved_models", exist_ok=True)

joblib.dump(
    preprocessor,
    "saved_models/preprocessor.pkl"
)

print("====================================")
print("Preprocessor trained successfully!")
print("Training Shape :", X_train_processed.shape)
print("Testing Shape  :", X_test_processed.shape)
print("Preprocessor saved successfully!")
print("====================================")