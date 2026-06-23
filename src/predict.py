import joblib
import pandas as pd

# Load saved preprocessor
preprocessor = joblib.load("saved_models/preprocessor.pkl")

# New employee data
new_employee = pd.DataFrame({
    "Age": [29],
    "MonthlyIncome": [45000],
    "DistanceFromHome": [12],
    "Gender": ["Male"],
    "MaritalStatus": ["Single"],
    "Department": ["Research & Development"]
})

# Transform data
processed_data = preprocessor.transform(new_employee)

print("====================================")
print("New Employee Data")
print(new_employee)
print("\nProcessed Shape:", processed_data.shape)
print("\nProcessed Data:")
print(processed_data)
print("====================================")