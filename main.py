import os

print("=" * 50)
print("Task 04 - AI/ML Preprocessing Protocol")
print("=" * 50)

print("\nRunning Training Pipeline...\n")
os.system("python src/train.py")

print("\nRunning Prediction Pipeline...\n")
os.system("python src/predict.py")

print("\nTask Completed Successfully!")