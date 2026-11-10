import numpy as np
import matplotlib.pyplot as plt

cm = np.array([
    [11, 1, 2, 0, 3, 0, 0, 0],
    [1, 12, 1, 0, 0, 0, 0, 0],
    [1, 0, 8, 0, 3, 0, 4, 1],
    [0, 0, 0, 10, 0, 0, 0, 3],
    [0, 0, 3, 0, 9, 2, 0, 0],
    [3, 1, 0, 1, 1, 11, 0, 0],
    [1, 0, 6, 1, 0, 1, 4, 0],
    [0, 0, 0, 4, 0, 0, 0, 11]
])

directions = np.arange(0, 360, 45)

# Accuracy for each stimulus direction
correct = np.diag(cm)
total = cm.sum(axis=1)
accuracy = correct / total * 100

print("Accuracy by stimulus direction:")
for direction, acc in zip(directions, accuracy):
    print(f"{direction}°: {acc:.2f}%")

print(f"\nOverall accuracy: {correct.sum() / total.sum() * 100:.2f}%")


angular_errors = []
error_counts = {}
for i in range(len(directions)):
    for j in range(len(directions)):
        true_angle = directions[i]
        pred_ang = directions[j]
        diff = abs(true_angle-pred_ang)
        circular_err = min(diff,360-diff)
        count = cm[i,j]
        angular_errors.extend([circular_err]*count)
        error_counts [circular_err] = (error_counts.get(circular_err,0)+count)
angular_errors = np.array(angular_errors)
print(f"Mean Angular Error: {np.mean(angular_errors):.2f}°")
print(f"Median Angular Error: {np.median(angular_errors):.2f}°")
print("\nError Distribuyion")
for error in sorted(error_counts):
    percentage = error_counts[error] / len(angular_errors) * 100
    print(f"{error}: {error_counts[error]} predictions ({percentage:.2f}%)")

# Day 12: Logistic Regression Angular Error

linear_cm = np.array([
    [11, 0, 1, 0, 3, 2, 0, 0],
    [1, 9, 1, 0, 0, 3, 0, 0],
    [0, 1, 9, 0, 3, 0, 3, 1],
    [0, 0, 0, 10, 0, 0, 0, 3],
    [0, 1, 3, 1, 9, 0, 0, 0],
    [1, 3, 0, 2, 2, 9, 0, 0],
    [0, 2, 4, 0, 0, 1, 6, 0],
    [0, 0, 0, 4, 0, 0, 0, 11]
])

def calculate_angular_errors(matrix, directions):
    errors = []
    for i in range(len(directions)):
        for j in range(len(directions)):
            difference = abs(directions[i] - directions[j])
            circular_error = min(difference, 360 - difference)
            errors.extend([circular_error] * matrix[i, j])
    return np.array(errors)
linear_errors = calculate_angular_errors(linear_cm, directions)
print("\nModel Comparison")
print("------------------------------")
print(f"Neural Network: {np.mean(angular_errors):.2f}°")
print(f"Logistic Regression: {np.mean(linear_errors):.2f}°")
print("\n180° Errors:")
print(f"Neural Network: {np.sum(angular_errors == 180)}")
print(f"Logistic Regression: {np.sum(linear_errors == 180)}")

# Final neural network test confusion matrix
cm = np.array([
    [11, 1, 2, 0, 3, 0, 0, 0],
    [1, 12, 1, 0, 0, 0, 0, 0],
    [1, 0, 8, 0, 3, 0, 4, 1],
    [0, 0, 0, 10, 0, 0, 0, 3],
    [0, 0, 3, 0, 9, 2, 0, 0],
    [3, 1, 0, 1, 1, 11, 0, 0],
    [1, 0, 6, 1, 0, 1, 4, 0],
    [0, 0, 0, 4, 0, 0, 0, 11]
])

directions = np.arange(0, 360, 45)
print("180-Degree Error Breakdown")
print("-" * 35)
total_errors = 0
for i, direction in enumerate(directions):
    opposite_idx = (i + 4) % 8
    errors = cm[i, opposite_idx]
    total = cm[i].sum()
    total_errors += errors
    print(
        f"{direction:3d}° → "
        f"{directions[opposite_idx]:3d}°: "
        f"{errors} errors ({errors / total * 100:.2f}%)"
    )
print("-" * 35)
print(f"Total 180° Errors: {total_errors}")
