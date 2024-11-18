import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset initialization (extracted from the excel file)
data = {
    "EMIRATE": [
        "Abu Dhabi", "Abu Dhabi", "Abu Dhabi", "Abu Dhabi", "Abu Dhabi", "Abu Dhabi",
        "Dubai", "Dubai", "Dubai", "Dubai", "Dubai", "Dubai",
        "Sharjah", "Sharjah", "Sharjah", "Sharjah", "Sharjah", "Sharjah",
        "Ajman", "Ajman", "Ajman", "Ajman", "Ajman", "Ajman",
        "Umm Al Quwain", "Umm Al Quwain", "Umm Al Quwain", "Umm Al Quwain", "Umm Al Quwain", "Umm Al Quwain",
        "Fujairah", "Fujairah", "Fujairah", "Fujairah", "Fujairah", "Fujairah",
        "Ras Al Khaimah", "Ras Al Khaimah", "Ras Al Khaimah", "Ras Al Khaimah", "Ras Al Khaimah", "Ras Al Khaimah"
    ],
    "CAUSE": [
        "Failure to Yield", "Reckless and Hazardous Driving", "Traffic Signal and Lane Violations", "Distracted or Inattentive Driving",
        "Intoxicated Driving", "Environmental and Mechanical Risks",
        "Failure to Yield", "Reckless and Hazardous Driving", "Traffic Signal and Lane Violations", "Intoxicated Driving",
        "Distracted or Inattentive Driving", "Environmental and Mechanical Risks",
        "Failure to Yield", "Distracted or Inattentive Driving", "Reckless and Hazardous Driving", "Intoxicated Driving",
        "Environmental and Mechanical Risks", "Traffic Signal and Lane Violations",
        "Distracted or Inattentive Driving", "Failure to Yield", "Reckless and Hazardous Driving", "Traffic Signal and Lane Violations",
        "Intoxicated Driving", "Environmental and Mechanical Risks",
        "Distracted or Inattentive Driving", "Failure to Yield", "Intoxicated Driving", "Reckless and Hazardous Driving",
        "Traffic Signal and Lane Violations", "Environmental and Mechanical Risks",
        "Failure to Yield", "Reckless and Hazardous Driving", "Distracted or Inattentive Driving", "Traffic Signal and Lane Violations",
        "Intoxicated Driving", "Environmental and Mechanical Risks",
        "Failure to Yield", "Distracted or Inattentive Driving", "Reckless and Hazardous Driving", "Traffic Signal and Lane Violations",
        "Intoxicated Driving", "Environmental and Mechanical Risks"
    ],
    "FREQUENCY": [
        963, 330, 283, 260, 87, 17,
        735, 355, 95, 82, 70, 8,
        248, 178, 64, 14, 11, 5,
        91, 48, 28, 3, 1, 0,
        25, 11, 2, 1, 0, 0,
        44, 39, 33, 6, 4, 2,
        56, 79, 25, 13, 5, 3
    ]
}

df = pd.DataFrame(data)

# Add predicted severity for Algorithm 1
def classify_severity_algo1(row):
    relative_frequency = row["FREQUENCY"] / df[df["EMIRATE"] == row["EMIRATE"]]["FREQUENCY"].sum()
    absolute_frequency = df.groupby("CAUSE")["FREQUENCY"].transform(lambda x: x / x.sum())[row.name]

    severity = "High" if relative_frequency >= 0.20 else "Medium" if relative_frequency >= 0.05 else "Low"

    if severity == "High" and absolute_frequency >= 0.20:
        severity = "Medium"
    elif severity == "Low" and absolute_frequency < 0.05:
        severity = "Medium"

    if row["CAUSE"] == "Intoxicated Driving":
        severity = "High"
    elif row["CAUSE"] == "Environmental and Mechanical Risks" and row["FREQUENCY"] < 5:
        severity = "Low"

    return severity

df["Severity Algorithm 1"] = df.apply(classify_severity_algo1, axis=1)

# Add predicted severity for Algorithm 2
severity_map = {
    "Intoxicated Driving": "High",
    "Reckless and Hazardous Driving": "High",
    "Failure to Yield": "Medium",
    "Traffic Signal and Lane Violations": "Medium",
    "Distracted or Inattentive Driving": "Low",
    "Environmental and Mechanical Risks": "Low"
}

def classify_severity_algo2(row):
    base_severity = severity_map[row["CAUSE"]]
    emirate_data = df[df["EMIRATE"] == row["EMIRATE"]]
    freq_percentile = np.percentile(emirate_data["FREQUENCY"], [25, 75])

    if base_severity == "Medium":
        if row["FREQUENCY"] >= freq_percentile[1]:  # Top 25%
            return "High"
        elif row["FREQUENCY"] <= freq_percentile[0]:  # Bottom 25%
            return "Low"
    return base_severity

df["Severity Algorithm 2"] = df.apply(classify_severity_algo2, axis=1)

# Simulate Actual Severity
np.random.seed(42)
actual_severity_map = {
    "Failure to Yield": [0.3, 0.5, 0.2],
    "Reckless and Hazardous Driving": [0.5, 0.4, 0.1],
    "Traffic Signal and Lane Violations": [0.2, 0.6, 0.2],
    "Distracted or Inattentive Driving": [0.1, 0.4, 0.5],
    "Intoxicated Driving": [0.7, 0.2, 0.1],
    "Environmental and Mechanical Risks": [0.05, 0.25, 0.7]
}

def simulate_actual_severity(row):
    probs = actual_severity_map[row["CAUSE"]]
    return np.random.choice(["High", "Medium", "Low"], p=probs)

df["Actual Severity"] = df.apply(simulate_actual_severity, axis=1)

# Evaluate the performance of both algorithms
results = {}

def evaluate_algorithm(predicted, actual, algorithm_name):
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average="macro")
    f1 = f1_score(actual, predicted, average="macro")
    cm = confusion_matrix(actual, predicted, labels=["High", "Medium", "Low"])
    
    results[algorithm_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "F1 Score": f1,
        "Confusion Matrix": cm
    }

evaluate_algorithm(df["Severity Algorithm 1"], df["Actual Severity"], "Algorithm 1")
evaluate_algorithm(df["Severity Algorithm 2"], df["Actual Severity"], "Algorithm 2")

# Display results in tabular format
metrics_df = pd.DataFrame({metric: [results["Algorithm 1"][metric], results["Algorithm 2"][metric]] for metric in ["Accuracy", "Precision", "F1 Score"]})
metrics_df.index = ["Algorithm 1", "Algorithm 2"]
print("|| PERFORMANCE METRICS ||")
print(metrics_df)

# Plot confusion matrices as heatmaps
labels = ["High", "Medium", "Low"]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot Algorithm 1 confusion matrix
sns.heatmap(results["Algorithm 1"]["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax[0])
ax[0].set_title("Confusion Matrix - Algorithm 1")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

# Plot Algorithm 2 confusion matrix
sns.heatmap(results["Algorithm 2"]["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax[1])
ax[1].set_title("Confusion Matrix - Algorithm 2")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()