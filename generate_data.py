
import pandas as pd
import numpy as np

# Random data generation
np.random.seed(42)

n = 1000

# Student academic information
data = {
    "student_id": range(1, n + 1),
    "attendance": np.random.randint(30, 101, n),
    "internal_marks": np.random.randint(10, 101, n),
    "assignment_completion": np.random.randint(20, 101, n),
    "previous_semester_gpa": np.round(np.random.uniform(2.0, 10.0, n), 2),
    "study_hours": np.round(np.random.uniform(0, 8, n), 1),
    "class_participation": np.random.randint(1, 11, n)
}

df = pd.DataFrame(data)

# Calculate academic risk score
risk_score = (
    (100 - df["attendance"]) * 0.25
    + (100 - df["internal_marks"]) * 0.25
    + (100 - df["assignment_completion"]) * 0.15
    + (10 - df["previous_semester_gpa"]) * 3
    + (8 - df["study_hours"]) * 1.5
    + (10 - df["class_participation"]) * 1
)

# Add small randomness
risk_score += np.random.normal(0, 5, n)

# Create risk categories
df["academic_risk"] = np.where(
    risk_score >= 65,
    "High Risk",
    "Low Risk"
)

# Save dataset
df.to_csv("student_data.csv", index=False)

print("Student dataset created successfully!")
print(df.head())
print("\nRisk Distribution:")
print(df["academic_risk"].value_counts())
