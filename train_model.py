import pandas as pd

from sklearn.linear_model import LinearRegression

import pickle

# Load data
df = pd.read_csv("D:\\Yasir Msc\\2nd sem\\ML\\deploypr\\students.csv")

# Features
X = df[["Age",
        "Gender",
        "StudyHours",
        "Attendance"]]

# Target
y = df["Marks"]

# Train model
model = LinearRegression()

model.fit(X, y)

# Save model
pickle.dump(model,
            open("model.pkl", "wb"))

print("Model Saved")