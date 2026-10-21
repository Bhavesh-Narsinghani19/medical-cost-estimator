
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
df = pd.read_csv("insurance.csv")

# 2. Select input features and output
X = df[["age", "sex", "bmi", "smoker", "region"]]
y = df["charges"]

# 3. Separate numerical and categorical columns
numerical_features = ["age", "bmi"]
categorical_features = ["sex", "smoker", "region"]

# 4. Convert categorical values into numerical values
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# 5. Create the machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Make predictions on test data
y_pred = model.predict(X_test)

# 9. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("===== MEDICAL COST ESTIMATOR =====")
print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# 10. Plot actual vs predicted charges
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Insurance Charges")
plt.ylabel("Predicted Insurance Charges")
plt.title("Actual vs Predicted Medical Costs")
plt.tight_layout()
plt.show()


# 11. Take custom input from the user

print("\n===== ENTER YOUR DETAILS =====")

# Get age
while True:
    try:
        age = int(input("Enter your age (18-100): "))
        if 18 <= age <= 100:
            break
        print("Please enter an age between 18 and 100.")
    except ValueError:
        print("Please enter a valid whole number.")

# Get sex
while True:
    sex = input("Enter sex (male/female): ").strip().lower()
    if sex in ["male", "female"]:
        break
    print("Please enter male or female.")

# Get BMI
while True:
    try:
        bmi = float(input("Enter BMI (10-60): "))
        if 10 <= bmi <= 60:
            break
        print("Please enter a BMI between 10 and 60.")
    except ValueError:
        print("Please enter a valid number.")

# Get smoking status
while True:
    smoker = input("Are you a smoker? (yes/no): ").strip().lower()
    if smoker in ["yes", "no"]:
        break
    print("Please enter yes or no.")

# Get region
regions = [
    "northeast",
    "northwest",
    "southeast",
    "southwest"
]

while True:
    region = input(
        "Enter region (northeast/northwest/"
        "southeast/southwest): "
    ).strip().lower()

    if region in regions:
        break
    print("Please enter a valid region.")

# Create a DataFrame with the user's details
new_person = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "smoker": smoker,
    "region": region
}])

# Predict insurance charges
predicted_charge = model.predict(new_person)[0]

# Display the result
print("\n===== PREDICTION RESULT =====")
print("Age:", age)
print("Sex:", sex.capitalize())
print("BMI:", bmi)
print("Smoker:", smoker.capitalize())
print("Region:", region.capitalize())
print(f"Predicted annual insurance charges: ${predicted_charge:,.2f}")

