
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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

# 5. Create both machine learning models

# Model 1: Linear Regression
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Model 2: Random Forest Regression
random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. Train both models
linear_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# 8. Make predictions using both models
linear_predictions = linear_model.predict(X_test)
rf_predictions = random_forest_model.predict(X_test)

# 9. Evaluate both models
models = {
    "Linear Regression": linear_predictions,
    "Random Forest": rf_predictions
}

print("\n===== MODEL COMPARISON =====")
print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

for name, predictions in models.items():
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

# Select the model with the lower MAE
best_model_name = min(
    models,
    key=lambda name: mean_absolute_error(y_test, models[name])
)

print("\nBest model based on MAE:", best_model_name)


# 10. Plot actual vs predicted charges for both models

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    linear_predictions,
    alpha=0.5,
    label="Linear Regression"
)

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.5,
    label="Random Forest"
)

# Reference line for perfect predictions
minimum = min(y_test.min(), linear_predictions.min(),
              rf_predictions.min())
maximum = max(y_test.max(), linear_predictions.max(),
              rf_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel("Actual Insurance Charges ($)")
plt.ylabel("Predicted Insurance Charges ($)")
plt.title("Actual vs Predicted: Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=300, bbox_inches="tight")
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
if best_model_name == "Linear Regression":
    selected_model = linear_model
else:
    selected_model = random_forest_model

predicted_charge = selected_model.predict(new_person)[0]

print("Model used:", best_model_name)

# Display the result
print("\n===== PREDICTION RESULT =====")
print("Age:", age)
print("Sex:", sex.capitalize())
print("BMI:", bmi)
print("Smoker:", smoker.capitalize())
print("Region:", region.capitalize())
print(f"Predicted annual insurance charges: ${predicted_charge:,.2f}")

