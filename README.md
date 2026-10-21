# Medical Cost Estimator

A machine learning project that predicts estimated medical insurance charges based on personal and lifestyle-related features.

## Features

- Predicts estimated annual medical insurance charges.
- Accepts custom user input through the terminal.
- Uses data preprocessing for categorical features.
- Trains a Linear Regression model.
- Evaluates performance using Mean Absolute Error (MAE) and R² score.
- Visualizes actual versus predicted insurance charges.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Git and GitHub

## Input Features

The model uses the following inputs:

- Age
- Sex
- Body Mass Index (BMI)
- Smoking status
- Region

## Dataset

The dataset file `insurance.csv` is not included in this repository because redistribution permission has not been established.

To run this project, obtain an authorized copy of the dataset and place it in the project root directory with the filename `insurance.csv`.

The dataset should contain the columns required by the program, including `age`, `sex`, `bmi`, `smoker`, `region`, and `charges`.

## Installation and Setup

1. Clone or download this repository.
2. Install Python 3.
3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Place an authorized copy of `insurance.csv` in the project folder.
5. Run the program:

   ```bash
   python medical_cost_estimator.py
   ```

6. Enter the requested details when prompted to receive a prediction.

## Model Evaluation

The current Linear Regression model achieved the following results on the existing dataset:

- **Mean Absolute Error (MAE):** 4222.91
- **R² Score:** 0.7801

These results are from the current version and may change after further improvements.

## Disclaimer

This project is intended for educational purposes only. Its predictions are estimates and should not be treated as actual insurance quotes or professional financial advice.

## Future Improvements

- Compare Linear Regression with Random Forest Regression.
- Improve model evaluation and visualizations.
- Develop a web interface using Streamlit.