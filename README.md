# Medical Cost Estimator

A machine learning project that predicts estimated annual medical insurance charges based on personal and lifestyle-related features. The project uses Python and Scikit-learn to train and evaluate regression models, and provides both a terminal-based interface and a web interface built with Streamlit.

## Features

- Predicts estimated annual medical insurance charges.
- Accepts custom user input through the terminal.
- Provides an interactive web interface using Streamlit.
- Preprocesses categorical features using one-hot encoding.
- Trains and compares Linear Regression and Random Forest Regression models.
- Evaluates model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.
- Visualizes actual versus predicted insurance charges.
- Selects the better-performing model based on Mean Absolute Error.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Git and GitHub

## Input Features

The model uses the following input features:

- **Age:** Age of the individual.
- **Sex:** Male or female.
- **BMI:** Body Mass Index.
- **Smoker:** Smoking status.
- **Region:** Residential region.

The target variable is `charges`, representing medical insurance charges.

## Dataset

The `insurance.csv` dataset is not included in this repository because permission to redistribute it has not been established.

To run the project, obtain an authorized copy of the dataset and place it in the project root directory with the filename `insurance.csv`.

The dataset must contain the following columns:

- `age`
- `sex`
- `bmi`
- `smoker`
- `region`
- `charges`

The dataset is excluded through `.gitignore` to prevent accidental uploading to GitHub.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the placeholders with your actual GitHub repository URL and repository name.

### 2. Create and activate a virtual environment (recommended)

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place your authorized `insurance.csv` file in the project's root directory, alongside `app.py` and `medical_cost_estimator.py`.

## Running the Project

### Option 1: Terminal-based application

Run:

```bash
python medical_cost_estimator.py
```

Enter the requested details when prompted to receive a medical insurance charge prediction.

### Option 2: Streamlit web application

Run:

```bash
streamlit run app.py
```

Streamlit will start a local web server and provide a URL, usually `http://localhost:8501`, where you can open the application in your browser.

Enter the requested personal and lifestyle-related details in the web interface to obtain an estimated insurance charge.

## Model Evaluation

The project compares Linear Regression and Random Forest Regression using a train-test split.

The following results were obtained from the current dataset and model configuration:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | $4,222.91 | $5,843.15 | 0.7801 |
| Random Forest Regression | $2,650.23 | $4,673.59 | 0.8593 |

Random Forest Regression performed better in this evaluation, achieving a lower MAE and RMSE and a higher R² score.

The results may vary if the dataset, preprocessing, model configuration, or train-test split changes.

A model comparison visualization is available in `images/model_comparison.png`.

## Project Structure

```text
Medical-Cost-Estimator/
├── .streamlit/
│   └── config.toml
├── images/
│   └── model_comparison.png
├── .gitignore
├── README.md
├── app.py
├── medical_cost_estimator.py
├── requirements.txt
└── insurance.csv  # Local file only; not committed to Git
```

The `insurance.csv` file is shown for reference only and should not be uploaded to the repository.

## Disclaimer

This project is intended for educational purposes only. Predictions are estimates based on the dataset and trained machine learning models. They are not actual insurance quotes, medical advice, or professional financial advice.

## Future Improvements

- Improve model evaluation using cross-validation.
- Experiment with additional regression algorithms and hyperparameter tuning.
- Add more informative model performance visualizations.
- Improve input validation and error handling.
- Deploy the Streamlit application for public access.

## Author

Developed as a machine learning project to explore data preprocessing, regression models, model evaluation, and interactive web application development.
