
import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Cost Estimator",
    page_icon="♡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# PATHS AND COLORS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "insurance.csv"
GRAPH_PATH = BASE_DIR / "images" / "model_comparison.png"

TEAL = "#0D514B"
DARK_TEAL = "#093D38"

# --------------------------------------------------
# THEME TOGGLE
# --------------------------------------------------

if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = False


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

def apply_theme(dark):
    if dark:
        background = "#111817"
        surface = "#1B2422"
        text = "#F0F5F3"
        secondary = "#B7C6C1"
        border = "#354641"
        input_bg = "#202B28"
    else:
        background = "#FFFFFF"
        surface = "#FFFFFF"
        text = "#202927"
        secondary = "#65716D"
        border = "#E0E7E4"
        input_bg = "#FFFFFF"

    st.markdown(
        f"""
        <style>
        :root {{
            --medical-teal: {TEAL};
        }}

        .stApp {{
            background-color: {background};
            color: {text};
        }}

        [data-testid="stHeader"] {{
            background-color: {background};
        }}

        [data-testid="stMainBlockContainer"] {{
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        label {{
            color: {text};
        }}

        .app-heading {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 0;
        }}

        .heart-logo {{
            width: 34px;
            height: 34px;
            flex-shrink: 0;
        }}

        .app-title {{
            color: {text};
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.25;
        }}

        .app-subtitle {{
            color: {secondary};
            font-size: 0.96rem;
            margin-top: 6px;
            margin-bottom: 28px;
        }}

        .section-heading {{
            color: {text};
            font-size: 1.12rem;
            font-weight: 650;
            margin-top: 8px;
            margin-bottom: 12px;
        }}

        div[data-testid="stForm"] {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 20px;
        }}

        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background-color: {input_bg};
            color: {text};
        }}

        /* Prediction button: white text on teal background */
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] > button[kind="primary"] {{
            background-color: #0D514B !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            border: 1px solid #0D514B !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            min-height: 48px;
            transition: background-color 0.2s ease;
        }}

        div[data-testid="stFormSubmitButton"] button *,
        div[data-testid="stButton"] > button[kind="primary"] * {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}

        div[data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stButton"] > button[kind="primary"]:hover {{
            background-color: #093D38 !important;
            border-color: #093D38 !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}

        /* Theme toggle button */
        div[data-testid="stButton"] > button[kind="secondary"] {{
            border: 1px solid {border};
            color: {text};
            background-color: {surface};
            border-radius: 10px;
        }}

        div[data-testid="stMetric"] {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 16px;
        }}

        div[data-testid="stMetricLabel"] {{
            color: {secondary};
        }}

        div[data-testid="stMetricValue"] {{
            color: {TEAL} !important;
        }}

        .prediction-label {{
            color: {secondary};
            font-size: 0.95rem;
            margin-bottom: 8px;
        }}

        .prediction-amount {{
            color: {TEAL};
            font-size: 2.25rem;
            font-weight: 750;
            letter-spacing: -0.8px;
            line-height: 1.25;
            margin-bottom: 8px;
        }}

        .prediction-model {{
            color: {secondary};
            font-size: 0.82rem;
        }}

        hr {{
            border-color: {border};
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {border};
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme(st.session_state.dark_theme)


# --------------------------------------------------
# MODEL CREATION
# --------------------------------------------------

def create_model(regressor):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["sex", "smoker", "region"],
            )
        ],
        remainder="passthrough",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


@st.cache_resource
def train_models():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "insurance.csv was not found. "
            "Place an authorized copy in the project folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "age",
        "sex",
        "bmi",
        "smoker",
        "region",
        "charges",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The dataset must contain these columns: "
            + ", ".join(sorted(required_columns))
        )

    if df[list(required_columns)].isnull().any().any():
        raise ValueError(
            "The dataset contains missing values. "
            "Please clean the dataset before training."
        )

    X = df[["age", "sex", "bmi", "smoker", "region"]]
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": create_model(
            LinearRegression()
        ),
        "Random Forest": create_model(
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1,
            )
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[name] = {
            "MAE": mean_absolute_error(
                y_test, predictions
            ),
            "RMSE": mean_squared_error(
                y_test, predictions
            ) ** 0.5,
            "R2": r2_score(
                y_test, predictions
            ),
        }

    best_name = min(
        results,
        key=lambda name: results[name]["MAE"],
    )

    if best_name == "Linear Regression":
        final_regressor = LinearRegression()
    else:
        final_regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )

    # Train the selected model using all available records
    best_model = create_model(final_regressor)
    best_model.fit(X, y)

    return df, results, best_name, best_model


# --------------------------------------------------
# HEADER AND THEME BUTTON
# --------------------------------------------------

header_left, header_right = st.columns(
    [5, 1],
    vertical_alignment="center",
)

with header_left:
    st.markdown(
        f"""
        <div class="app-heading">
            <svg class="heart-logo"
                 viewBox="0 0 32 32"
                 fill="none"
                 xmlns="http://www.w3.org/2000/svg">
                <path
                    d="M16 27.2
                       C13.8 25.1 4 18.3 4 11.4
                       C4 7.5 6.8 5 10.1 5
                       C12.6 5 14.5 6.5 16 8.6
                       C17.5 6.5 19.4 5 21.9 5
                       C25.2 5 28 7.5 28 11.4
                       C28 18.3 18.2 25.1 16 27.2Z"
                    stroke="{TEAL}"
                    stroke-width="2.2"
                    stroke-linecap="round"
                    stroke-linejoin="round"/>
                <path
                    d="M7 16H12L14.5 11.5L18 20L20.5 15.5H25"
                    stroke="{TEAL}"
                    stroke-width="1.8"
                    stroke-linecap="round"
                    stroke-linejoin="round"/>
            </svg>
            <h1 class="app-title">Medical Cost Estimator</h1>
        </div>
        <div class="app-subtitle">
            Machine-learning-powered cost prediction
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    theme_label = (
        "☀️" if st.session_state.dark_theme else "🌙"
    )

    if st.button(
        theme_label,
        help="Switch between light and dark themes",
        use_container_width=True,
    ):
        st.session_state.dark_theme = (
            not st.session_state.dark_theme
        )
        st.rerun()


# --------------------------------------------------
# LOAD AND TRAIN MODELS
# --------------------------------------------------

try:
    with st.spinner("Preparing the machine learning models..."):
        df, results, best_name, best_model = train_models()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error(f"Unable to prepare the models: {error}")
    st.stop()


# --------------------------------------------------
# DATASET AND MODEL INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Dataset Records", f"{len(df):,}")
col2.metric("Models Compared", len(results))
col3.metric("Selected Model", best_name)

st.divider()


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">Enter your details</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=20,
            step=1,
        )

        sex = st.selectbox(
            "Sex",
            ["male", "female"],
            format_func=str.capitalize,
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=23.2,
            step=0.1,
        )

    with col2:
        smoker_choice = st.selectbox(
            "Smoker",
            ["no", "yes"],
            format_func=str.capitalize,
        )

        region = st.selectbox(
            "Region",
            [
                "southeast",
                "northeast",
                "northwest",
                "southwest",
            ],
            format_func=str.capitalize,
        )

    submitted = st.form_submit_button(
        "Predict Insurance Charges",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------

if submitted:
    new_person = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "smoker": smoker_choice,
                "region": region,
            }
        ]
    )

    predicted_charge = best_model.predict(new_person)[0]

    st.divider()

    st.markdown(
        f"""
        <div class="prediction-label">
            Estimated annual charges
        </div>
        <div class="prediction-amount">
            ${predicted_charge:,.2f}
        </div>
        <div class="prediction-model">
            Calculated using {best_name}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View the details you entered"):
        detail1, detail2 = st.columns(2)

        detail1.write(f"**Age:** {age} years")
        detail1.write(f"**Sex:** {sex.capitalize()}")
        detail1.write(f"**BMI:** {bmi:.1f}")

        detail2.write(
            f"**Smoker:** {smoker_choice.capitalize()}"
        )
        detail2.write(
            f"**Region:** {region.capitalize()}"
        )


# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">'
    "Model Performance Comparison</div>",
    unsafe_allow_html=True,
)

comparison_df = pd.DataFrame(results).T
comparison_df.index.name = "Model"

st.dataframe(
    comparison_df.style.format(
        {
            "MAE": "${:,.2f}",
            "RMSE": "${:,.2f}",
            "R2": "{:.4f}",
        }
    ),
    use_container_width=True,
)

st.caption(
    "Metrics are calculated on a held-out test set. "
    "The selected model is then retrained on the full dataset."
)


# --------------------------------------------------
# SMALLER MODEL COMPARISON GRAPH
# --------------------------------------------------

if GRAPH_PATH.exists():
    st.divider()

    st.markdown(
        '<div class="section-heading">'
        "Actual vs Predicted Charges</div>",
        unsafe_allow_html=True,
    )

    st.image(
        str(GRAPH_PATH),
        caption="Actual and predicted medical charges",
        width=600,
    )
