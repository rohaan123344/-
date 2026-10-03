import os
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Aicademy 360 | Iris Flower Detective",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f7fbff 0%,
            #eef6ff 50%,
            #f8f2ff 100%
        );
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #071a35 0%,
            #0b2d59 55%,
            #102f52 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Headings */
    h1 {
        color: #075985;
        font-weight: 800;
    }

    h2 {
        color: #0f4c81;
    }

    h3 {
        color: #164e63;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #dbeafe;
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0px 5px 15px rgba(30, 64, 175, 0.08);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 700;
    }

    /* Primary prediction button */
    button[kind="primaryFormSubmit"] {
        background: linear-gradient(
            90deg,
            #059669,
            #0284c7
        ) !important;
        color: white !important;
        border-radius: 15px !important;
        font-weight: 800 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATASET
# =========================================================

if not os.path.exists("iris.csv"):
    st.error(
        "⚠️ iris.csv was not found. "
        "Please put iris.csv in the same folder as app1.py."
    )
    st.stop()


df = pd.read_csv("iris.csv")


# =========================================================
# CHECK DATASET COLUMNS
# =========================================================

feature_columns = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

required_columns = feature_columns + ["species"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "Your iris.csv file is missing these columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

X = df[feature_columns]

y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)


model.fit(
    X_train,
    y_train,
)


test_predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    test_predictions
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=model.classes_,
)


cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual {name.title()}"
        for name in model.classes_
    ],
    columns=[
        f"Predicted {name.title()}"
        for name in model.classes_
    ],
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🌸 Iris Detective")

    st.caption("Aicademy 360")

    st.divider()

    st.subheader("🚀 Project Information")

    st.write("🤖 **Model:** Random Forest")

    st.write(
        "📚 **Type:** Supervised Classification"
    )

    st.write(
        "🌸 **Dataset:** Iris Flowers"
    )

    st.write(
        "🌳 **Decision Trees:** 100"
    )

    st.write(
        f"🎯 **Accuracy:** {accuracy:.0%}"
    )

    st.divider()

    st.subheader("🎓 Assignment")

    st.success(
        "Machine Learning Assignment #1"
    )

    st.subheader("🛠️ Skills")

    st.write("🐍 Python")
    st.write("🐼 Pandas")
    st.write("🤖 Scikit-Learn")
    st.write("🌳 Random Forest")
    st.write("📊 Model Evaluation")
    st.write("🎈 Streamlit")

    st.divider()

    st.caption(
        "Learn • Practice • Build"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🌸 Iris Flower Detective")

st.subheader(
    "Machine Learning Flower Classification System"
)

st.write(
    "Enter four flower measurements and let a trained "
    "Random Forest model predict whether the flower is "
    "Setosa, Versicolor, or Virginica."
)

st.caption(
    f"Aicademy 360 • Machine Learning Assignment #1 • "
    f"{len(df)} flowers • Test Accuracy: {accuracy:.0%}"
)


# =========================================================
# INTRODUCTION
# =========================================================

st.info(
    "🌸 Choose four flower measurements and let the trained "
    "Random Forest model analyze the measurements and "
    "predict the flower species."
)


# =========================================================
# TOP METRICS
# =========================================================

metric_1, metric_2, metric_3, metric_4 = st.columns(4)


metric_1.metric(
    "🌸 Flowers",
    len(df),
)


metric_2.metric(
    "📐 Input Features",
    len(feature_columns),
)


metric_3.metric(
    "🌿 Species Classes",
    y.nunique(),
)


metric_4.metric(
    "🎯 Test Accuracy",
    f"{accuracy:.0%}",
)


st.divider()


# =========================================================
# TABS
# =========================================================

prediction_tab, performance_tab, data_tab, learn_tab = st.tabs(
    [
        "🌸 Predict a Flower",
        "📊 Model Performance",
        "🧾 Dataset Explorer",
        "🧠 Learn the Project",
    ]
)


# =========================================================
# PREDICTION TAB
# =========================================================

with prediction_tab:

    st.header("🌸 Predict an Iris Flower")

    st.write(
        "Adjust the four measurements below and let "
        "the machine-learning model identify the species."
    )

    st.info(
        "💡 The Random Forest model learned patterns from "
        "the training data. Your measurements are passed "
        "to the trained model to make a prediction."
    )

    with st.form(
        "flower_prediction_form"
    ):

        left_col, right_col = st.columns(2)

        with left_col:

            st.subheader(
                "📏 Sepal Measurements"
            )

            sepal_length = st.slider(
                "Sepal length (cm)",
                min_value=float(
                    df["sepal_length_cm"].min()
                ),
                max_value=float(
                    df["sepal_length_cm"].max()
                ),
                value=float(
                    df["sepal_length_cm"].mean()
                ),
                step=0.1,
            )

            sepal_width = st.slider(
                "Sepal width (cm)",
                min_value=float(
                    df["sepal_width_cm"].min()
                ),
                max_value=float(
                    df["sepal_width_cm"].max()
                ),
                value=float(
                    df["sepal_width_cm"].mean()
                ),
                step=0.1,
            )

        with right_col:

            st.subheader(
                "🌸 Petal Measurements"
            )

            petal_length = st.slider(
                "Petal length (cm)",
                min_value=float(
                    df["petal_length_cm"].min()
                ),
                max_value=float(
                    df["petal_length_cm"].max()
                ),
                value=float(
                    df["petal_length_cm"].mean()
                ),
                step=0.1,
            )

            petal_width = st.slider(
                "Petal width (cm)",
                min_value=float(
                    df["petal_width_cm"].min()
                ),
                max_value=float(
                    df["petal_width_cm"].max()
                ),
                value=float(
                    df["petal_width_cm"].mean()
                ),
                step=0.1,
            )

        predict_button = st.form_submit_button(
            "🔍 Predict Species",
            type="primary",
            use_container_width=True,
        )


    # =====================================================
    # INPUT DATA
    # =====================================================

    input_flower = pd.DataFrame(
        [
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
            ]
        ],
        columns=feature_columns,
    )


    with st.expander(
        "🔎 See the exact data sent to the model"
    ):

        st.dataframe(
            input_flower,
            use_container_width=True,
            hide_index=True,
        )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    if predict_button:

        with st.spinner(
            "🤖 Analyzing flower characteristics..."
        ):

            prediction = model.predict(
                input_flower
            )[0]

            probabilities = model.predict_proba(
                input_flower
            )[0]


        highest_probability = probabilities.max()


        st.balloons()


        st.success(
            f"🌸 Predicted Species: {prediction.title()}"
        )


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "🤖 Highest Model Score",
                f"{highest_probability:.1%}",
            )


        with result_col2:

            st.metric(
                "🌿 Predicted Species",
                prediction.title(),
            )


        # =================================================
        # SPECIES INFORMATION
        # =================================================

        species_info = {

            "setosa":
                "🌱 Setosa is one of the three Iris species "
                "included in this dataset.",

            "versicolor":
                "🌿 Versicolor is one of the three Iris "
                "species used in this project.",

            "virginica":
                "🌸 Virginica is one of the three Iris "
                "species used in this classification task.",
        }


        if prediction.lower() in species_info:

            st.info(
                species_info[
                    prediction.lower()
                ]
            )


        # =================================================
        # MODEL PROBABILITIES
        # =================================================

        probability_df = pd.DataFrame(
            {
                "Species": [
                    name.title()
                    for name in model.classes_
                ],
                "Probability": probabilities,
            }
        ).set_index(
            "Species"
        )


        st.subheader(
            "📊 Model Confidence by Species"
        )


        st.bar_chart(
            probability_df
        )


        st.caption(
            "The probability is the model's estimated "
            "score based on patterns learned from the "
            "training data."
        )


# =========================================================
# MODEL PERFORMANCE TAB
# =========================================================

with performance_tab:

    st.header(
        "📊 Model Performance"
    )

    st.write(
        "This section shows how the trained Random Forest "
        "model performed on unseen test data."
    )


    correct_predictions = int(
        (test_predictions == y_test).sum()
    )


    incorrect_predictions = int(
        (test_predictions != y_test).sum()
    )


    total_test_flowers = len(
        y_test
    )


    performance_col1, performance_col2, performance_col3 = (
        st.columns(3)
    )


    with performance_col1:

        st.metric(
            "✅ Correct Predictions",
            correct_predictions,
        )


    with performance_col2:

        st.metric(
            "❌ Incorrect Predictions",
            incorrect_predictions,
        )


    with performance_col3:

        st.metric(
            "🧪 Test Flowers",
            total_test_flowers,
        )


    st.divider()


    st.success(
        f"The model correctly classified "
        f"{correct_predictions} out of "
        f"{total_test_flowers} unseen test flowers, "
        f"giving {accuracy:.0%} accuracy."
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(
        "🧩 Confusion Matrix"
    )


    st.write(
        "Rows show the actual species. Columns show "
        "what the model predicted. Numbers on the "
        "diagonal represent correct predictions."
    )


    st.dataframe(
        cm_df,
        use_container_width=True,
    )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader(
        "📋 Precision, Recall and F1 Score"
    )


    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )


    report_df = (
        pd.DataFrame(report)
        .transpose()
        .round(2)
    )


    st.dataframe(
        report_df,
        use_container_width=True,
    )


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader(
        "🌳 Feature Importance"
    )


    st.write(
        "Feature importance shows how much each flower "
        "measurement contributed to the Random Forest's "
        "decisions."
    )


    importance_df = pd.DataFrame(
        {
            "Feature": [
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width",
            ],
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )


    st.dataframe(
        importance_df.round(3),
        use_container_width=True,
        hide_index=True,
    )


    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )


# =========================================================
# DATASET EXPLORER TAB
# =========================================================

with data_tab:

    st.header(
        "🧾 Explore the Iris Dataset"
    )


    st.write(
        "Use the controls below to explore the "
        "dataset used for this machine-learning project."
    )


    # =====================================================
    # DATASET METRICS
    # =====================================================

    data_col1, data_col2, data_col3, data_col4 = (
        st.columns(4)
    )


    with data_col1:

        st.metric(
            "📚 Total Records",
            len(df),
        )


    with data_col2:

        st.metric(
            "📏 Features",
            len(feature_columns),
        )


    with data_col3:

        st.metric(
            "🌿 Species",
            y.nunique(),
        )


    with data_col4:

        st.metric(
            "🔢 Missing Values",
            int(
                df.isnull().sum().sum()
            ),
        )


    st.divider()


    # =====================================================
    # SPECIES FILTER
    # =====================================================

    species_filter = st.selectbox(
        "🌸 Choose a species to view",
        [
            "All Species",
            "Setosa",
            "Versicolor",
            "Virginica",
        ],
    )


    if species_filter == "All Species":

        filtered_df = df

    else:

        filtered_df = df[
            df["species"]
            == species_filter.lower()
        ]


    show_table = st.checkbox(
        "📄 Show dataset table",
        value=True,
    )


    if show_table:

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )


    # =====================================================
    # SPECIES COUNTS
    # =====================================================

    st.subheader(
        "🌿 Species Counts"
    )


    species_counts = (
        df["species"]
        .value_counts()
    )


    st.bar_chart(
        species_counts
    )


    # =====================================================
    # NUMERICAL SUMMARY
    # =====================================================

    st.subheader(
        "📈 Numerical Summary"
    )


    st.dataframe(
        df[feature_columns]
        .describe()
        .round(2),
        use_container_width=True,
    )


    # =====================================================
    # DOWNLOAD DATASET
    # =====================================================

    st.subheader(
        "📥 Download Dataset"
    )


    csv_data = df.to_csv(
        index=False
    )


    st.download_button(
        label="📥 Download iris.csv",
        data=csv_data,
        file_name="iris_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# LEARN TAB
# =========================================================

with learn_tab:

    st.header(
        "🧠 Complete Machine Learning Workflow"
    )


    st.write(
        "This project demonstrates the basic workflow "
        "of a supervised machine-learning classification problem."
    )


    # =====================================================
    # WORKFLOW
    # =====================================================

    st.subheader(
        "🔄 Project Workflow"
    )


    st.markdown(
        "### 1️⃣ CSV Data"
    )

    st.write(
        "We start with a labelled dataset containing Iris flowers."
    )


    st.markdown(
        "### 2️⃣ Features + Target"
    )

    st.write(
        "Four flower measurements are used as X and species is y."
    )


    st.markdown(
        "### 3️⃣ Train/Test Split"
    )

    st.write(
        "80% of the data is used for training and 20% for testing."
    )


    st.markdown(
        "### 4️⃣ Random Forest"
    )

    st.write(
        "100 Decision Trees learn patterns from the training data."
    )


    st.markdown(
        "### 5️⃣ Evaluation"
    )

    st.write(
        "The trained model is tested using unseen flowers."
    )


    st.markdown(
        "### 6️⃣ Streamlit Input"
    )

    st.write(
        "The user enters four flower measurements."
    )


    st.markdown(
        "### 7️⃣ Prediction"
    )

    st.write(
        "The model predicts one of three Iris species."
    )


    st.markdown(
        "### 8️⃣ Deployment"
    )

    st.write(
        "The Streamlit application can be uploaded and shared online."
    )


    st.divider()


    # =====================================================
    # SKILLS
    # =====================================================

    st.subheader(
        "🛠️ Skills Demonstrated"
    )


    skill_col1, skill_col2, skill_col3, skill_col4 = (
        st.columns(4)
    )


    with skill_col1:

        st.success("🐍 Python")
        st.success("🐼 Pandas")


    with skill_col2:

        st.success("🤖 Scikit-Learn")
        st.success("🌳 Random Forest")


    with skill_col3:

        st.success("📊 Classification")
        st.success("📈 Model Evaluation")


    with skill_col4:

        st.success("🎈 Streamlit")
        st.success("🚀 Deployment")


    st.divider()


    # =====================================================
    # CLASSIFICATION EXPLANATION
    # =====================================================

    st.subheader(
        "📚 What is Classification?"
    )


    st.info(
        "Classification is a type of machine learning "
        "where a model learns to predict a category or "
        "class. In this project, the categories are "
        "Setosa, Versicolor, and Virginica."
    )


    # =====================================================
    # ASSIGNMENT SUMMARY
    # =====================================================

    st.subheader(
        "🎓 Assignment Summary"
    )


    st.write(
        "**Project:** Iris Flower Classification"
    )


    st.write(
        "**Machine Learning Type:** Supervised Learning"
    )


    st.write(
        "**Algorithm:** Random Forest Classifier"
    )


    st.write(
        "**Dataset:** Iris Flower Dataset"
    )


    st.write(
        "**Input Features:** Sepal Length, Sepal Width, "
        "Petal Length, Petal Width"
    )


    st.write(
        "**Target:** Flower Species"
    )


    st.write(
        "**Train/Test Split:** 80% / 20%"
    )


    st.write(
        "**Number of Trees:** 100"
    )


# =========================================================
# CLEAN FOOTER
# =========================================================

st.divider()

st.caption(
    "🌸 Aicademy 360 | Iris Flower Detective | "
    "Machine Learning Assignment #1"
)

st.caption(
    "Built with Python • Pandas • Scikit-Learn • Streamlit"
)

st.caption(
    "👨‍💻 Built by Rohaan Asif"
)