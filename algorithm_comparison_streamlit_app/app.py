# Basic App Setup
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Algorithm Comparison", layout="wide")
st.title("📊 ML Algorithm Comparison Dashboard")
st.write("Compare multiple machine learning algorithms on HR Attrition data")

# Load Dataset
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "HR-Employee-Attrition_updated.csv")
    return pd.read_csv(file_path)


df = load_data()
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Feature & Target Split
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

test_size = st.slider("Test Size (%)", 10, 40, 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

# Model Selection (Interactive 🔥)
st.subheader("🤖 Select Algorithms")

algorithms = st.multiselect(
    "Choose algorithms to train:",
    ["Logistic Regression", "Decision Tree", "SVM", "Random Forest"],
    default=["Logistic Regression", "Decision Tree"]
)

results = {}

# Train Models
if st.button("🚀 Train Models"):
    if "Logistic Regression" in algorithms:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        results["Logistic Regression"] = accuracy_score(
            y_test, lr.predict(X_test)
        ) * 100

    if "Decision Tree" in algorithms:
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        results["Decision Tree"] = accuracy_score(
            y_test, dt.predict(X_test)
        ) * 100

    if "SVM" in algorithms:
        svm = SVC()
        svm.fit(X_train, y_train)
        results["SVM"] = accuracy_score(
            y_test, svm.predict(X_test)
        ) * 100

    if "Random Forest" in algorithms:
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        results["Random Forest"] = accuracy_score(
            y_test, rf.predict(X_test)
        ) * 100

# Display Accuracy Results
if results:
    st.subheader("📈 Model Accuracy Comparison")

    result_df = pd.DataFrame(
        results.items(), columns=["Algorithm", "Accuracy (%)"]
    )
    st.table(result_df)

    fig, ax = plt.subplots()
    ax.bar(result_df["Algorithm"], result_df["Accuracy (%)"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Algorithm Comparison")
    st.pyplot(fig)
