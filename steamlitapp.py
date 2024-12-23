import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Linear Regression App")

# Step 1: Upload dataset
st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded dataset:")
    st.write(data.head())

    # Step 2: Select features and target
    st.header("Select Features and Target")
    features = st.multiselect("Select feature column(s) for X:", options=data.columns)
    target = st.selectbox("Select the target column for y:", options=data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Step 3: Split data into training and testing sets
        st.header("Train-Test Split")
        test_size = st.slider("Test size (percentage)", min_value=10, max_value=50, value=20, step=5) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Step 4: Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 5: Make predictions
        y_pred = model.predict(X_test)

        # Step 6: Display results
        st.header("Model Performance")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

        # Step 7: Plot predictions vs actual
        st.header("Prediction vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
