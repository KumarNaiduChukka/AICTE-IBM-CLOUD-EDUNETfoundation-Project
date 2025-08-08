import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pipeline import get_pipeline

# Define the model path
MODEL_PATH = "model.pkl"
DATA_PATH = "Train_data.csv"

def train_model():
    """
    This function trains the model and saves it to a file.

    :return: The trained model.
    :rtype: sklearn.pipeline.Pipeline
    """
    st.write("Training model...")
    # Load the training data
    if not os.path.exists(DATA_PATH):
        st.error(f"Training data file not found: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)

    # The target column is 'class'
    X = df.drop("class", axis=1)
    y = df["class"]

    # Get the pipeline
    pipeline = get_pipeline()

    # Train the model
    pipeline.fit(X.values, y.values)

    # Save the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    st.write("Model training complete.")
    return pipeline

def load_model():
    """
    This function loads the model from a file.

    :return: The loaded model.
    :rtype: sklearn.pipeline.Pipeline
    """
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

def get_user_input():
    """
    This function gets the user input for the features from the sidebar.

    :return: A DataFrame containing the user's input.
    :rtype: pd.DataFrame
    """
    st.sidebar.header("User Input Features")

    # Get the column names from the training data
    if not os.path.exists(DATA_PATH):
        st.error(f"Training data file not found: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    features = df.drop("class", axis=1).columns

    user_input = {}

    # Create input fields for each feature
    for feature in features:
        if df[feature].dtype == "object":
            # Create a selectbox for categorical features
            unique_values = df[feature].unique()
            user_input[feature] = st.sidebar.selectbox(feature, unique_values)
        else:
            # Create a number input for numerical features
            user_input[feature] = st.sidebar.number_input(feature, value=float(df[feature].mean()))

    # Create a DataFrame from the user's input
    data = pd.DataFrame(user_input, index=[0])
    return data

# Set the title of the app
st.title("Network Intrusion Detection")

# Load or train the model
if os.path.exists(MODEL_PATH):
    # st.write("Loading existing model.")
    model = load_model()
else:
    st.write("No existing model found. Training a new one.")
    model = train_model()

st.write("Model loaded successfully.")

# Get the user's input
user_input_df = get_user_input()

# Display the user's input
st.subheader("User Input parameters")
st.write(user_input_df)

# Add a predict button
if st.sidebar.button("Predict"):
    # Make a prediction
    prediction = model.predict(user_input_df.values)
    prediction_probability = model.predict_proba(user_input_df.values)

    # Display the prediction
    st.subheader("Prediction")
    st.write(prediction[0])

    # Display the prediction probability
    st.subheader("Prediction Probability")
    st.write(prediction_probability)
