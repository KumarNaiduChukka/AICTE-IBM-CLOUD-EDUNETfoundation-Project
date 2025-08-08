# Network Intrusion Detection

This project is a Streamlit web application for detecting network intrusions. It uses a pre-trained machine learning model to classify network traffic as either normal or an intrusion.

## Features

- **User-friendly web interface:** The application provides a simple web interface for users to input network traffic data.
- **Real-time predictions:** The application uses a trained model to make real-time predictions on the user's input.
- **Easy to use:** The application is easy to use and requires no special knowledge of machine learning.

## How to run the application

1. **Install the dependencies:**

   You will need to install the following Python libraries:
   - `streamlit`
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `snapml`
   - `autoai-libs`

   You can install them using pip:
   ```bash
   pip install streamlit pandas numpy scikit-learn snapml autoai-libs
   ```

2. **Run the Streamlit application:**

   Once you have installed the dependencies, you can run the application using the following command:
   ```bash
   streamlit run Stramlit.py
   ```

   This will start the Streamlit development server and open the application in your web browser.

## How it works

The application uses a machine learning model to classify network traffic. The model is a `SnapRandomForestClassifier` that has been trained on the [NSL-KDD dataset](https://www.unb.ca/cic/datasets/nsl.html).

The application does the following:
1. When the application is first started, it checks if a trained model file (`model.pkl`) exists.
2. If the model file does not exist, the application trains a new model using the `Train_data.csv` file and saves it as `model.pkl`.
3. If the model file exists, the application loads the pre-trained model.
4. The application displays a user interface with input fields for the user to enter network traffic data.
5. When the user clicks the "Predict" button, the application uses the loaded model to make a prediction on the user's input.
6. The application displays the prediction to the user.
