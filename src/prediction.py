import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import HTTPException
import joblib
from sklearn.metrics import confusion_matrix

class MakePredictions:
    '''
    Make predictions on the new data given to us
    '''
    def __init__(self, model_dir, scaler_dir):
        """
        Initializes a new instance of the MakePredictions class.
        """
        #scaler_path = os.path.join(scaler_dir, "scaler.pkl")
        self.model_dir = model_dir
        self.model = None
        #self.scaler = joblib.load(scaler_path)

    def load_model(self, model_number=None):
        """
        Load a specific model from the directory.
        """
        if model_number is None or model_number == -1:
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith('model_') and f.endswith('.pkl')]
            if not model_files:
                raise FileNotFoundError("No model files found in the directory.")
            model_numbers = [int(f.split('_')[1].split('.')[0]) for f in model_files]
            latest_model_number = max(model_numbers)
            model_filename = f'model_{latest_model_number}.pkl'
        else:
            model_filename = f'model_{model_number}.pkl'

        model_path = os.path.join(self.model_dir, model_filename)
        self.model = joblib.load(open(model_path, 'rb'))

    def load_data(self, data):
        """
        Load the new data into a pandas DataFrame.
        """
        columns = ['category', 'amt', 'gender', 'city_pop', 'trans_day',
                   'trans_month', 'trans_year', 'trans_hour', 'trans_minute']
        df = pd.DataFrame(data, columns=columns)
        return df

    def preprocess_data(self, df):
        """
        Preprocess the new data by scaling it using the previously fitted scaler.
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been loaded. Call 'load_scaler' first.")
        scaled_data = self.scaler.transform(df)
        return scaled_data

    def make_prediction(self, data):
        """
        Make predictions on the new data.
        """
        df = self.load_data(data)
        scaled_data = self.preprocess_data(df)
        predictions = self.model.predict(scaled_data)
        return predictions

# Function to plot the confusion matrix and save the image
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')

    # Ensure the static/images directory exists
    static_dir = "static/images"
    os.makedirs(static_dir, exist_ok=True)  # Create static/images directory if it doesn't exist

    # Save the plot in the static/images directory
    plot_filename = 'confusion_matrix.png'
    plot_path = os.path.join(static_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return plot_path

# FastAPI prediction endpoint function
def make_prediction(model, data):
    try:
        predictor = MakePredictions(model_dir="models", scaler_dir="data/scaler")
        predictor.load_model(model_number=1)
        prediction = predictor.make_prediction(data)
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
