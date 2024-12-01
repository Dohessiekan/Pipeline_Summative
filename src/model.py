import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
from fastapi import HTTPException


class FraudDetectionModel:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def load_model(self):
        """
        Load the saved model from the specified path.
        """
        try:
            with open(self.model_path, 'rb') as model_file:
                self.model = pk.load(model_file)
            return self.model
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Model file not found.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    def load_data(self):
        """
        Load the dataset from CSV files.
        """
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=['is_fraud'])  # Features
        y = data['is_fraud']  # Target
        self.X_train, self.X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]  # Train-test split
        self.y_train, self.y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

    def train_model(self, X=None, y=None):
        """
        Train the RandomForest model on the provided data or class attributes.
        """
        if X is not None and y is not None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
        elif self.X_train is not None and self.y_train is not None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
        else:
            raise ValueError("No training data provided or loaded.")

    def retrain_model(self, new_data_path):
        """
        Retrain the RandomForest model with a new dataset.
        """
        # Load new data
        new_data = pd.read_csv(new_data_path)
        X = new_data.drop(columns=['is_fraud'])  # Features
        y = new_data['is_fraud']  # Target

        # Retrain model
        self.train_model(X, y)

        # Evaluate model on new data
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return self.model, accuracy

    def make_predictions(self):
        """
        Make predictions on the test data.
        """
        if not self.model:
            raise HTTPException(status_code=500, detail="Model not loaded.")
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate_model(self):
        """
        Evaluate the model on test data.
        """
        if self.y_pred is None:
            raise HTTPException(status_code=500, detail="No predictions made yet.")
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        matrix = confusion_matrix(self.y_test, self.y_pred)
        return accuracy, report, matrix

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix using seaborn.
        """
        if self.y_pred is None:
            raise HTTPException(status_code=500, detail="No predictions made yet.")
        matrix = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def save_model(self):
        """
        Save the trained model to a file.
        """
        try:
            model_filename = os.path.join(self.model_path, 'fraud_detection_model.pkl')
            with open(model_filename, 'wb') as model_file:
                pk.dump(self.model, model_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving model: {str(e)}")


if __name__ == "__main__":
    model_path = './models/fraud_detection_model.pkl'
    data_path = './data/fraudTest.csv'

    fraud_model = FraudDetectionModel(model_path, data_path)

    # Load initial data and model
    fraud_model.load_data()
    fraud_model.load_model()

    # Retrain model with new data
    new_data_path = './data/fraudTest.csv'
    model, accuracy = fraud_model.retrain_model(new_data_path)
    print(f"Retrained Model Accuracy: {accuracy}")

    # Make predictions
    fraud_model.make_predictions()

    # Evaluate model
    accuracy, report, matrix = fraud_model.evaluate_model()
    print(f'Accuracy: {accuracy}')
    print('Classification Report:\n', report)
    print('Confusion Matrix:\n', matrix)

    # Plot confusion matrix
    fraud_model.plot_confusion_matrix()

    # Save the trained model
    fraud_model.save_model()
