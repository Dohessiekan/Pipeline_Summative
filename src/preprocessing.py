import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from src.model import FraudDetectionModel  # Import the model class
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.df = None
        self.scaler = StandardScaler()  # We will fit the scaler dynamically
        self.label_encoders = {}

    def load_data(self, file_path=None):
        """Loads the data from a CSV file"""
        if file_path:
            self.file_path = file_path  # Update file path if provided
        self.df = pd.read_csv(self.file_path)
        return self.df

    def check_missing_values(self):
        """Checks for missing values in the dataset"""
        return self.df.isnull().sum()

    def drop_missing_values(self):
        """Drops rows with missing values"""
        self.df = self.df.dropna()
        return self.df

    def describe_data(self):
        """Returns descriptive statistics of the data"""
        return self.df.describe()

    def get_data_info(self):
        """Returns information about the dataset"""
        return self.df.info()

    def drop_unnecessary_columns(self, columns):
        """Drops unnecessary columns from the dataset"""
        self.df = self.df.drop(columns=columns)
        return self.df

    def encode_categorical_columns(self, columns):
        """Encodes categorical columns using Label Encoding"""
        for col in columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        return self.df

    def detect_outliers(self):
        """Detects outliers using the IQR method"""
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((self.df < (Q1 - 1.5 * IQR)) | (self.df > (Q3 + 1.5 * IQR))).sum()
        return outliers

    def scale_features(self, X):
        """Scales the features using StandardScaler"""
        return self.scaler.fit_transform(X)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Splits the data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def preprocess(self, drop_columns, categorical_columns, target_column, test_size=0.2, random_state=42, file_path=None):
        """Main preprocessing pipeline"""
        self.load_data(file_path)  # Load new data if available
        self.drop_missing_values()
        self.drop_unnecessary_columns(drop_columns)
        
        # Extract datetime features
        self.df['trans_date_trans_time'] = pd.to_datetime(self.df['trans_date_trans_time'])
        self.df['trans_day'] = self.df['trans_date_trans_time'].dt.day
        self.df['trans_month'] = self.df['trans_date_trans_time'].dt.month
        self.df['trans_year'] = self.df['trans_date_trans_time'].dt.year
        self.df['trans_hour'] = self.df['trans_date_trans_time'].dt.hour
        self.df['trans_minute'] = self.df['trans_date_trans_time'].dt.minute
        self.df.drop(columns=['trans_date_trans_time'], inplace=True)

        # Encode categorical columns
        self.encode_categorical_columns(categorical_columns)

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        X = self.scale_features(X)
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Set file path here (could be updated dynamically if new data is uploaded)
    file_path = "../data/fraudTest.csv"
    
    preprocessor = DataPreprocessor(file_path)

    # Define columns to drop, categorical columns, and target column
    drop_columns = ['Unnamed: 0', 'cc_num', 'merch_lat', 'merch_long', 'zip', 'first', 'last', 'unix_time', 'street', 'job', 'dob', 'city', 'state', 'trans_num', 'merchant']
    categorical_columns = ["category", "gender"]
    target_column = "is_fraud"

    # Preprocessing the data
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        drop_columns=drop_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        test_size=0.2,
        random_state=42,
        file_path=file_path  # Optional: Set dynamically if the file path changes
    )

    # Initialize and load the model class
    model_instance = FraudDetectionModel(model_dir="../data/model")

    # Train the model using the preprocessed data
    model_instance.train_model(X_train, y_train)

    # Save the retrained model and scaler
    model_instance.save_model()
    joblib.dump(preprocessor.scaler, "../data/scaler/scaler.pkl")
