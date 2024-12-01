# Pipeline_Summative

FraudCardDetection
Overview
FraudCardDetection is a machine learning project aimed at detecting fraudulent credit card transactions. The project uses a dataset containing various transaction attributes to classify whether a transaction is fraudulent or not. The goal is to build a reliable model that can predict potential fraud in real-time.

Dataset
The project uses the Credit Card Fraud Detection dataset, which includes the following attributes:

index: Index of the record
trans_date_trans_time: Date and time of the transaction
cc_num: Credit card number
merchant: Merchant where the transaction occurred
category: Category of the transaction
amt: Transaction amount
first: First name of the cardholder
last: Last name of the cardholder
gender: Gender of the cardholder
street: Street address of the cardholder
city: City where the cardholder resides
state: State where the cardholder resides
zip: Postal code of the cardholder
lat: Latitude of the merchant location
long: Longitude of the merchant location
city_pop: Population of the city where the cardholder resides
job: Cardholder's job
dob: Date of birth of the cardholder
trans_num: Unique transaction number
unix_time: Unix timestamp of the transaction
merch_lat: Latitude of the merchant's location
merch_long: Longitude of the merchant's location
is_fraud: Target class (1 = fraudulent, 0 = non-fraudulent)
Project Structure
The project is organized as follows:

FraudCardDetection/
├── data/

│   ├── fraudTest.csv

│   └── fraudTrain.csv

│
├── notebooks/
│   └── FraudCardDetection.ipynb

│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── make_predictions.py

│
├── README.md

├── requirements.txt
└── models/
    ├── fraud_detection_model.pkl
    └── fraud_detection_model.tf


    
Installation

To set up the project, follow these steps:

Clone the repository:

git clone https://github.com/yourusername/FraudCardDetection.git

cd FraudCardDetection

Create a virtual environment and activate it:
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
Install the required dependencies:

pip install -r requirements.txt
 After that:
just lunch the main.py for the API and you will have all the endpoint


Models
This project uses the following models, which have been trained on the credit card fraud detection dataset:

fraud_detection_model.pkl: A serialized model trained using a classification algorithm (Using vanilla model and RandonForest).

Frontend
The frontend is built with FastAPI and provides the following features:

Data Loading: Load training and testing datasets.
Model Training: Train the machine learning model on the provided datasets.
Model Evaluation: Evaluate the model using metrics like accuracy, classification report, and confusion matrix.
Model Retraining: Retrain the model via an API endpoint.
Predictions: Make predictions on new transaction data to detect fraud.

Technologies
The project uses the following technologies:

Backend: Python, FastAPI, JavaScript
Machine Learning: scikit-learn, TensorFlow
Visualization: Matplotlib, Seaborn
