import os
import pickle
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.prediction import MakePredictions
from src.model import FraudDetectionModel
from src.preprocessing import DataPreprocessor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI()

# Define directories
MODEL_DIR = './models'
DATA_DIR = './data'
MODEL_PATH = os.path.join(MODEL_DIR, 'model1.pkl') 

# Initialize instances for preprocessing, model, and predictions
preprocessor = DataPreprocessor(file_path=os.path.join(DATA_DIR, 'fraudTest.csv'))
model_instance = FraudDetectionModel(DATA_DIR, MODEL_DIR)
prediction_instance = MakePredictions(model_dir=MODEL_DIR, scaler_dir="")

# Add CORS middleware
origins = [
    "https://pipeline-frontend-summative.onrender.com",  # Frontend URL
    "https://pipeline-summative-1.onrender.com",         # API URL
    "http://localhost:8000",
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5000",  # Local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Updated with deployed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the input schema for prediction
class PredictionInput(BaseModel):
    category: int = Field(..., ge=0, le=50, description="Category must be an integer")
    amt: float
    gender: int = Field(..., ge=0, le=1, description="Gender must be 0 (female) or 1 (male)")
    city_pop: int
    trans_day: int
    trans_month: int
    trans_year: int
    trans_hour: int
    trans_minute: int
    lat: float
    long: float


# Helper function to load the pre-trained model
def load_model():
    """Load the pre-trained model."""
    global model_instance
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as model_file:
            model_instance.model = pickle.load(model_file)
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.on_event("startup")
def startup_event():
    """Load the model when the app starts."""
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")
    
    
@app.get("/get_dataset")
def get_dataset():
    # Specify the path to the dataset (assuming it's inside a folder named 'data')
    dataset_path = os.path.join(os.getcwd(), 'data', 'fraudTest.csv')
    
    # Check if the file exists before attempting to read it
    if not os.path.exists(dataset_path):
        return JSONResponse(content={"error": "Dataset not found!"}, status_code=404)

    # Load your dataset
    df = pd.read_csv(dataset_path)

    # Filter only numerical columns for correlation
    numeric_df = df.select_dtypes(include=['number'])

    # Get the correlation matrix of the numeric columns
    correlation_matrix = numeric_df.corr()

    # Convert the correlation matrix to a list of lists for sending as JSON
    correlation_data = correlation_matrix.values.tolist()
    features = correlation_matrix.columns.tolist()

    # Prepare the transaction data (or relevant columns for trend analysis and histogram)
    transactions = df[['trans_date_trans_time', 'amt']].to_dict(orient='records')

    # Return the correlation data, feature names, and transaction data in the response
    return JSONResponse(content={"correlation_data": correlation_data, "features": features, "transactions": transactions})



# Prediction endpoint
@app.post("/predict/")  # Keep this as is
async def make_prediction(input: PredictionInput):
    """Make a prediction using the pre-trained model."""
    try:
        # Prepare the input data for prediction
        input_data = input.model_dump()  # Use model_dump instead of dict
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model_instance.model.predict(input_df)
        prediction_label = "fraud" if prediction[0] == 1 else "not fraud"
        
        return {"prediction": prediction_label}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/")  # Retraining the model and saving confusion matrix
async def retrain(file: UploadFile):
    """
    Retrain the model with a new dataset uploaded by the user.
    Each retrained model is saved with a unique filename, and the accuracy is returned.
    """
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    
    # Save the uploaded file
    upload_folder = "uploaded_data"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Preprocess and retrain the model
    try:
        drop_columns = [
            'Unnamed: 0', 'cc_num', 'merch_lat', 'merch_long', 'zip', 
            'first', 'last', 'unix_time', 'street', 'job', 'dob', 
            'city', 'state', 'trans_num', 'merchant'
        ]
        categorical_columns = ["gender", "category"]
        target_column = "is_fraud"
        
        # Use the correct method for preprocessing
        X_train, X_test, y_train, y_test = preprocessor.preprocess(drop_columns, categorical_columns, target_column)

        # Train the model
        model_instance.train_model(X_train, y_train)

        # Evaluate the model
        accuracy = model_instance.model.score(X_test, y_test)

        # Save confusion matrix plot
        y_pred = model_instance.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        cm_path = 'confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()

        # Save the retrained model with a unique filename
        new_model_filename = f"model_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.pkl"
        new_model_path = os.path.join(MODEL_DIR, new_model_filename)
        with open(new_model_path, 'wb') as model_file:
            pickle.dump(model_instance.model, model_file)

        return {
            "message": "Model retrained successfully.",
            "accuracy": f"{accuracy:.2f}",
            "model_file": new_model_filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {e}")

@app.get("/confusion-matrix/")
def get_confusion_matrix():
    try:
        # Use the correct file path
        matrix_path = 'confusion_matrix.png'  # This should match the path where it's saved
        if not os.path.exists(matrix_path):
            raise HTTPException(status_code=404, detail="Confusion matrix image not found.")
        return FileResponse(matrix_path, media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving confusion matrix: {e}")


# Endpoint to get the training history plot (if you are saving training history plots)
@app.get("/training-history/")  
def get_training_history():
    try:
        return FileResponse('training_history.png')  # Adjust the path if necessary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving training history: {e}")
