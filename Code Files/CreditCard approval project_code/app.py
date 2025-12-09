from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np 
from typing import List
import pickle
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://127.0.0.1:8000", # Aapka FastAPI server
    "http://127.0.0.1:5500",
     "null" 
]

app = FastAPI(

    title = "CREDIT CARD APPROVAL",
    description = "Welcome in Credit Card Approval AI APP .... \nThis app is used Decision Tree Model",
    version="1.0.0"
              
             )


# CORS MIDDLEWARE BLOCK (Ab origins defined hai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


import os, sys, pickle

# ✅ Function to get correct path (for both .py & .exe)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # when running from .exe
    except Exception:
        base_path = os.path.abspath(".")  # when running normally
    return os.path.join(base_path, relative_path)

# ✅ Load model and scaler using correct paths
try:
    model_path = resource_path("Credit_Card_Approval.pkl")
    scaler_path = resource_path("scaler.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    print("MODEL AND SCALER LOADED SUCCESSFULLY ✅")

except Exception as e:
    print("Model or Scaler Load Error ❌:", e)
    model = None
    scaler = None




class CreditCardFeatures(BaseModel):
    Credit_Score: int
    Debt: float
    Income: float
    Loan_Amount: int
    Account_Age: int
    Years_Employed: int
    Num_Bank_Accounts: int
    Num_Credit_Cards: int
    Gender: int         # 1 = Male, 0 = Female
    Married: int        # 1 = Yes, 0 = No
    Dependents: int
    Credit_Cards_Limit: int
    City: int                 # 1=Islamabad, 2=Karachi, 3=Lahore, 4=Multan
    Education_Level: int      # 1=Bachelor, 2=High School, 3=Master, 4=PhD
    Employment_Type: int      # 1=Government, 2=Private, 3=Self-employed
    Housing_Status: int       # 1=Family, 2=Mortgage, 3=Own, 4=Rent
    Citizenship: int          # 1=By Birth, 2=By Other Means



def preprocess_input(features: CreditCardFeatures):
    # Input ko dict me convert karo
    data = features.dict()

    # One-hot columns ka structure banाओ
    df = pd.DataFrame({
        'Credit_Score': [data['Credit_Score']],
        'Debt': [data['Debt']],
        'Income': [data['Income']],
        'Loan_Amount': [data['Loan_Amount']],
        'Account_Age': [data['Account_Age']],
        'Years_Employed': [data['Years_Employed']],
        'Num_Bank_Accounts': [data['Num_Bank_Accounts']],
        'Num_Credit_Cards': [data['Num_Credit_Cards']],
        'Gender': [data['Gender']],
        'Married': [data['Married']],
        'Dependents': [data['Dependents']],
        'Credit_Cards_Limit': [data['Credit_Cards_Limit']],

        # One-hot encoded features:
        'City_Islamabad': [1 if data['City'] == 1 else 0],
        'City_Karachi': [1 if data['City'] == 2 else 0],
        'City_Lahore': [1 if data['City'] == 3 else 0],
        'City_Multan': [1 if data['City'] == 4 else 0],

        'Education_Level_Bachelor': [1 if data['Education_Level'] == 1 else 0],
        'Education_Level_High School': [1 if data['Education_Level'] == 2 else 0],
        'Education_Level_Master': [1 if data['Education_Level'] == 3 else 0],
        'Education_Level_PhD': [1 if data['Education_Level'] == 4 else 0],

        'Employment_Type_Government': [1 if data['Employment_Type'] == 1 else 0],
        'Employment_Type_Private': [1 if data['Employment_Type'] == 2 else 0],
        'Employment_Type_Self-employed': [1 if data['Employment_Type'] == 3 else 0],

        'Housing_Status_Family': [1 if data['Housing_Status'] == 1 else 0],
        'Housing_Status_Mortgage': [1 if data['Housing_Status'] == 2 else 0],
        'Housing_Status_Own': [1 if data['Housing_Status'] == 3 else 0],
        'Housing_Status_Rent': [1 if data['Housing_Status'] == 4 else 0],

        'Citizenship_By Birth': [1 if data['Citizenship'] == 1 else 0],
        'Citizenship_By Other Means': [1 if data['Citizenship'] == 2 else 0]
    })

    return df




class Predicted_Response(BaseModel):

    Approved: str

@app.get("/")   
def welcome():
          
       return {"WELCOME BRO :"}


@app.post("/PREDICT", response_model=Predicted_Response)
def Predict_approval(features: CreditCardFeatures):
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500 , detail="Model or Scaler not loaded. Server configuration error.")
    
    try:
        input_data = preprocess_input(features)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
        return {"Approved": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app , host="127.0.0.1" , port=8000)