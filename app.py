from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = FastAPI()

#Define the structure of the incoming data
class FraudInput(BaseModel):
    age: int
    policy_state: str
    policy_csl: str 
    incident_type: str 
    collision_type: str
    incident_severity: str
    authorities_contacted: str 
    incident_state: str 
    incident_city: str
    number_of_vehicles_involved: int
    property_damage: str
    bodily_injuries: int
    witnesses: int
    police_report_available: str
    total_claim_amount: int
    injury_claim: int
    property_claim: int
    vehicle_claim: int
    auto_make: str
    auto_model: str
    auto_year: int

    @app.get("/")
    def home():
        return {"message":"Fraud Detection API is Running"}

    @app.post("/predict")
    def predict_fraud(data: dict):
        # 1. Initiate CustomData with UI inputs
        custom_data = CustomData(**data)
        feature_df = custom_data.get_data_as_data_frame()

        #2. Call Prediction Pipeline
        predict_pipeline =PredictPipeline()
        result = predict_pipeline.predict(feature_df)

    # 3. Return result
        status = "Fraudulent" if result == 1 else "Legitimate"
        return {"prediction": status}