import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(slef):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            # 2. Get probabilities from the model
            model = model_dict['model']
            threshold = model_dict['threshold']
            
            probs = model.predict_proba(data_scaled)[:, 1]
            
            # 3. Apply the custom threshold we found during training
            preds = (probs >= threshold).astype(int)
            
            return preds[0]
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, 
                 age: int, policy_state: str, policy_csl: str, 
                 incident_type: str, collision_type: str, incident_severity: str,
                 authorities_contacted: str, incident_state: str, incident_city: str,
                 number_of_vehicles_involved: int, property_damage: str, 
                 bodily_injuries: int, witnesses: int, police_report_available: str,
                 total_claim_amount: int, injury_claim: int, property_claim: int, 
                 vehicle_claim: int, auto_make: str, auto_model: str, auto_year: int,
                 policy_bind_date: str, incident_date: str,
                 # --- ADDING THE MISSING 11 FIELDS HERE ---
                 incident_hour_of_the_day: int, policy_deductable: int,
                 umbrella_limit: int, capital_gains: int, capital_loss: int,
                 insured_sex: str, insured_relationship: str, insured_occupation: str,
                 insured_education_level: str, months_as_customer: int,
                 insured_hobbies: str):
        
        self.data_dict = {
            "age": [age], "policy_state": [policy_state], "policy_csl": [policy_csl],
            "incident_type": [incident_type], "collision_type": [collision_type],
            "incident_severity": [incident_severity], "authorities_contacted": [authorities_contacted],
            "incident_state": [incident_state], "incident_city": [incident_city],
            "number_of_vehicles_involved": [number_of_vehicles_involved],
            "property_damage": [property_damage], "bodily_injuries": [bodily_injuries],
            "witnesses": [witnesses], "police_report_available": [police_report_available],
            "total_claim_amount": [total_claim_amount], "injury_claim": [injury_claim],
            "property_claim": [property_claim], "vehicle_claim": [vehicle_claim],
            "auto_make": [auto_make], "auto_model": [auto_model], "auto_year": [auto_year],
            "policy_bind_date": [policy_bind_date], "incident_date": [incident_date],
            # --- MAPPING THE MISSING FIELDS ---
            "incident_hour_of_the_day": [incident_hour_of_the_day],
            "policy_deductable": [policy_deductable],
            "umbrella_limit": [umbrella_limit],
            "capital_gains": [capital_gains], # Note the dash if your CSV used 'capital-gains'
            "capital_loss": [capital_loss],   # Note the dash if your CSV used 'capital-loss'
            "insured_sex": [insured_sex],
            "insured_relationship": [insured_relationship],
            "insured_occupation": [insured_occupation],
            "insured_education_level": [insured_education_level],
            "months_as_customer": [months_as_customer],
            "insured_hobbies": [insured_hobbies]
        }

    def get_data_as_data_frame(self):
        return pd.DataFrame(self.data_dict)