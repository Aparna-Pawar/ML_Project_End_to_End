import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def predict(self, features):
        try:
            model_dict = load_object(file_path='artifacts/model.pkl')
            preprocessor = load_object(file_path='artifacts/preprocessor.pkl')
           
            denom = features['total_claim_amount'].replace(0, 1)
            features['injury_ratio'] = features['injury_claim'] / denom
            features['property_ratio'] = features['property_claim'] / denom
            features['vehicle_ratio'] = features['vehicle_claim'] / denom
            
            # 3. Rename columns to match the hyphenated versions in your training set
            features.rename(columns={'capital_gains': 'capital-gains', 'capital_loss': 'capital-loss'}, inplace=True)

            # 4. ALIGN COLUMNS
            # These must be the columns your preprocessor expects as INPUT.
            # Based on your error, 'policy_bind_date' MUST be in this list.
            input_columns = [
                'age', 'months_as_customer', 'policy_bind_date', 'policy_state', 
                'policy_csl', 'policy_deductable', 'umbrella_limit', 'capital-gains', 
                'capital-loss', 'insured_sex', 'insured_education_level', 
                'insured_occupation', 'insured_hobbies', 'insured_relationship', 
                'incident_date', 'incident_type', 'collision_type', 'incident_severity', 
                'authorities_contacted', 'incident_state', 'incident_city', 
                'incident_hour_of_the_day', 'number_of_vehicles_involved', 
                'property_damage', 'bodily_injuries', 'witnesses', 
                'police_report_available', 'total_claim_amount', 'injury_claim', 
                'property_claim', 'vehicle_claim', 'auto_make', 'auto_model', 
                'auto_year', 'injury_ratio', 'property_ratio', 'vehicle_ratio'
            ]

            # Filter and reorder
            features = features[input_columns]

            # 5. TRANSFORM (The preprocessor will now find 'policy_bind_date' and be happy)
            data_scaled = preprocessor.transform(features)

            # 6. PREDICT
            model = model_dict['model']
            threshold = model_dict.get('threshold', 0.5)
            
            probs = model.predict_proba(data_scaled)[:, 1]
            return 1 if probs[0] >= threshold else 0
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, **kwargs):
        self.data_dict = {k: [v] for k, v in kwargs.items()}

    def get_data_as_data_frame(self):
        return pd.DataFrame(self.data_dict)