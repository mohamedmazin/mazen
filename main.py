from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os

app = FastAPI(
    title="Career Track Recommender API",
    description="API for recommending career tracks based on student skills and experience.",
    version="2.0.0"
)

# Add CORS Middleware to allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
MODEL_PATH = 'recommender_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file {MODEL_PATH} not found. Please run train_final_model.py first.")

with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
    
    # Fix for scikit-learn version mismatch (monotonic_cst error)
    if data and 'model' in data:
        model_obj = data['model']
        if hasattr(model_obj, 'estimators_'):
            for tree in model_obj.estimators_:
                if not hasattr(tree, 'monotonic_cst'):
                    tree.monotonic_cst = None
    
    model = data['model']
    encoders = data['encoders']
    features = data['features']
    target_col = data.get('target_col', 'Career_Goals')

class UserInterests(BaseModel):
    Python: int = 5
    Java: int = 5
    C_plus_plus: int = 5
    JavaScript: int = 5
    C_sharp: int = 5
    PHP: int = 5
    Ruby: int = 5
    Swift: int = 5
    Go: int = 5
    Rust: int = 5
    Software_Development_Experience: int = 5
    Database_Management: int = 5
    Networking_Skills: int = 5
    Web_Development_Experience: int = 5
    Communication_Skills: int = 5
    Problem_Solving_Abilities: int = 5
    Teamwork_Collaboration: int = 5
    Time_Management: int = 5
    Adaptability: int = 5
    Personal_Interests: str = "Coding"
    Internship_Experience: str = "No"
    Certifications_Training: str = "No"
    Leadership_Experience: str = "No"

@app.get("/", tags=["Health"])
def read_root():
    return {
        "status": "online",
        "message": "Career Track Recommender AI API is running",
        "docs": "/docs"
    }

@app.post("/recommend", tags=["Prediction"])
def recommend(interests: UserInterests):
    try:
        # Convert input to dictionary
        input_dict = interests.model_dump()
        
        # Map pydantic field names back to dataset column names
        # Also handle potential case sensitivity issues by normalizing keys
        final_dict = {}
        for k, v in input_dict.items():
            # Handle special characters and ensure mapping
            col_name = k.replace('C_plus_plus', 'C++').replace('C_sharp', 'C#')
            final_dict[col_name] = v
            
        # Create a list for prediction in the exact order of features
        processed_values = []
        for col in features:
            # Try to get value with exact match, or case-insensitive if needed
            val = final_dict.get(col)
            if val is None:
                # Fallback for common naming variations
                normalized_final_dict = {k.lower().replace(' ', '_'): v for k, v in final_dict.items()}
                normalized_col = col.lower().replace(' ', '_')
                val = normalized_final_dict.get(normalized_col, 5) # Default to 5 if not found
            
            # If it's a categorical feature, encode it
            if col in encoders and hasattr(encoders[col], 'classes_'):
                le = encoders[col]
                val_str = str(val)
                if val_str not in le.classes_:
                    val_str = le.classes_[0]
                processed_values.append(le.transform([val_str])[0])
            else:
                # Numerical feature
                try:
                    processed_values.append(int(val))
                except:
                    processed_values.append(5)
            
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([processed_values], columns=features)
        
        # Get probabilities
        probs = model.predict_proba(input_df)[0]
        
        # Ensure probabilities are normalized and in [0, 1] range
        if np.max(probs) > 1.0 or np.any(probs < 0):
            probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs) + 1e-6)
        probs = probs / (np.sum(probs) + 1e-6)
        
        # Get top 5 indices
        top_indices = np.argsort(probs)[::-1][:5]
        
        # Get top 5 labels
        le_target = encoders[target_col]
        recommendations = []
        for idx in top_indices:
            raw_name = le_target.inverse_transform([idx])[0]
            # Format name: replace '-' with ' ' and title case
            formatted_name = raw_name.replace('-', ' ').title()
            
            # Formatting confidence as percentage string like in streamlit
            confidence_val = float(probs[idx])
            formatted_confidence = f"{confidence_val:.2%}"
            
            recommendations.append({
                "track": formatted_name,
                "confidence": formatted_confidence
            })
            
        return {
            "success": True,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
