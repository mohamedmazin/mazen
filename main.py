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
        input_dict = interests.dict()
        
        # Map pydantic field names back to dataset column names
        final_dict = {}
        for k, v in input_dict.items():
            col_name = k.replace('C_plus_plus', 'C++').replace('C_sharp', 'C#')
            final_dict[col_name] = v
            
        # Preprocess features using saved encoders
        for col in features:
            le = encoders[col]
            val = str(final_dict.get(col, "0"))
            if val not in le.classes_:
                val = le.classes_[0]
            final_dict[col] = le.transform([val])[0]
            
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([final_dict])[features]
        
        # Get probabilities
        probs = model.predict_proba(input_df)[0]
        
        # Get top 5 indices
        top_indices = np.argsort(probs)[::-1][:5]
        
        # Get top 5 labels
        le_target = encoders[target_col]
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                "track": le_target.inverse_transform([idx])[0],
                "confidence": round(float(probs[idx]), 4)
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
