import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(page_title="Track Recommender AI", layout="wide")

st.title("🚀 Career Track Recommender AI")
st.markdown("ادخل مهاراتك وخبراتك للحصول على أفضل الترشيحات للتراكات البرمجية.")

# Load model and encoders
MODEL_PATH = 'recommender_model.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

data = load_model()

if data is None:
    st.error(f"Model file {MODEL_PATH} not found. Please train the model first.")
else:
    model = data['model']
    encoders = data['encoders']
    features = data['features']
    target_col = data.get('target_col', 'Career_Goals')

    # Create UI for inputs
    st.header("📊 Skills & Experience (Rate from 0 to 9)")
    st.info("اختر مستواك في كل مهارة (0 تعني لا يوجد خبرة، 9 تعني خبير)")
    
    inputs = {}
    
    # Numeric features
    numeric_features = [
        'Python', 'Java', 'C++', 'JavaScript', 'C#', 'PHP', 'Ruby', 'Swift', 'Go', 'Rust',
        'Software_Development_Experience', 'Database_Management', 'Networking_Skills', 
        'Web_Development_Experience', 'Communication_Skills', 'Problem_Solving_Abilities', 
        'Teamwork_Collaboration', 'Time_Management', 'Adaptability'
    ]
    
    for feature in numeric_features:
        # Using horizontal radio buttons for 0-9 selection
        inputs[feature] = st.radio(
            f"**{feature.replace('_', ' ')}**",
            options=list(range(10)),
            index=5,
            horizontal=True,
            key=feature
        )
        st.markdown("---")
            
    st.header("📋 Personal Info & Experience")
    
    inputs['Personal_Interests'] = st.selectbox("Preferences", encoders['Personal_Interests'].classes_)
    st.markdown("---")
    inputs['Internship_Experience'] = st.selectbox("Internship Experience", encoders['Internship_Experience'].classes_)
    st.markdown("---")
    inputs['Certifications_Training'] = st.selectbox("Certifications & Training", encoders['Certifications_Training'].classes_)
    st.markdown("---")
    inputs['Leadership_Experience'] = st.selectbox("Leadership Experience", encoders['Leadership_Experience'].classes_)
    st.markdown("---")

    if st.button("Get Recommendations", type="primary"):
        try:
            # Preprocess inputs
            processed_inputs = {}
            for col in features:
                le = encoders[col]
                val = str(inputs[col])
                if val not in le.classes_:
                    val = le.classes_[0]
                processed_inputs[col] = le.transform([val])[0]
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([processed_inputs])[features]
            
            # Get probabilities
            probs = model.predict_proba(input_df)[0]
            top_indices = np.argsort(probs)[::-1][:5]
            
            # Get labels
            le_target = encoders[target_col]
            
            st.header("🎯 Top 5 Recommended Tracks")
            
            for i, idx in enumerate(top_indices):
                track_name = le_target.inverse_transform([idx])[0]
                confidence = probs[idx]
                
                # Progress bar for confidence
                st.subheader(f"{i+1}. {track_name}")
                st.progress(float(confidence))
                st.write(f"Confidence: {confidence:.2%}")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.info("هذا الموديل مدرب على الداتا سيت المحدثة ليناسب تراكات المشروع.")
