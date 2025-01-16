#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import streamlit as st
import joblib

# Load the trained LightGBM model
model = joblib.load('best_lightgbm_model.pkl')

# Load encoders and mappings
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('doc_supplier_mapping.pkl', 'rb') as f:
    doc_supplier_mapping = pickle.load(f)

# Extract encoders
label_encoder_period = label_encoders['Period']

# Define the Streamlit app
st.image("logo.jpg", width=150)  # Add logo
st.title("Farmer IP Prediction Application")
st.write("Enter the features to predict the Farmer IP:")

# Dropdown options for DOC Supplier
doc_supplier_options = list(doc_supplier_mapping.keys())

# Function to preprocess user inputs
def preprocess_inputs(features):
    # Encode categorical features
    features['Period'] = label_encoder_period.transform([features['Period']])[0]
    features['DOC Supplier'] = doc_supplier_mapping.get(features['DOC Supplier'], 0)  # Default to 0 if not found

    # Calculate derived features
    features['Feed Shipment (Kg)/Bird'] = (features['Total Feed Used (Zak)'] * 50) / features['Population']
    features['FCR'] = (features['Total Feed Used (Zak)'] * 50) / features['Weight (Kg)']
    features['FCR Difference'] = features['FCR'] - features['Standard FCR']
    features['Harvest ABW'] = features['Weight (Kg)'] / features['Harvested Birds']
    features['Mortality (Birds)'] = features['Population'] - features['Harvested Birds']
    features['Mortality %'] = (features['Mortality (Birds)'] / features['Population']) * 100
    features['Remaining Stock'] = (features['Population'] - features['Harvested Birds']) - features['Deaths']

    # Create the feature array
    input_features = np.array([
        features['DOC Supplier'], features['DOC Price'], features['Feed Shipment (Kg)/Bird'], features['FCR'],
        features['FCR Difference'], features['Standard FCR'], features['Harvest ABW'], features['Remaining Stock'],
        features['Mortality (Birds)'], features['Mortality %'], features['Weight (Kg)'], features['Population'],
        features['Total Feed Used (Zak)'], features['Harvested Birds'], features['Period'], features['Average Harvest Age']
    ])
    return input_features

# Collect user input
features = {
    'Period': st.selectbox("Select Period", label_encoder_period.classes_),
    'Population': st.number_input("Population", min_value=1, value=10000),
    'DOC Supplier': st.selectbox("Select DOC Supplier", doc_supplier_options),
    'DOC Price': st.number_input("DOC Price", min_value=0.0, value=5000.0),
    'Total Feed Used (Zak)': st.number_input("Total Feed Used (Zak)", min_value=0.0, value=500.0),
    'Deaths': st.number_input("Deaths", min_value=0.0, value=0.0),
    'Weight (Kg)': st.number_input("Weight (Kg)", min_value=0.0, value=5000.0),
    'Harvested Birds': st.number_input("Harvested Birds", min_value=0, value=10000),
    'Standard FCR': st.number_input("Standard FCR", min_value=0.0, value=1.8),
    'Average Harvest Age': st.number_input("Average Harvest Age", min_value=0.0, value=40.0)
}

# Preprocess inputs
input_features = preprocess_inputs(features)

# Predict IP value
if st.button("Predict IP"):
    prediction = model.predict(input_features.reshape(1, -1))
    st.write(f"Predicted IP: {prediction[0]:.2f}")

# Information for users
st.write("""
### Derived Variables
The following features are automatically calculated based on your inputs:
- **Feed Shipment (Kg)/Bird**
- **FCR**
- **FCR Difference**
- **Harvest ABW**
- **Remaining Stock**
- **Mortality (Birds)**
- **Mortality %**
""")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




