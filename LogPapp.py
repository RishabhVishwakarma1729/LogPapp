import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
import pickle

# Set page configuration
st.set_page_config(page_title="Caco2 Permeability Prediction", layout="wide")

# Title and description
st.title("Caco2 Permeability Prediction")
st.markdown("""
This application predicts the Caco2 permeability of molecules using a pre-trained machine learning model.
Enter a SMILES string to get a prediction.
""")

# Function to calculate molecular descriptors
def calculate_descriptors(smiles, radius=2, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        desc_names = [desc_name[0] for desc_name in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
        descriptors = calc.CalcDescriptors(mol)
        desc_dict = dict(zip(desc_names, descriptors))
        desc_dict['HBondDonorCount'] = Lipinski.NumHDonors(mol)
        desc_dict['HBondAcceptorCount'] = Lipinski.NumHAcceptors(mol)
        desc_dict['RotatableBondCount'] = Lipinski.NumRotatableBonds(mol)
        desc_dict['TPSA'] = MolSurf.TPSA(mol)
        desc_dict['LabuteASA'] = MolSurf.LabuteASA(mol)
        desc_dict['HeavyAtomCount'] = Lipinski.HeavyAtomCount(mol)
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))
        for i, bit in enumerate(fp):
            desc_dict[f'Morgan_{i}'] = bit
        return desc_dict
    except:
        return None

# Function to check if molecule is in applicability domain
def is_in_applicability_domain(X_train, X_new, threshold_factor=3.0):
    mean_train = np.mean(X_train, axis=0)
    distances_train = np.sqrt(np.sum((X_train - mean_train)**2, axis=1))
    threshold = np.mean(distances_train) + threshold_factor * np.std(distances_train)
    distances_new = np.sqrt(np.sum((X_new - mean_train)**2, axis=1))
    return distances_new <= threshold

# Function to predict permeability
def predict_permeability(smiles, model, selected_features, X_train):
    # Calculate descriptors for the SMILES
    desc_dict = calculate_descriptors(smiles)
    
    if desc_dict is None:
        return {
            "error": "Invalid SMILES or could not calculate descriptors"
        }
    
    # Convert to DataFrame
    desc_df = pd.DataFrame([desc_dict])
    
    # Ensure all needed columns are present
    all_columns = X_train.columns
    X_pred = pd.DataFrame(0, index=[0], columns=all_columns)
    
    for col in desc_df.columns:
        if col in X_pred.columns:
            X_pred[col] = desc_df[col]
    
    # Select the features needed for prediction
    X_pred_selected = X_pred[selected_features]
    
    # Make prediction
    prediction = model.predict(X_pred_selected)[0]
    
    # Check if molecule is in applicability domain
    in_domain = is_in_applicability_domain(X_train[selected_features], X_pred_selected)[0]
    
    return {
        "SMILES": smiles,
        "Prediction": prediction,
        "In_Domain": in_domain
    }

# Load pre-trained model and related data
@st.cache_resource
def load_model():
    try:
        with open('caco2_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
        with open('X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        return model, selected_features, X_train
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Main app
model, selected_features, X_train = load_model()

if model is not None and selected_features is not None and X_train is not None:
    # Input area for SMILES
    smiles_input = st.text_input("Enter SMILES string:", "")
    
    # Optional: Display molecular structure
    if smiles_input:
        try:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                st.write("Molecular Structure:")
                # Convert to image and display
                from rdkit.Chem import Draw
                mol_img = Draw.MolToImage(mol)
                st.image(mol_img)
        except:
            st.warning("Could not render molecular structure")
    
    # Prediction button
    if st.button("Predict") and smiles_input:
        # Show spinner while calculating
        with st.spinner("Calculating..."):
            result = predict_permeability(smiles_input, model, selected_features, X_train)
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display results
                st.subheader("Prediction Results")
                
                # Format as table
                result_df = pd.DataFrame({
                    "SMILES": [result["SMILES"]],
                    "Predicted Permeability": [f"{result['Prediction']:.4f}"],
                    "In Applicability Domain": ["Yes" if result["In_Domain"] else "No"]
                })
                
                st.table(result_df)
                
                # Display warning if not in applicability domain
                if not result["In_Domain"]:
                    st.warning("⚠️ Warning: This molecule is outside the applicability domain of the model. Prediction may be less reliable.")
                
                # Add some molecular properties
                if "mol" in locals() and mol:
                    st.subheader("Molecular Properties")
                    props_df = pd.DataFrame({
                        "Property": ["Molecular Weight", "LogP", "H-Bond Donors", "H-Bond Acceptors", "Rotatable Bonds", "TPSA"],
                        "Value": [
                            f"{Descriptors.MolWt(mol):.2f}",
                            f"{Descriptors.MolLogP(mol):.2f}",
                            Lipinski.NumHDonors(mol),
                            Lipinski.NumHAcceptors(mol),
                            Lipinski.NumRotatableBonds(mol),
                            f"{MolSurf.TPSA(mol):.2f}"
                        ]
                    })
                    st.table(props_df)
else:
    st.error("Model files not found. Please ensure 'caco2_model.pkl', 'selected_features.pkl', and 'X_train.pkl' are in the same directory as this app.")
    
    # Provide instructions for missing model files
    st.info("""
    If you're running this app for the first time, you need to:
    1. Make sure your trained model files are saved in the same directory as this app
    2. Check that the model files have the expected names: 'caco2_model.pkl', 'selected_features.pkl', and 'X_train.pkl'
    """)
