# === Libraries ===

# Importing Streamlit for creating the web app UI
import streamlit as st

# Importing Pandas for handling data and DataFrames
import pandas as pd

# Importing NumPy for performing numerical operations
import numpy as np

# Importing joblib for loading pre-trained ML models and saved data
import joblib

# Importing RDKit for handling molecules and calculating descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

# Importing scikit-learn for metrics (used here for mean squared error, though unused)
from sklearn.metrics import mean_squared_error

# Importing XGBoost for building the regression model
from xgboost import XGBRegressor


# === Loading Model and Features ===

# Loading trained XGBoost model for Caco-2 permeability prediction
model = joblib.load("caco2_model.pkl")

# Loading selected features used during model training
selected_features = joblib.load("selected_features.pkl")

# Loading reference training data for performing Applicability Domain checks
X_train_ref = joblib.load("ad_reference_data.pkl")


# === Defining Descriptor Calculator Function ===
def calculate_descriptors(smiles, radius=2, nBits=2048):
    """
    Taking a SMILES string and calculating a set of molecular descriptors and Morgan fingerprint bits.
    Returning a dictionary of descriptors.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)  # Converting SMILES to RDKit molecule
        if mol is None:
            return None

        # Getting list of standard RDKit descriptors
        desc_names = [desc_name[0] for desc_name in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

        # Calculating descriptor values
        descriptors = calc.CalcDescriptors(mol)
        desc_dict = dict(zip(desc_names, descriptors))

        # Adding Lipinski and other custom descriptors
        desc_dict['HBondDonorCount'] = Lipinski.NumHDonors(mol)
        desc_dict['HBondAcceptorCount'] = Lipinski.NumHAcceptors(mol)
        desc_dict['RotatableBondCount'] = Lipinski.NumRotatableBonds(mol)
        desc_dict['TPSA'] = MolSurf.TPSA(mol)
        desc_dict['LabuteASA'] = MolSurf.LabuteASA(mol)
        desc_dict['HeavyAtomCount'] = Lipinski.HeavyAtomCount(mol)

        # Generating Morgan fingerprint (bit vector)
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))
        for i, bit in enumerate(fp):
            desc_dict[f'Morgan_{i}'] = bit

        return desc_dict
    except:
        # Failing silently and returning None if an error occurs (e.g., invalid SMILES)
        return None


# === Defining Applicability Domain Check Function ===
def is_in_applicability_domain(X_train, X_new, threshold_factor=3.0):
    """
    Checking whether the new sample is within the Applicability Domain (AD) of the training set.
    Based on calculating Euclidean distance from the mean of training data.
    """
    mean_train = np.mean(X_train, axis=0)
    distances_train = np.sqrt(np.sum((X_train - mean_train) ** 2, axis=1))
    threshold = np.mean(distances_train) + threshold_factor * np.std(distances_train)

    distances_new = np.sqrt(np.sum((X_new - mean_train) ** 2, axis=1))
    return distances_new <= threshold


# === Building Streamlit UI ===

# Setting Streamlit page configuration: title and layout
st.set_page_config(page_title="Caco-2 Permeability Predictor", layout="centered")

# Displaying app title and description
st.title("🧪 Caco-2 Permeability Prediction")
st.markdown("Enter SMILES strings to predict **Caco-2 permeability** using an XGBoost model.")

# Creating text input box for SMILES strings
smiles_input = st.text_area("Enter SMILES (one per line):", height=150)

# Creating button to trigger prediction
if st.button("Predict"):
    # Splitting and stripping input into individual SMILES
    smiles_list = [s.strip() for s in smiles_input.strip().split("\n") if s.strip()]
    results = []

    # Iterating over each SMILES string
    for smi in smiles_list:
        desc = calculate_descriptors(smi)
        if desc is None:
            results.append({"SMILES": smi, "Prediction": "Invalid SMILES", "In Domain": "N/A"})
            continue

        # Converting descriptor dictionary to DataFrame
        desc_df = pd.DataFrame([desc])

        # Creating prediction DataFrame with columns matching training data
        X_pred = pd.DataFrame(0, index=range(1), columns=X_train_ref.columns)
        for col in desc_df.columns:
            if col in X_pred.columns:
                X_pred[col] = desc_df[col].values

        # Selecting only model-relevant features
        X_selected = X_pred[selected_features]

        # Making prediction using trained model
        pred = model.predict(X_selected)[0]

        # Performing applicability domain check
        in_domain = is_in_applicability_domain(X_train_ref[selected_features], X_selected)[0]

        # Appending result
        results.append({
            "SMILES": smi,
            "Prediction": round(pred, 4),
            "In Domain": "✅ Yes" if in_domain else "⚠️ No"
        })

    # Displaying results
    st.markdown("### Results:")
    st.dataframe(pd.DataFrame(results))
