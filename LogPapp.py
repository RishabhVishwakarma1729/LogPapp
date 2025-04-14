import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# === Load model and selected features ===
model = joblib.load("models/caco2_model.pkl")
selected_features = joblib.load("models/selected_features.pkl")
X_train_ref = joblib.load("models/ad_reference_data.pkl")  # For applicability domain check

# === Descriptor calculator ===
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

# === Applicability Domain Check ===
def is_in_applicability_domain(X_train, X_new, threshold_factor=3.0):
    mean_train = np.mean(X_train, axis=0)
    distances_train = np.sqrt(np.sum((X_train - mean_train) ** 2, axis=1))
    threshold = np.mean(distances_train) + threshold_factor * np.std(distances_train)
    distances_new = np.sqrt(np.sum((X_new - mean_train) ** 2, axis=1))
    return distances_new <= threshold

# === Streamlit UI ===
st.set_page_config(page_title="Caco-2 Permeability Predictor", layout="centered")
st.title("🧪 Caco-2 Permeability Prediction")
st.markdown("Enter SMILES strings to predict **Caco-2 permeability** using an XGBoost model.")

smiles_input = st.text_area("Enter SMILES (one per line):", height=150)

if st.button("Predict"):
    smiles_list = [s.strip() for s in smiles_input.strip().split("\n") if s.strip()]
    results = []
    
    for smi in smiles_list:
        desc = calculate_descriptors(smi)
        if desc is None:
            results.append({"SMILES": smi, "Prediction": "Invalid SMILES", "In Domain": "N/A"})
            continue

        desc_df = pd.DataFrame([desc])
        X_pred = pd.DataFrame(0, index=range(1), columns=X_train_ref.columns)
        for col in desc_df.columns:
            if col in X_pred.columns:
                X_pred[col] = desc_df[col].values

        X_selected = X_pred[selected_features]
        pred = model.predict(X_selected)[0]
        in_domain = is_in_applicability_domain(X_train_ref[selected_features], X_selected)[0]

        results.append({
            "SMILES": smi,
            "Prediction": round(pred, 4),
            "In Domain": "✅ Yes" if in_domain else "⚠️ No"
        })

    st.markdown("### Results:")
    st.dataframe(pd.DataFrame(results))

