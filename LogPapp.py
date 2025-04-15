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


# === Defining Applicability Domain Check Function ===
def is_in_applicability_domain(X_train, X_new, threshold_factor=3.0):
    mean_train = np.mean(X_train, axis=0)
    distances_train = np.sqrt(np.sum((X_train - mean_train) ** 2, axis=1))
    threshold = np.mean(distances_train) + threshold_factor * np.std(distances_train)

    distances_new = np.sqrt(np.sum((X_new - mean_train) ** 2, axis=1))
    return distances_new <= threshold


# === Building Streamlit UI ===

st.set_page_config(page_title="Caco-2 Permeability Predictor", layout="centered")

st.title("🧪 Caco-2 Permeability Prediction")
st.markdown("Enter SMILES strings to predict **Caco-2 permeability**.")
st.markdown("""
**Caco-2 permeability** refers to the ability of a compound to pass through a layer of Caco-2 cells, which are human colorectal adenocarcinoma cells commonly used as an *in vitro* model of the intestinal barrier.

---

### 🔍 Key Points

- **Caco-2 Cells**: Form tight junctions after ~21 days, resembling the intestinal lining.
- **Purpose**: Simulates **passive diffusion** across the intestinal wall.
- **Measured As**:  
  - `P_app` (Apparent Permeability Coefficient), in cm/s or log(P_app)

---

### 🧬 Why It Matters

- Predicts **oral bioavailability**
- Flags **poorly absorbed** or **efflux-prone** compounds
- Reduces reliance on early animal testing
""")

smiles_input = st.text_area("Enter SMILES (one per line):", height=150)

if st.button("Predict"):
    smiles_list = [s.strip() for s in smiles_input.strip().split("\n") if s.strip()]
    results = []

    for smi in smiles_list:
        desc = calculate_descriptors(smi)
        if desc is None:
            results.append({"SMILES": smi, "log(P_app)": "Invalid SMILES", "P_app (cm/s)": "N/A", "In Domain": "N/A"})
            continue

        desc_df = pd.DataFrame([desc])

        X_pred = pd.DataFrame(0, index=range(1), columns=X_train_ref.columns)
        for col in desc_df.columns:
            if col in X_pred.columns:
                X_pred[col] = desc_df[col].values

        X_selected = X_pred[selected_features]

        pred = model.predict(X_selected)[0]
        p_app = 10 ** pred
        in_domain = is_in_applicability_domain(X_train_ref[selected_features], X_selected)[0]

        results.append({
            "SMILES": smi,
            "log(P_app)": round(pred, 4),
            "P_app (cm/s)": "{:.2e}".format(p_app),
            "In Domain": "✅ Yes" if in_domain else "⚠️ No"
        })

    st.markdown("### Results:")
    st.dataframe(pd.DataFrame(results))
