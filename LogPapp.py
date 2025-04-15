# === Libraries ===

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# === Load Model and Data ===

model = joblib.load("caco2_model.pkl")
selected_features = joblib.load("selected_features.pkl")
X_train_ref = joblib.load("ad_reference_data.pkl")


# === Helper Functions ===

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


def is_in_applicability_domain(X_train, X_new, threshold_factor=3.0):
    mean_train = np.mean(X_train, axis=0)
    distances_train = np.sqrt(np.sum((X_train - mean_train) ** 2, axis=1))
    threshold = np.mean(distances_train) + threshold_factor * np.std(distances_train)

    distances_new = np.sqrt(np.sum((X_new - mean_train) ** 2, axis=1))
    return distances_new <= threshold


def classify_log_papp(log_papp):
    if log_papp > -5.0:
        return "High permeability ✅"
    elif -6.0 <= log_papp <= -5.0:
        return "Moderate permeability ⚠️"
    else:
        return "Low permeability ❌"


# === Streamlit UI ===

st.set_page_config(page_title="Caco-2 Permeability Predictor", layout="centered")

st.title("🧪 Caco-2 Permeability Prediction")
st.markdown("Enter SMILES strings to predict **Caco-2 permeability**.")

smiles_input = st.text_area("Enter SMILES (one per line):", height=150)


if st.button("Predict"):
    smiles_list = [s.strip() for s in smiles_input.strip().split("\n") if s.strip()]
    results = []

    for smi in smiles_list:
        desc = calculate_descriptors(smi)
        if desc is None:
            results.append({"SMILES": smi, "log(P_app)": "Invalid", "P_app (cm/s)": "N/A", "Permeability Class": "N/A", "In Domain": "N/A"})
            continue

        desc_df = pd.DataFrame([desc])
        X_pred = pd.DataFrame(0, index=range(1), columns=X_train_ref.columns)
        for col in desc_df.columns:
            if col in X_pred.columns:
                X_pred[col] = desc_df[col].values

        X_selected = X_pred[selected_features]
        log_papp = model.predict(X_selected)[0]
        p_app = 10 ** log_papp
        in_domain = is_in_applicability_domain(X_train_ref[selected_features], X_selected)[0]
        permeability_class = classify_log_papp(log_papp)

        results.append({
            "SMILES": smi,
            "log(P_app)": round(log_papp, 4),
            "P_app (cm/s)": "{:.2e}".format(p_app),
            "Permeability Class": permeability_class,
            "In Domain": "✅ Yes" if in_domain else "⚠️ No"
        })

    st.markdown("### Results:")
    st.dataframe(pd.DataFrame(results))


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

st.markdown("---")
st.markdown("### 📊 Our Model's Performance")
st.markdown("Split used - Scaffold")
st.markdown("""
**Training Set**  
- RMSE: `0.1027`  
- R²: `0.9834`  

**Validation Set**  
- RMSE: `0.1626`  
- R²: `0.9559`  

**Test Set**  
- RMSE: `0.1347`  
- R²: `0.9712`
""")
