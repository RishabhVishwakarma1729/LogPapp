# Caco-2 Permeability Prediction Web Application

This repository contains a **Streamlit-based web application** for predicting **Caco-2 cell permeability** of chemical compounds using a pre-trained **XGBoost regression model**. The app accepts **SMILES strings** as input and returns predicted permeability values along with an **Applicability Domain (AD) check** to validate the reliability of predictions.

---

## ✨ Features

- User-friendly web interface via Streamlit
- Accepts one or multiple SMILES inputs
- Calculates molecular descriptors and fingerprints using RDKit
- Predicts permeability using an XGBoost model
- Performs Applicability Domain checking
- Displays predictions in a tabular format

---

## 📄 Requirements

Install all required dependencies with:

```bash
pip install streamlit pandas numpy scikit-learn xgboost joblib
```

> **Note:** RDKit cannot be installed via pip for most users. Instead, use conda:

```bash
conda install -c rdkit rdkit
```

---

## 📁 Files in This Project

- `app.py` - Main Streamlit application script
- `caco2_model.pkl` - Pre-trained XGBoost regression model
- `selected_features.pkl` - List of selected molecular features
- `ad_reference_data.pkl` - Training set descriptors for Applicability Domain (AD) check

---

## 📓 Full Code Explanation

The `app.py` script is divided into the following parts:

---

### 📊 Library Imports

```python
import streamlit as st
```
- **Streamlit**: A framework for building interactive web applications in Python. Used to create input boxes, buttons, display data, and set page layout.

```python
import pandas as pd
```
- **Pandas**: A powerful library for data manipulation and analysis. Used here to create and manage DataFrames for descriptor data and results.

```python
import numpy as np
```
- **NumPy**: Provides support for arrays and numerical computing. Used for mathematical operations including calculating Euclidean distances.

```python
import joblib
```
- **joblib**: Efficient tool for saving and loading Python objects. Used here to load the trained model and metadata like selected features.

```python
from rdkit import Chem
```
- **RDKit.Chem**: Converts SMILES strings to molecular objects using `Chem.MolFromSmiles()`.

```python
from rdkit.Chem import Descriptors, Lipinski, MolSurf, AllChem
```
- **Descriptors**: Contains many molecular properties like molecular weight, logP, etc.
- **Lipinski**: Functions to calculate Lipinski Rule-of-5 descriptors (H-bond donors/acceptors, rotatable bonds, etc.).
- **MolSurf**: Calculates molecular surface descriptors such as TPSA and LabuteASA.
- **AllChem**: Used for generating molecular fingerprints like Morgan fingerprints.

```python
from rdkit.ML.Descriptors import MoleculeDescriptors
```
- **MoleculeDescriptors**: Facilitates bulk descriptor calculation using descriptor names.

```python
from sklearn.metrics import mean_squared_error
```
- **mean_squared_error**: Not used in this script but typically used for model evaluation.

```python
from xgboost import XGBRegressor
```
- **XGBRegressor**: Gradient boosting model used for regression tasks.

---

### 🧬 Load Model and Metadata

```python
model = joblib.load("caco2_model.pkl")
selected_features = joblib.load("selected_features.pkl")
X_train_ref = joblib.load("ad_reference_data.pkl")
```
- Load the trained XGBoost model.
- Load the list of features selected during model training.
- Load the reference training descriptor data used for AD checks.

---

### 🎓 Descriptor Calculation Function

```python
def calculate_descriptors(smiles, radius=2, nBits=2048):
```
- Converts SMILES to RDKit molecule.
- Calculates:
  - Standard descriptors using `Descriptors._descList`
  - Lipinski descriptors
  - TPSA and LabuteASA
  - Morgan fingerprint bits (using `GetMorganFingerprintAsBitVect`)
- Combines all into a dictionary.

Returns `None` if SMILES is invalid.

---

### ⚡ Applicability Domain Check

```python
def is_in_applicability_domain(X_train, X_new, threshold_factor=3.0):
```
- Computes the Euclidean distance of the new compound from the mean of training set.
- If distance <= mean + 3 × std deviation, then it is within the applicability domain.
- Returns a boolean array.

---

### 🔍 Streamlit App Interface

```python
st.set_page_config(page_title="Caco-2 Permeability Predictor", layout="centered")
st.title("🧪 Caco-2 Permeability Prediction")
st.markdown("Enter SMILES strings to predict **Caco-2 permeability** using an XGBoost model.")
```
- Sets up the title and layout of the app.
- Displays introductory instructions.

```python
smiles_input = st.text_area("Enter SMILES (one per line):", height=150)
```
- Creates a text area where users can input one or more SMILES.

---

### ⌛ Prediction Workflow

```python
if st.button("Predict"):
```
- Trigger prediction when button is clicked.

#### Inside the Loop:

1. **Parse SMILES**
2. **Calculate descriptors**
3. **Prepare input DataFrame**
4. **Match to training features**
5. **Predict permeability using XGBoost model**
6. **Run AD check**
7. **Store result**

Invalid SMILES are flagged.

---

### 📊 Display Output

```python
st.markdown("### Results:")
st.dataframe(pd.DataFrame(results))
```
- Displays the prediction results as a table with:
  - SMILES
  - Predicted permeability
  - Applicability Domain check (Yes/No)

---

## 🎯 How to Use

### 1. Run the app:
```bash
streamlit run app.py
```

### 2. Enter one or more SMILES strings in the input box:
```
CC(=O)Oc1ccccc1C(=O)O
CCCCN(CC)CC
C1=CC=CC=C1
```

### 3. Click "Predict" to get:
- Caco-2 permeability values
- Applicability Domain status

---

## 🔒 Glossary

- **SMILES**: A line notation for describing chemical structures.
- **Caco-2**: A human epithelial colorectal adenocarcinoma cell line used to study drug permeability.
- **Descriptors**: Numeric values representing molecular properties.
- **Morgan Fingerprints**: A type of circular substructure fingerprint.
- **Applicability Domain**: Defines whether a new molecule is similar to the training data.
- **XGBoost**: A popular and efficient gradient boosting framework.

---

## 💼 License

This project is open source and free to use under the MIT License.

---

## 🚀 Future Improvements

- Add batch upload for SMILES via CSV
- Allow downloading prediction results
- Include molecular visualizations
- Integrate uncertainty estimation

---

## 🙏 Acknowledgements

- [RDKit](https://www.rdkit.org/) for molecular processing
- [XGBoost](https://xgboost.readthedocs.io/) for the regression model
- [Streamlit](https://streamlit.io/) for rapid app development

---

## 📅 Last Updated

April 2025

