import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination.default import DefaultMultiObjectiveTermination
import optuna
import warnings
import io
import base64

warnings.filterwarnings('ignore')
np.random.seed(42)

# Set page configuration
st.set_page_config(page_title="Caco2 Permeability Prediction", layout="wide")

# Title and description
st.title("Caco2 Permeability Prediction")
st.markdown("""
This application predicts the Caco2 permeability of molecules using machine learning. 
Upload data or enter SMILES directly for prediction.
""")

# Descriptor + Fingerprint Calculation
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

# NSGA-II Feature Selection Problem
class FeatureSelectionProblem(Problem):
    def __init__(self, X_train, y_train, X_val, y_val, n_features):
        super().__init__(n_var=n_features, n_obj=2, n_constr=0, xl=0, xu=1)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
    def _evaluate(self, x, out, *args, **kwargs):
        n_samples = x.shape[0]
        f1 = np.zeros(n_samples)
        f2 = np.zeros(n_samples)
        
        for i in range(n_samples):
            selected = x[i, :] > 0.5
            if not np.any(selected):
                f1[i] = 9999
                f2[i] = 0
                continue
                
            n_selected = np.sum(selected)
            f2[i] = n_selected / len(selected)
            
            try:
                X_train_selected = self.X_train.iloc[:, selected]
                X_val_selected = self.X_val.iloc[:, selected]
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                model.fit(X_train_selected, self.y_train)
                y_pred = model.predict(X_val_selected)
                f1[i] = np.sqrt(mean_squared_error(self.y_val, y_pred))
            except:
                f1[i] = 9999
                
        out["F"] = np.column_stack([f1, f2])

# Applicability Domain
def is_in_applicability_domain(X_train, X_new, threshold_factor=3.0):
    mean_train = np.mean(X_train, axis=0)
    distances_train = np.sqrt(np.sum((X_train - mean_train)**2, axis=1))
    threshold = np.mean(distances_train) + threshold_factor * np.std(distances_train)
    distances_new = np.sqrt(np.sum((X_new - mean_train)**2, axis=1))
    return distances_new <= threshold

# Prediction function
def predict_caco2(smiles_list, model, selected_features, X_train):
    descriptor_dicts = []
    valid_smiles = []
    invalid_smiles = []
    
    for smiles in smiles_list:
        if not smiles.strip():
            continue
        desc_dict = calculate_descriptors(smiles.strip())
        if desc_dict is not None:
            descriptor_dicts.append(desc_dict)
            valid_smiles.append(smiles.strip())
        else:
            invalid_smiles.append(smiles.strip())
            
    if not descriptor_dicts:
        return {"error": "No valid molecules found", "invalid_smiles": invalid_smiles}
    
    desc_df = pd.DataFrame(descriptor_dicts)
    all_columns = X_train.columns
    X_pred = pd.DataFrame(0, index=range(len(desc_df)), columns=all_columns)
    
    for col in desc_df.columns:
        if col in X_pred.columns:
            X_pred[col] = desc_df[col]
            
    X_pred_selected = X_pred[selected_features]
    y_pred = model.predict(X_pred_selected)
    in_domain = is_in_applicability_domain(X_train[selected_features], X_pred_selected)
    
    return {
        "SMILES": valid_smiles,
        "Prediction": y_pred.tolist(),
        "In_Domain": in_domain.tolist(),
        "invalid_smiles": invalid_smiles
    }

# Function to create a download link for dataframe
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# Function to train model with feedback
def train_model(df_train, df_valid, df_test, external_data=None):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Calculating descriptors and fingerprints...")
    progress_bar.progress(10)
    
    descriptor_dicts = []
    valid_indices = []
    
    for idx, smiles in enumerate(df_train['Drug']):
        desc = calculate_descriptors(smiles)
        if desc is not None:
            descriptor_dicts.append(desc)
            valid_indices.append(idx)
            
    desc_df = pd.DataFrame(descriptor_dicts).reset_index(drop=True)
    y_all = df_train.iloc[valid_indices]['Y'].reset_index(drop=True)
    
    desc_df = desc_df.dropna(axis=1, thresh=0.95 * len(desc_df))
    desc_df = desc_df.fillna(desc_df.median())
    
    status_text.text("Performing feature selection...")
    progress_bar.progress(30)
    
    var_thresh = VarianceThreshold(threshold=0.01)
    X_var = var_thresh.fit_transform(desc_df)
    X = pd.DataFrame(X_var, columns=desc_df.columns[var_thresh.get_support()])
    y = y_all
    
    X_train = X.loc[df_train.index.intersection(X.index)]
    X_valid = X.loc[df_valid.index.intersection(X.index)]
    X_test = X.loc[df_test.index.intersection(X.index)]
    
    y_train = y.loc[X_train.index]
    y_valid = y.loc[X_valid.index]
    y_test = y.loc[X_test.index]
    
    # Feature selection with NSGA-II
    status_text.text("Running NSGA-II for feature selection...")
    progress_bar.progress(50)
    
    problem = FeatureSelectionProblem(X_train, y_train, X_valid, y_valid, X_train.shape[1])
    algorithm = NSGA2(
        pop_size=50,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True
    )
    
    termination = DefaultMultiObjectiveTermination(
        xtol=0.001,
        cvtol=0.0001,
        ftol=0.0001,
        period=20,
        n_max_gen=50,
        n_max_evals=5000
    )
    
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=False
    )
    
    selected_solution = result.X[np.argmin(result.F[:, 0])]
    selected_features = X_train.columns[selected_solution > 0.5].tolist()
    
    status_text.text("Running hyperparameter optimization...")
    progress_bar.progress(70)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        X_sel = X_train[selected_features]
        
        for train_idx, val_idx in kf.split(X_sel):
            model = XGBRegressor(**params, random_state=42)
            model.fit(X_sel.iloc[train_idx], y_train.iloc[train_idx])
            pred = model.predict(X_sel.iloc[val_idx])
            scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], pred)))
            
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    
    status_text.text("Training final model...")
    progress_bar.progress(90)
    
    final_model = XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train[selected_features], y_train)
    
    y_train_pred = final_model.predict(X_train[selected_features])
    y_valid_pred = final_model.predict(X_valid[selected_features])
    y_test_pred = final_model.predict(X_test[selected_features])
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_r2 = r2_score(y_train, y_train_pred)
    valid_r2 = r2_score(y_valid, y_valid_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    external_metrics = None
    if external_data is not None:
        # Process external validation data
        ext_desc_dicts = []
        ext_valid_indices = []
        
        for idx, smiles in enumerate(external_data['Drug']):
            desc = calculate_descriptors(smiles)
            if desc is not None:
                ext_desc_dicts.append(desc)
                ext_valid_indices.append(idx)
                
        ext_desc_df = pd.DataFrame(ext_desc_dicts).reset_index(drop=True)
        ext_y = external_data.iloc[ext_valid_indices]['Y'].reset_index(drop=True)
        
        # Fill any missing columns with zeros
        for col in X.columns:
            if col not in ext_desc_df.columns:
                ext_desc_df[col] = 0
                
        # Select only the columns that are in X
        ext_desc_df = ext_desc_df[X.columns]
        
        # Fill NaN values with the median from the training set
        for col in ext_desc_df.columns:
            if ext_desc_df[col].isna().any():
                ext_desc_df[col] = ext_desc_df[col].fillna(X[col].median())
        
        ext_pred = final_model.predict(ext_desc_df[selected_features])
        ext_rmse = np.sqrt(mean_squared_error(ext_y, ext_pred))
        ext_r2 = r2_score(ext_y, ext_pred)
        
        external_metrics = {
            'rmse': ext_rmse,
            'r2': ext_r2,
            'predictions': pd.DataFrame({
                'Observed': ext_y,
                'Predicted': ext_pred
            })
        }
    
    progress_bar.progress(100)
    status_text.text("Model training complete!")
    
    return {
        'model': final_model,
        'selected_features': selected_features,
        'X_train': X_train,
        'metrics': {
            'train': {'rmse': train_rmse, 'r2': train_r2},
            'valid': {'rmse': valid_rmse, 'r2': valid_r2},
            'test': {'rmse': test_rmse, 'r2': test_r2}
        },
        'external_metrics': external_metrics
    }

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Train Model", "Make Predictions", "About"])

if page == "Home":
    st.header("Welcome to the Caco2 Permeability Prediction App")
    st.markdown("""
    This application helps predict Caco2 permeability of chemical compounds using machine learning.
    
    ### Features:
    - Train custom models with your data
    - Make predictions for new compounds
    - Evaluate model performance
    - Check applicability domain
    
    ### Get Started:
    1. Navigate to "Train Model" to create a custom model
    2. Or go to "Make Predictions" to use a pre-trained model
    
    ### About Caco2 Permeability:
    Caco2 permeability is an important parameter in drug discovery and development, 
    as it helps predict intestinal absorption of compounds.
    """)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/RDKit_2D.svg/1200px-RDKit_2D.svg.png", width=300)

elif page == "Train Model":
    st.header("Train Custom Caco2 Prediction Model")
    
    # Option to use built-in dataset or upload custom data
    data_option = st.radio("Choose data source:", ["Use TDC Caco2_Wang dataset", "Upload custom dataset"])
    
    if data_option == "Use TDC Caco2_Wang dataset":
        st.info("Using the TDC Caco2_Wang dataset. This will download data from the TDC package.")
        use_external = st.checkbox("Include external validation dataset")
        
        if use_external:
            external_file = st.file_uploader("Upload external validation CSV (must have 'Drug' and 'Y' columns)", type=["csv"])
        else:
            external_file = None
            
        if st.button("Train Model"):
            try:
                from tdc.single_pred import ADME
                data = ADME(name='Caco2_Wang')
                df = data.get_data()
                split = data.get_split(method='scaffold')
                
                if external_file:
                    external_data = pd.read_csv(external_file)
                    if 'Drug' not in external_data.columns or 'Y' not in external_data.columns:
                        st.error("External data must contain 'Drug' and 'Y' columns")
                    else:
                        st.success("External validation data loaded successfully")
                else:
                    external_data = None
                
                result = train_model(df, split['train'], split['valid'], split['test'], external_data)
                
                # Save model
                if 'model_results' not in st.session_state:
                    st.session_state.model_results = {}
                    
                st.session_state.model_results = result
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Training Set")
                    st.metric("RMSE", f"{result['metrics']['train']['rmse']:.4f}")
                    st.metric("R²", f"{result['metrics']['train']['r2']:.4f}")
                
                with col2:
                    st.subheader("Validation Set")
                    st.metric("RMSE", f"{result['metrics']['valid']['rmse']:.4f}")
                    st.metric("R²", f"{result['metrics']['valid']['r2']:.4f}")
                
                with col3:
                    st.subheader("Test Set")
                    st.metric("RMSE", f"{result['metrics']['test']['rmse']:.4f}")
                    st.metric("R²", f"{result['metrics']['test']['r2']:.4f}")
                
                if result['external_metrics']:
                    st.subheader("External Validation Set")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"{result['external_metrics']['rmse']:.4f}")
                    with col2:
                        st.metric("R²", f"{result['external_metrics']['r2']:.4f}")
                    
                    st.subheader("External Validation Predictions")
                    st.dataframe(result['external_metrics']['predictions'])
                    
                    # Download predictions
                    st.markdown(
                        get_table_download_link(
                            result['external_metrics']['predictions'],
                            "external_validation_predictions",
                            "Download external validation predictions"
                        ),
                        unsafe_allow_html=True
                    )
                
                # Plot feature importance
                st.subheader("Feature Importance")
                importances = result['model'].feature_importances_
                indices = np.argsort(importances)[-10:]  # Top 10 features
                
                plt.figure(figsize=(10, 6))
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)), [result['selected_features'][i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title('Top 10 Features by Importance')
                st.pyplot(plt)
                
                # Save button for model
                if st.button("Save Model"):
                    joblib.dump(result['model'], 'caco2_model.joblib')
                    joblib.dump(result['selected_features'], 'selected_features.joblib')
                    joblib.dump(result['X_train'], 'X_train.joblib')
                    st.success("Model saved successfully!")
                    
            except Exception as e:
                st.error(f"An error occurred during model training: {str(e)}")
    
    else:
        st.subheader("Upload your training dataset")
        st.markdown("""
        Please upload a CSV file with at least two columns:
        - 'Drug': SMILES strings of compounds
        - 'Y': Caco2 permeability values
        """)
        
        train_file = st.file_uploader("Upload training CSV", type=["csv"])
        
        if train_file:
            df_custom = pd.read_csv(train_file)
            
            if 'Drug' not in df_custom.columns or 'Y' not in df_custom.columns:
                st.error("Data must contain 'Drug' and 'Y' columns")
            else:
                st.success(f"Data loaded successfully with {len(df_custom)} compounds")
                st.dataframe(df_custom.head())
                
                # Split options
                split_method = st.selectbox(
                    "Choose split method",
                    ["Random (80/10/10)", "Custom Ratio", "Upload separate validation/test sets"]
                )
                
                if split_method == "Random (80/10/10)":
                    from sklearn.model_selection import train_test_split
                    
                    if st.button("Train Model"):
                        train_idx, temp_idx = train_test_split(df_custom.index, test_size=0.2, random_state=42)
                        valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
                        
                        train_data = df_custom.loc[train_idx]
                        valid_data = df_custom.loc[valid_idx]
                        test_data = df_custom.loc[test_idx]
                        
                        result = train_model(train_data, valid_data, test_data)
                        
                        if 'model_results' not in st.session_state:
                            st.session_state.model_results = {}
                            
                        st.session_state.model_results = result
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.subheader("Training Set")
                            st.metric("RMSE", f"{result['metrics']['train']['rmse']:.4f}")
                            st.metric("R²", f"{result['metrics']['train']['r2']:.4f}")
                        
                        with col2:
                            st.subheader("Validation Set")
                            st.metric("RMSE", f"{result['metrics']['valid']['rmse']:.4f}")
                            st.metric("R²", f"{result['metrics']['valid']['r2']:.4f}")
                        
                        with col3:
                            st.subheader("Test Set")
                            st.metric("RMSE", f"{result['metrics']['test']['rmse']:.4f}")
                            st.metric("R²", f"{result['metrics']['test']['r2']:.4f}")
                
                elif split_method == "Custom Ratio":
                    col1, col2 = st.columns(2)
                    with col1:
                        train_ratio = st.slider("Training set %", 50, 90, 70)
                    with col2:
                        valid_ratio = st.slider("Validation set %", 5, 30, 15)
                    
                    test_ratio = 100 - train_ratio - valid_ratio
                    st.info(f"Test set: {test_ratio}%")
                    
                    if st.button("Train Model"):
                        from sklearn.model_selection import train_test_split
                        
                        train_size = train_ratio / 100
                        valid_size = valid_ratio / (100 - train_ratio)
                        
                        train_idx, temp_idx = train_test_split(df_custom.index, test_size=(1-train_size), random_state=42)
                        valid_idx, test_idx = train_test_split(temp_idx, test_size=(1-valid_size), random_state=42)
                        
                        train_data = df_custom.loc[train_idx]
                        valid_data = df_custom.loc[valid_idx]
                        test_data = df_custom.loc[test_idx]
                        
                        result = train_model(train_data, valid_data, test_data)
                        
                        if 'model_results' not in st.session_state:
                            st.session_state.model_results = {}
                            
                        st.session_state.model_results = result
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.subheader("Training Set")
                            st.metric("RMSE", f"{result['metrics']['train']['rmse']:.4f}")
                            st.metric("R²", f"{result['metrics']['train']['r2']:.4f}")
                        
                        with col2:
                            st.subheader("Validation Set")
                            st.metric("RMSE", f"{result['metrics']['valid']['rmse']:.4f}")
                            st.metric("R²", f"{result['metrics']['valid']['r2']:.4f}")
                        
                        with col3:
                            st.subheader("Test Set")
                            st.metric("RMSE", f"{result['metrics']['test']['rmse']:.4f}")
                            st.metric("R²", f"{result['metrics']['test']['r2']:.4f}")
                
                else:  # Upload separate validation/test sets
                    valid_file = st.file_uploader("Upload validation CSV", type=["csv"])
                    test_file = st.file_uploader("Upload test CSV", type=["csv"])
                    
                    if valid_file and test_file:
                        valid_data = pd.read_csv(valid_file)
                        test_data = pd.read_csv(test_file)
                        
                        if ('Drug' not in valid_data.columns or 'Y' not in valid_data.columns or
                            'Drug' not in test_data.columns or 'Y' not in test_data.columns):
                            st.error("All datasets must contain 'Drug' and 'Y' columns")
                        else:
                            st.success("All datasets loaded successfully")
                            
                            if st.button("Train Model"):
                                result = train_model(df_custom, valid_data, test_data)
                                
                                if 'model_results' not in st.session_state:
                                    st.session_state.model_results = {}
                                    
                                st.session_state.model_results = result
                                
                                # Display metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.subheader("Training Set")
                                    st.metric("RMSE", f"{result['metrics']['train']['rmse']:.4f}")
                                    st.metric("R²", f"{result['metrics']['train']['r2']:.4f}")
                                
                                with col2:
                                    st.subheader("Validation Set")
                                    st.metric("RMSE", f"{result['metrics']['valid']['rmse']:.4f}")
                                    st.metric("R²", f"{result['metrics']['valid']['r2']:.4f}")
                                
                                with col3:
                                    st.subheader("Test Set")
                                    st.metric("RMSE", f"{result['metrics']['test']['rmse']:.4f}")
                                    st.metric("R²", f"{result['metrics']['test']['r2']:.4f}")

elif page == "Make Predictions":
    st.header("Predict Caco2 Permeability")
    
    # Options for model selection
    model_option = st.radio(
        "Choose model source:",
        ["Use session model", "Upload saved model", "Use built-in model"]
    )
    
    model = None
    selected_features = None
    X_train = None
    
    if model_option == "Use session model":
        if 'model_results' in st.session_state and st.session_state.model_results:
            model = st.session_state.model_results['model']
            selected_features = st.session_state.model_results['selected_features']
            X_train = st.session_state.model_results['X_train']
            st.success("Using model from current session")
        else:
            st.error("No model found in current session. Please train a model first or choose another option.")
    
    elif model_option == "Upload saved model":
        model_file = st.file_uploader("Upload model file (.joblib)", type=["joblib"])
        features_file = st.file_uploader("Upload selected features file (.joblib)", type=["joblib"])
        x_train_file = st.file_uploader("Upload X_train file (.joblib)", type=["joblib"])
        
        if model_file and features_file and x_train_file:
            try:
                model = joblib.load(model_file)
                selected_features = joblib.load(features_file)
                X_train = joblib.load(x_train_file)
                st.success("Model loaded successfully")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    elif model_option == "Use built-in model":
        st.info("This option uses a pre-trained model based on the Caco2_Wang dataset")
        
        # In a real application, you would load a pre-trained model here
        # For demonstration, we'll simulate having a built-in model
        st.warning("Built-in model not available in this demo. Please train a model first.")
    
    # Input method selection
    if model and selected_features and X_train:
        input_method = st.radio(
            "Choose input method:",
            ["Enter SMILES", "Upload CSV file"]
        )
        
        if input_method == "Enter SMILES":
            smiles_input = st.text_area("Enter SMILES (one per line):", height=150)
            
            if st.button("Predict") and smiles_input:
                smiles_list = smiles_input.strip().split('\n')
                results = predict_caco2(smiles_list, model, selected_features, X_train)
                
                if "error" in results:
                    st.error(results["error"])
                    if results["invalid_smiles"]:
                        st.warning(f"Invalid SMILES: {', '.join(results['invalid_smiles'])}")
                else:
                    # Create result dataframe
                    result_df = pd.DataFrame({
                    "SMILES": results["SMILES"],
                    "Prediction": results["Prediction"],
                    "In_Domain": results["In_Domain"]
                    })
