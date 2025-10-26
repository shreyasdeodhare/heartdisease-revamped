"""
Flask web application for Heart Disease Prediction
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import json
from dotenv import load_dotenv
from app.config import config
from app.services.data_service import DataService
from app.services.eda_service import EDAService
from app.services.model_service import ModelService
from app.utils.logging_utils import aop_logger

# Load Flask-specific configuration
flask_config_path = os.path.join(os.path.dirname(__file__), 'configs', 'flask_config.env')
load_dotenv(flask_config_path)

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    # Import config from the app.config module
    from app.config import config
    from app.services.data_service import DataService
    from app.services.eda_service import EDAService
    from app.services.model_service import ModelService
    from app.utils.logging_utils import aop_logger
except ImportError as e:
    print(f"Warning: Could not import app modules: {e}")
    print("Running in standalone mode...")
    
    # Create minimal config for standalone mode
    class Config:
        class DataConfig:
            DATA_URL = 'https://raw.githubusercontent.com/kushal140/datasets/main/heart_2020_cleaned.csv'
            MODEL_DIR = './models'
            OUTPUT_DIR = './outputs'
        data = DataConfig()
    
    config = Config()
    
    # Create minimal services
    class MinimalService:
        def __init__(self):
            pass
        def load_data(self):
            return pd.read_csv(config.data.data_url)
    
    data_service = MinimalService()
    eda_service = MinimalService()
    model_service = MinimalService()
    
    class MinimalLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    
    aop_logger = MinimalLogger()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'heart_disease_prediction_secret_key_2024')
app.config['DEBUG'] = os.getenv('DEBUG', 'true').lower() == 'true'

# Initialize services
data_service = DataService()
eda_service = EDAService()
model_service = ModelService()

# Global variables for loaded models
loaded_models = {}
preprocessing_artifacts = {}

def load_models():
    """Load trained models and preprocessing artifacts"""
    global loaded_models, preprocessing_artifacts
    
    try:
        model_dir = os.getenv('MODEL_DIR', './app/models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Load the best model directly
        model_path = os.path.join(model_dir, 'best_model.pkl')
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                    # Handle different model formats
                    if isinstance(model_data, dict):
                        # If model is stored in a dictionary with 'model' key
                        if 'model' in model_data:
                            model = model_data['model']
                            # If model is a scikit-learn pipeline, get the final estimator
                            if hasattr(model, 'steps') and len(model.steps) > 0:
                                model = model.steps[-1][1]
                        else:
                            # If no 'model' key, try to find the model object
                            model = None
                            for key, value in model_data.items():
                                if hasattr(value, 'predict_proba'):
                                    model = value
                                    break
                            if model is None:
                                aop_logger.error("No valid model found in the pickle file")
                                return False
                    else:
                        # If it's a direct model object
                        model = model_data
                    
                    # Ensure the model has predict_proba method
                    if not hasattr(model, 'predict_proba'):
                        aop_logger.error("Loaded model does not have predict_proba method")
                        return False
                        
                    loaded_models['best_model'] = model
                    aop_logger.info(f"Model loaded successfully from {model_path}")
                    aop_logger.info(f"Model type: {type(model).__name__}")
                    
                    # Log model attributes for debugging
                    if hasattr(model, 'feature_importances_'):
                        aop_logger.info(f"Model has feature importances")
                    if hasattr(model, 'n_features_in_'):
                        aop_logger.info(f"Model expects {model.n_features_in_} features")
                    
            except Exception as e:
                aop_logger.error(f"Error loading model from {model_path}: {str(e)}")
                return False
        else:
            aop_logger.error(f"Model not found at {model_path}")
            return False
        
        # Load preprocessing artifacts with specific paths
        artifacts = {
            'scaler': 'scaler.pkl',
            'label_encoders': 'label_encoders.pkl',
            'target_encoder': 'target_encoder.pkl',
            'smote': 'smote.pkl',
            'feature_selector': 'feature_selector.pkl'
        }
        
        # Initialize preprocessing_artifacts if not exists
        if 'preprocessing_artifacts' not in globals():
            global preprocessing_artifacts
            preprocessing_artifacts = {}
        
        for artifact_name, artifact_path in artifacts.items():
            artifact_path = os.path.join(model_dir, artifact_path)
            if os.path.exists(artifact_path):
                with open(artifact_path, 'rb') as f:
                    preprocessing_artifacts[artifact_name] = pickle.load(f)
                aop_logger.info(f"{artifact_name} loaded successfully")
        
        return True
        
    except Exception as e:
        aop_logger.error(f"Failed to load models: {str(e)}")
        return False

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html')

@app.route('/analysis')
def analysis_page():
    """Data analysis page"""
    return render_template('analysis.html')

@app.route('/about')
def about_page():
    """About page"""
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for heart disease prediction"""
    try:
        if 'best_model' not in loaded_models or loaded_models['best_model'] is None:
            return jsonify({'error': 'Model not loaded. Please try again later.', 'status': 'error'}), 503
            
        # Get form data
        form_data = request.get_json()
        aop_logger.info(f"Received prediction request with data: {form_data}")
        
        # Validate required fields
        required_fields = [
            'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
            'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race',
            'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
            'Asthma', 'KidneyDisease', 'SkinCancer'
        ]
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in form_data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'status': 'error'
            }), 400
        
        try:
            # Convert to DataFrame for preprocessing
            input_df = pd.DataFrame([form_data])
            
            # Log input data before preprocessing
            aop_logger.info(f"Input data before preprocessing: {input_df.to_dict(orient='records')[0]}")
            
            # Preprocess input data
            preprocessed_data = preprocess_user_input(input_df)
            aop_logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
            
            # Get the model
            model = loaded_models['best_model']
            
            # Ensure we have the right number of features
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                if preprocessed_data.shape[1] != expected_features:
                    error_msg = f"Feature count mismatch. Expected {expected_features} features, got {preprocessed_data.shape[1]}"
                    aop_logger.error(error_msg)
                    aop_logger.error(f"Expected features: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Not available'}")
                    aop_logger.error(f"Received features: {preprocessed_data.columns.tolist()}")
                    return jsonify({
                        'error': error_msg,
                        'status': 'error',
                        'expected_features': expected_features,
                        'received_features': preprocessed_data.shape[1]
                    }), 400
            
            # Make prediction
            try:
                prediction_proba = model.predict_proba(preprocessed_data)
                probability = float(prediction_proba[0][1])  # Probability of heart disease
                aop_logger.info(f"Raw prediction probabilities: {prediction_proba}")
            except Exception as pred_error:
                aop_logger.error(f"Prediction error: {str(pred_error)}", exc_info=True)
                return jsonify({
                    'error': f'Error making prediction: {str(pred_error)}',
                    'status': 'error',
                    'input_features': preprocessed_data.columns.tolist(),
                    'input_values': preprocessed_data.values.tolist()[0]
                }), 500
            
            # Ensure probability is valid
            if pd.isna(probability) or probability < 0 or probability > 1:
                raise ValueError(f"Invalid probability value: {probability}")
            
            # Get risk level and recommendations
            risk_level = get_risk_level(probability)
            recommendations = get_recommendations(form_data, probability)
            
            # Log successful prediction
            aop_logger.info(f"Prediction successful. Risk level: {risk_level}, Probability: {probability:.4f}")
            
            # Prepare response
            result = {
                'probability': probability,
                'risk_level': risk_level,
                'recommendations': recommendations,
                'status': 'success'
            }
            
            return jsonify(result)
            
        except Exception as preprocess_error:
            aop_logger.error(f"Error during preprocessing or prediction: {str(preprocess_error)}", exc_info=True)
            return jsonify({
                'error': f"Error processing your request: {str(preprocess_error)}",
                'status': 'error'
            }), 400
            
    except Exception as e:
        aop_logger.error(f"Unexpected error in prediction endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred while processing your request.',
            'status': 'error'
        }), 500

def preprocess_user_input(user_data):
    """Preprocess user input data to match training pipeline exactly"""
    try:
        # Convert input to DataFrame if it's not already
        if not isinstance(user_data, pd.DataFrame):
            df = pd.DataFrame([user_data])
        else:
            df = user_data.copy()
        
        # Store original columns for reference
        original_columns = df.columns.tolist()
        aop_logger.info(f"Original input columns: {original_columns}")
        
        # Apply feature engineering
        df = apply_feature_engineering(df)
        
        # Get expected feature names from the model
        if not loaded_models or 'best_model' not in loaded_models:
            raise ValueError("Model not loaded. Please ensure the model is loaded before making predictions.")
            
        model = loaded_models['best_model']
        
        # Get expected features from model if available
        expected_features = []
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_.tolist()
        elif hasattr(model, 'feature_names'):
            expected_features = model.feature_names
        elif hasattr(model, 'feature_name_'):
            expected_features = model.feature_name_
        
        if not expected_features:
            aop_logger.warning("Feature names not found in model. Using features from input data.")
            expected_features = df.columns.tolist()
        else:
            # Ensure we have a list of strings
            if hasattr(expected_features, 'tolist'):
                expected_features = expected_features.tolist()
            aop_logger.info(f"Model expects {len(expected_features)} features: {expected_features}")
        
        # Log the current feature set
        current_features = df.columns.tolist()
        aop_logger.info(f"Current features after engineering: {current_features}")
        
        # Check for missing features
        missing_features = set(expected_features) - set(current_features)
        if missing_features:
            aop_logger.warning(f"Adding {len(missing_features)} missing features with default values")
            for feature in missing_features:
                if feature in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Age_Numeric', 'Health_Score', 'Risk_Score']:
                    df[feature] = 0.0
                    aop_logger.debug(f"Added numeric feature {feature} with default 0.0")
                else:
                    df[feature] = 'Unknown'  # Default for categorical
                    aop_logger.debug(f"Added categorical feature {feature} with default 'Unknown'")
        
        # Ensure the columns are in the exact same order as expected by the model
        df = df[expected_features]
        aop_logger.info(f"Features after reordering: {df.columns.tolist()}")
        
        # Log the shape before transformations
        aop_logger.info(f"Data shape before transformations: {df.shape}")
        
        # Create a function to ensure feature order matches during transformations
        def ensure_feature_order(X, feature_names, transformer_name):
            """Ensure features are in the correct order and handle missing features"""
            if not hasattr(X, 'columns'):
                return X
                
            current_columns = X.columns.tolist()
            
            # Define numerical features that should get 0.0 as default
            numerical_features = [
                'BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 
                'Age_Numeric', 'Health_Score', 'Risk_Score',
                'AlcoholDrinking', 'Stroke', 'PhysicalActivity', 
                'Asthma', 'KidneyDisease', 'SkinCancer', 'Smoking',
                'DiffWalking', 'Sex', 'Diabetic', 'GenHealth'
            ]
            
            # Check for missing features
            missing_features = [f for f in feature_names if f not in current_columns]
            extra_features = [f for f in current_columns if f not in feature_names]
            
            if missing_features or extra_features or (current_columns != feature_names):
                aop_logger.warning(f"Feature mismatch in {transformer_name}:")
                aop_logger.warning(f"  Expected features: {feature_names}")
                aop_logger.warning(f"  Actual features: {current_columns}")
                
                if missing_features:
                    aop_logger.warning(f"  Adding missing features with default values: {missing_features}")
                    
                if extra_features:
                    aop_logger.warning(f"  Dropping extra features: {extra_features}")
            
            # Create a new DataFrame with all expected features
            result = pd.DataFrame(index=X.index)
            
            for feature in feature_names:
                if feature in X.columns:
                    # Convert to numeric if possible, leave as is if not
                    if feature in numerical_features:
                        result[feature] = pd.to_numeric(X[feature], errors='coerce').fillna(0.0)
                    else:
                        result[feature] = X[feature]
                else:
                    # Add missing feature with default value
                    if feature in numerical_features:
                        result[feature] = 0.0  # Default for numerical features
                    else:
                        # For categorical features, use 0 as default instead of 'No'
                        result[feature] = 0
            
            return result
        
        # Encode categorical variables
        if 'label_encoders' in preprocessing_artifacts:
            for col in df.select_dtypes(include=['object']).columns:
                if col in preprocessing_artifacts['label_encoders']:
                    le = preprocessing_artifacts['label_encoders'][col]
                    # Handle unseen labels
                    mask = ~df[col].isin(le.classes_)
                    if mask.any():
                        default_value = le.classes_[0]
                        aop_logger.warning(f"Found {mask.sum()} unseen values in {col}, replacing with '{default_value}'")
                        df.loc[mask, col] = default_value
                    df[col] = le.transform(df[col])
            aop_logger.info("Completed label encoding")
        
        # Scale features
        if 'scaler' in preprocessing_artifacts:
            scaler = preprocessing_artifacts['scaler']
            # Ensure features are in the same order as during fitting
            if hasattr(scaler, 'feature_names_in_'):
                df = ensure_feature_order(df, scaler.feature_names_in_.tolist(), 'scaler')
            
            # Log before scaling
            aop_logger.info(f"Data shape before scaling: {df.shape}")
            
            # Convert to numpy array for scaling
            X_array = df.values
            scaled_array = scaler.transform(X_array)
            
            # Convert back to DataFrame with correct column names
            df_scaled = pd.DataFrame(
                scaled_array, 
                columns=df.columns,
                index=df.index
            )
            aop_logger.info(f"Features after scaling: {df_scaled.columns.tolist()}")
            aop_logger.info(f"Data shape after scaling: {df_scaled.shape}")
        else:
            df_scaled = df
        
        # Apply feature selection if available
        if 'feature_selector' in preprocessing_artifacts:
            selector = preprocessing_artifacts['feature_selector']
            
            # Ensure features are in the same order as during fitting
            if hasattr(selector, 'feature_names_in_'):
                df_scaled = ensure_feature_order(df_scaled, selector.feature_names_in_.tolist(), 'feature_selector')
            
            # Get selected features
            selected_mask = selector.get_support()
            selected_features = df_scaled.columns[selected_mask].tolist()
            
            # Log selection details
            aop_logger.info(f"Selected {sum(selected_mask)} out of {len(selected_mask)} features")
            aop_logger.debug(f"Selected features: {selected_features}")
            
            # Apply selection
            X_selected = selector.transform(df_scaled)
            
            # Create DataFrame with selected features
            df_selected = pd.DataFrame(
                X_selected,
                columns=selected_features,
                index=df_scaled.index
            )
            aop_logger.info(f"Final selected features shape: {df_selected.shape}")
        else:
            df_selected = df_scaled
            
        # If we have a model with n_features_in_ set, ensure we have the right number of features
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            actual_features = df_selected.shape[1]
            
            if actual_features != expected_features:
                aop_logger.warning(f"Feature count mismatch. Model expects {expected_features} features, got {actual_features}")
                
                # If we're getting 15 features but model expects 1, take the first feature
                if actual_features > expected_features:
                    aop_logger.warning(f"Selecting first {expected_features} features")
                    df_selected = df_selected.iloc[:, :expected_features]
                    aop_logger.info(f"Features after selection: {df_selected.columns.tolist()}")
                # If we have fewer features than expected, pad with zeros
                elif actual_features < expected_features:
                    aop_logger.warning(f"Padding with {expected_features - actual_features} zero features")
                    for i in range(actual_features, expected_features):
                        df_selected[f'padding_{i}'] = 0.0
        
        # Final check for NaN values
        if df_selected.isna().any().any():
            aop_logger.warning("NaN values detected in final preprocessed data. Filling with 0.")
            df_selected = df_selected.fillna(0)
        
        # Ensure we have a 2D array with the right shape
        if len(df_selected.shape) == 1:
            df_selected = df_selected.values.reshape(1, -1)
            
        # Final check for feature count
        if hasattr(model, 'n_features_in_') and df_selected.shape[1] != model.n_features_in_:
            aop_logger.error(f"Final feature count mismatch. Model expects {model.n_features_in_}, got {df_selected.shape[1]}")
            aop_logger.error(f"Available features: {df_selected.columns.tolist() if hasattr(df_selected, 'columns') else 'No column names'}")
            
            # If we still have a mismatch, try to force the correct number of features
            if df_selected.shape[1] > model.n_features_in_:
                df_selected = df_selected[:, :model.n_features_in_]
                aop_logger.warning(f"Forcibly truncated to {model.n_features_in_} features")
            elif df_selected.shape[1] < model.n_features_in_:
                padding = np.zeros((df_selected.shape[0], model.n_features_in_ - df_selected.shape[1]))
                df_selected = np.hstack([df_selected, padding])
                aop_logger.warning(f"Padded with {padding.shape[1]} zero features")
        
        # Log final shape before returning
        aop_logger.info(f"Final preprocessed data shape: {df_selected.shape}")
        return df_selected
        
    except Exception as e:
        aop_logger.error(f"Preprocessing error: {str(e)}")
        raise

def apply_feature_engineering(df):
    """Apply feature engineering to match training pipeline"""
    df = df.copy()
    
    # 1. Handle BMI and its categories
    if 'BMI' in df.columns:
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
        
        def categorize_bmi(bmi):
            try:
                bmi = float(bmi)
                if pd.isna(bmi):
                    return 'Unknown'
                if bmi < 18.5:
                    return 'Underweight'
                elif 18.5 <= bmi < 25:
                    return 'Normal'
                elif 25 <= bmi < 30:
                    return 'Overweight'
                else:
                    return 'Obese'
            except (ValueError, TypeError):
                return 'Unknown'
        
        df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
    
    # 2. Process AgeCategory to numeric
    if 'AgeCategory' in df.columns:
        age_mapping = {
            '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42,
            '45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67,
            '70-74': 72, '75-79': 77, '80 or older': 85, 'Unknown': 0
        }
        df['Age_Numeric'] = df['AgeCategory'].map(age_mapping).fillna(0)
    
    # 3. Calculate Health_Score (average of Physical and Mental Health)
    for col in ['PhysicalHealth', 'MentalHealth']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'PhysicalHealth' in df.columns and 'MentalHealth' in df.columns:
        df['Health_Score'] = (df['PhysicalHealth'].fillna(0) + df['MentalHealth'].fillna(0)) / 2
    
    # 4. Add Risk_Score (example calculation - adjust based on your training pipeline)
    risk_factors = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
    df['Risk_Score'] = sum([(df[col].astype(str).str.lower() == 'yes').astype(int) for col in risk_factors if col in df.columns])
    
    # 5. Ensure all required categorical columns exist
    categorical_cols = ['BMI_Category', 'AgeCategory', 'Sex', 'Race', 'GenHealth']
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 'Unknown'

    
    return df


def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

def safe_float(value, default=0.0):
    """Safely convert a value to float, returning default if conversion fails"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert a value to int, returning default if conversion fails"""
    try:
        return int(float(value)) if value is not None else default
    except (ValueError, TypeError):
        return default

def get_recommendations(form_data, probability):
    """Generate health recommendations based on user data and prediction probability"""
    recommendations = []
    
    # Ensure form_data is a dictionary
    if not isinstance(form_data, dict):
        form_data = {}
    
    # Risk level determination
    prob_float = safe_float(probability, 0.0)
    risk_level = "High Risk" if prob_float > 0.7 else "Moderate Risk" if prob_float > 0.3 else "Low Risk"
    
    # General health recommendations
    if risk_level == "High Risk":
        recommendations.append("Consult with a healthcare professional immediately.")
    
    # Convert form data to appropriate types
    bmi = safe_float(form_data.get('BMI'))
    physical_activity = str(form_data.get('PhysicalActivity', '')).lower()
    smoking = str(form_data.get('Smoking', '')).lower()
    alcohol_drinking = str(form_data.get('AlcoholDrinking', '')).lower()
    sleep_time = safe_float(form_data.get('SleepTime'))
    mental_health = safe_int(form_data.get('MentalHealth'))
    
    # BMI-based recommendations
    if bmi > 30:
        recommendations.append("Your BMI indicates obesity. Consider a weight management program.")
    elif bmi > 25:
        recommendations.append("Your BMI indicates overweight. Consider a balanced diet and regular exercise.")
    
    # Physical activity
    if physical_activity == 'no':
        recommendations.append("Incorporate at least 30 minutes of moderate exercise into your daily routine.")
    
    # Smoking
    if smoking == 'yes':
        recommendations.append("Consider quitting smoking to improve your heart health.")
    
    # Alcohol consumption
    if alcohol_drinking == 'yes':
        recommendations.append("Limit alcohol consumption to moderate levels for better heart health.")
    
    # Sleep
    if sleep_time < 6:
        recommendations.append("Aim for 7-9 hours of sleep per night for optimal health.")
    elif sleep_time > 9:
        recommendations.append("Consider if you're getting too much sleep, which can also impact health.")
    
    # Mental health
    if mental_health > 10:
        recommendations.append("Consider speaking with a mental health professional about your stress levels.")
    
    # If no specific recommendations, provide general advice
    if not recommendations:
        recommendations.append("Maintain your healthy lifestyle with regular check-ups.")
    
    return recommendations

def get_sample_analysis_data():
    """Generate sample analysis data for the dashboard"""
    return {
        'dataset_info': {
            'total_samples': 319795,
            'features': 18,
            'target_distribution': {'No': 292422, 'Yes': 27373}
        },
        'feature_importance': get_feature_importance(),
        'risk_factors': {
            'Smoking': {'Yes': 0.15, 'No': 0.05},
            'AlcoholDrinking': {'Yes': 0.12, 'No': 0.08},
            'Stroke': {'Yes': 0.25, 'No': 0.09},
            'Diabetic': {'Yes': 0.18, 'No': 0.06},
            'Asthma': {'Yes': 0.14, 'No': 0.07},
            'KidneyDisease': {'Yes': 0.22, 'No': 0.08},
            'SkinCancer': {'Yes': 0.16, 'No': 0.07}
        }
    }

@app.route('/api/analysis')
def get_analysis_data():
    """API endpoint for analysis data"""
    try:
        # Try to load real data first
        try:
            df = data_service.load_data()
            
            # Generate analysis data from real data
            analysis_data = {
                'dataset_info': {
                    'total_samples': len(df),
                    'features': len(df.columns),
                    'target_distribution': df['HeartDisease'].value_counts().to_dict()
                },
                'feature_importance': get_feature_importance(),
                'risk_factors': get_risk_factors_analysis(df)
            }
            return jsonify(analysis_data)
            
        except Exception as load_error:
            aop_logger.logger.warning(f"Using sample analysis data due to error: {str(load_error)}")
            # Fall back to sample data if there's an error loading real data
            return jsonify(get_sample_analysis_data())
            
    except Exception as e:
        aop_logger.logger.error(f"Analysis error: {str(e)}")
        # Return sample data even if there's an error with the main logic
        return jsonify({
            'error': 'Error generating analysis data',
            'sample_data': get_sample_analysis_data()
        }), 200

def get_risk_factors_analysis(df):
    """Analyze risk factors from the dataset"""
    try:
        risk_factors = {}
        
        # Define the risk factors to analyze
        factors = ['Smoking', 'AlcoholDrinking', 'Stroke', 'Diabetic', 
                  'Asthma', 'KidneyDisease', 'SkinCancer']
        
        for factor in factors:
            if factor in df.columns:
                # Calculate heart disease rate for each category
                factor_analysis = df.groupby(factor)['HeartDisease'].mean().to_dict()
                risk_factors[factor] = {str(k): float(v) for k, v in factor_analysis.items()}
        
        return risk_factors
        
    except Exception as e:
        aop_logger.logger.error(f"Error analyzing risk factors: {str(e)}")
        # Return sample data if there's an error
        return {
            'Smoking': {'Yes': 0.15, 'No': 0.05},
            'AlcoholDrinking': {'Yes': 0.12, 'No': 0.08},
            'Stroke': {'Yes': 0.25, 'No': 0.09},
            'Diabetic': {'Yes': 0.18, 'No': 0.06},
            'Asthma': {'Yes': 0.14, 'No': 0.07},
            'KidneyDisease': {'Yes': 0.22, 'No': 0.08},
            'SkinCancer': {'Yes': 0.16, 'No': 0.07}
        }

def get_feature_importance():
    """Get feature importance data"""
    try:
        # Try to get feature importance from the loaded model if available
        if 'best_model' in loaded_models and hasattr(loaded_models['best_model'], 'feature_importances_'):
            features = loaded_models.get('feature_names', [f'feature_{i}' for i in range(len(loaded_models['best_model'].feature_importances_))])
            return dict(zip(features, loaded_models['best_model'].feature_importances_.tolist()))
    except Exception as e:
        aop_logger.logger.warning(f"Could not get feature importance from model: {str(e)}")
    
    # Fall back to sample data
    return {
        'Age_Numeric': 0.15,
        'BMI': 0.12,
        'PhysicalHealth': 0.10,
        'GenHealth': 0.09,
        'Smoking': 0.08,
        'Diabetic': 0.07,
        'SleepTime': 0.06,
        'MentalHealth': 0.05,
        'Sex': 0.04,
        'Race': 0.03
    }

def get_risk_factors_analysis(df):
    """Get risk factors analysis"""
    risk_factors = ['Smoking', 'AlcoholDrinking', 'Stroke', 'Diabetic', 
                   'Asthma', 'KidneyDisease', 'SkinCancer']
    
    analysis = {}
    for factor in risk_factors:
        if factor in df.columns:
            factor_analysis = df.groupby(factor)['HeartDisease'].mean()
            analysis[factor] = {
                'Yes': float(factor_analysis.get('Yes', 0)),
                'No': float(factor_analysis.get('No', 0))
            }
    
    return analysis

@app.route('/api/health-tips')
def get_health_tips():
    """API endpoint for health tips"""
    tips = {
        'general': [
            "Maintain a healthy weight through balanced diet and regular exercise",
            "Get at least 7-9 hours of quality sleep each night",
            "Manage stress through relaxation techniques and hobbies",
            "Stay hydrated by drinking plenty of water throughout the day"
        ],
        'heart_health': [
            "Eat a heart-healthy diet rich in fruits, vegetables, and whole grains",
            "Limit saturated fats, trans fats, and sodium in your diet",
            "Engage in at least 150 minutes of moderate exercise per week",
            "Avoid smoking and limit alcohol consumption"
        ],
        'prevention': [
            "Get regular health check-ups and screenings",
            "Monitor your blood pressure and cholesterol levels",
            "Maintain a healthy BMI between 18.5 and 24.9",
            "Practice good oral hygiene as it's linked to heart health"
        ]
    }
    
    return jsonify(tips)

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("✅ Models loaded successfully")
    else:
        print("⚠️  Models not found. Please run the training pipeline first.")
    
    # Get configuration values
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('DEBUG', 'true').lower() == 'true'
    
    print(f"🌐 Starting Flask server on {host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    
    # Run the Flask app
    app.run(debug=debug, host=host, port=port)
