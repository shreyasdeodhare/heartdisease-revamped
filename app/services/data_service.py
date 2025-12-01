

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

from app.config import config
from app.utils.logging_utils import data_logger, aop_logger
from app.utils.error_handling import error_handler, data_validator


class DataService:
    """Service for data processing operations"""
    
    def __init__(self):
        self.logger = aop_logger.logger
        self.scaler = None
        self.label_encoders = {}
        self.target_encoder = None
        self.smote = None
        self.feature_selector = None
        
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def load_data(self, data_url: str = None) -> pd.DataFrame:
        """Load data from URL or local file"""
        data_url = data_url or config.data.data_url
        
        try:
            self.logger.info(f"Starting data download from: {data_url}")
            self.logger.info("This may take a few moments for large datasets...")
            
            df = pd.read_csv(data_url)
            
            self.logger.info(f"Data download completed. Shape: {df.shape}")
            self.logger.info(f"Columns: {list(df.columns)}")
            self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Validate data
            self.logger.info("Validating downloaded data...")
            data_validator.validate_dataframe(df)
            data_logger.log_data_loading(data_url, df.shape)
            
            self.logger.info("Data loading and validation completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset"""
        self.logger.info("Starting data preprocessing")
        
        
        df_processed = df.copy()
        
       
        df_processed = self._engineer_features(df_processed)
        
        
        df_processed = self._handle_missing_values(df_processed)
        
        self.logger.info(f"Preprocessing completed. Shape: {df_processed.shape}")
        return df_processed
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        self.logger.info("Starting feature engineering...")
        self.logger.info(f"Input data shape: {df.shape}")
        
       
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif 18.5 <= bmi < 25:
                return 'Normal'
            elif 25 <= bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'
        
        df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
        
        
        age_mapping = {
            '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42,
            '45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67,
            '70-74': 72, '75-79': 77, '80 or older': 85
        }
        df['Age_Numeric'] = df['AgeCategory'].map(age_mapping)
        
        df['Health_Score'] = (df['PhysicalHealth'] + df['MentalHealth']) / 2
        
        risk_factors = ['Smoking', 'AlcoholDrinking', 'Stroke', 'Diabetic', 
                       'Asthma', 'KidneyDisease', 'SkinCancer']
        df['Risk_Score'] = 0
        for factor in risk_factors:
            if factor in df.columns:
                df['Risk_Score'] += (df[factor] == 'Yes').astype(int)
        
        self.logger.info(f"Feature engineering completed. New shape: {df.shape}")
        self.logger.info("New features created: BMI_Category, Age_Numeric, Health_Score, Risk_Score")
        data_logger.log_feature_engineering(['BMI_Category', 'Age_Numeric', 'Health_Score', 'Risk_Score'])
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        missing_values = df.isnull().sum()
        if missing_values.any():
            self.logger.warning(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        self.logger.info("Starting categorical variable encoding...")
        self.logger.info(f"Input shape: {df.shape}")
        
        df_encoded = df.copy()
        
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        self.logger.info(f"Found {len(categorical_columns)} categorical columns: {categorical_columns}")
        
        self.logger.info("Applying label encoding to categorical features...")
        for i, col in enumerate(categorical_columns):
            if col != 'HeartDisease':
                self.logger.info(f"Encoding column {i+1}/{len(categorical_columns)}: {col}")
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                self.logger.info(f"Encoded {col} with {len(le.classes_)} classes: {le.classes_}")
        
        self.logger.info("Encoding target variable: HeartDisease")
        self.target_encoder = LabelEncoder()
        df_encoded['HeartDisease'] = self.target_encoder.fit_transform(df_encoded['HeartDisease'])
        self.logger.info(f"Target variable encoded with classes: {self.target_encoder.classes_}")
        
        self.logger.info(f"Categorical encoding completed. Final shape: {df_encoded.shape}")
        return df_encoded
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using MinMaxScaler"""
        self.logger.info("Scaling features")
        
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        self.logger.info(f"Features scaled. Shape: {X_scaled.shape}")
        return X_scaled
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using multiple techniques"""
        self.logger.info("Starting class imbalance handling...")
        self.logger.info(f"Input data shape: {X.shape}")
        
        # Check class distribution
        class_counts = y.value_counts()
        self.logger.info(f"Original class distribution: {class_counts.to_dict()}")
        data_logger.log_class_balance(class_counts.to_dict())
        
        # Apply SMOTE
        self.logger.info("Applying SMOTE for class balancing...")
        self.smote = SMOTE(random_state=config.model.smote_random_state, k_neighbors=config.model.smote_k_neighbors)
        X_balanced, y_balanced = self.smote.fit_resample(X, y)
        
        new_class_counts = pd.Series(y_balanced).value_counts()
        self.logger.info(f"SMOTE completed. New shape: {X_balanced.shape}")
        self.logger.info(f"New class distribution: {new_class_counts.to_dict()}")
        return X_balanced, y_balanced
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using multiple techniques"""
        self.logger.info(f"Starting feature selection on {X.shape[0]} samples...")
        
        # Use subset for feature selection to speed up process
        if X.shape[0] > 50000:
            self.logger.info("Large dataset detected. Using sample for feature selection...")
            sample_size = 50000
            sample_idx = np.random.choice(X.index, size=sample_size, replace=False)
            X_sample = X.loc[sample_idx]
            y_sample = y.loc[sample_idx]
            self.logger.info(f"Using sample of {sample_size} for feature selection")
        else:
            X_sample = X
            y_sample = y
        
        # Method 1: Statistical Feature Selection (F-test)
        self.logger.info("Running statistical feature selection...")
        selector_f = SelectKBest(score_func=f_classif, k=config.model.n_features_select)
        selector_f.fit(X_sample, y_sample)
        selected_features_f = X.columns[selector_f.get_support()].tolist()
        self.logger.info(f"Statistical selection completed: {len(selected_features_f)} features")
        
        # Method 2: Recursive Feature Elimination (simplified)
        self.logger.info("Running RFE feature selection...")
        rf_selector = RandomForestClassifier(n_estimators=10, random_state=config.model.random_state)
        rfe = RFE(estimator=rf_selector, n_features_to_select=config.model.n_features_select)
        rfe.fit(X_sample, y_sample)
        selected_features_rfe = X.columns[rfe.get_support()].tolist()
        self.logger.info(f"RFE selection completed: {len(selected_features_rfe)} features")
        
        # Use RFE selected features
        X_final = X[selected_features_rfe]
        self.feature_selector = rfe
        
        self.logger.info(f"Feature selection completed. Selected {len(selected_features_rfe)} features: {selected_features_rfe}")
        return X_final, selected_features_rfe
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        self.logger.info("Splitting data into train and test sets")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.model.test_size,
            random_state=config.model.random_state,
            stratify=y
        )
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive data summary"""
        summary = {
            'shape': df.shape,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'target_distribution': df['HeartDisease'].value_counts().to_dict() if 'HeartDisease' in df.columns else None
        }
        
        return summary
    
    def save_preprocessing_artifacts(self, output_dir: str) -> None:
        """Save preprocessing artifacts"""
        import pickle
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scaler
        if self.scaler:
            with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save label encoders
        if self.label_encoders:
            with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
                pickle.dump(self.label_encoders, f)
        
        # Save target encoder
        if self.target_encoder:
            with open(os.path.join(output_dir, 'target_encoder.pkl'), 'wb') as f:
                pickle.dump(self.target_encoder, f)
        
        # Save SMOTE
        if self.smote:
            with open(os.path.join(output_dir, 'smote.pkl'), 'wb') as f:
                pickle.dump(self.smote, f)
        
        # Save feature selector
        if self.feature_selector:
            with open(os.path.join(output_dir, 'feature_selector.pkl'), 'wb') as f:
                pickle.dump(self.feature_selector, f)
        
        self.logger.info(f"Preprocessing artifacts saved to {output_dir}")
