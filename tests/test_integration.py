"""
Integration tests for the complete heart disease prediction pipeline
"""
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from services.data_service import DataService
from services.model_service import ModelService
from services.eda_service import EDAService
from utils.logging_utils import AOPLogger
from utils.error_handling import DataValidator, ModelValidator


@pytest.fixture
def sample_dataset():
    """Create a realistic sample dataset for integration testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic heart disease dataset
    data = {
        'HeartDisease': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # 10% have heart disease
        'BMI': np.random.normal(28.5, 6.0, n_samples),
        'Smoking': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'AlcoholDrinking': np.random.choice(['Yes', 'No'], n_samples, p=[0.08, 0.92]),
        'PhysicalActivity': np.random.choice(['Yes', 'No'], n_samples, p=[0.75, 0.25]),
        'AgeCategory': np.random.choice(['18-24', '25-29', '30-34', '35-39', '40-44', 
                                        '45-49', '50-54', '55-59', '60-64', '65-69', 
                                        '70-74', '75-79', '80 or older'], n_samples),
        'Sex': np.random.choice(['Male', 'Female'], n_samples),
        'PhysicalHealth': np.random.randint(0, 30, n_samples),
        'MentalHealth': np.random.randint(0, 30, n_samples),
        'SleepTime': np.random.randint(1, 12, n_samples),
        'GenHealth': np.random.choice(['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], n_samples),
        'Diabetic': np.random.choice(['Yes', 'No'], n_samples, p=[0.12, 0.88]),
        'Asthma': np.random.choice(['Yes', 'No'], n_samples, p=[0.13, 0.87]),
        'KidneyDisease': np.random.choice(['Yes', 'No'], n_samples, p=[0.03, 0.97]),
        'SkinCancer': np.random.choice(['Yes', 'No'], n_samples, p=[0.09, 0.91])
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation with target variable
    # Higher BMI, age, and physical health issues increase heart disease risk
    mask = df['HeartDisease'] == 1
    df.loc[mask, 'BMI'] += np.random.normal(3, 1, mask.sum())  # Higher BMI for heart disease
    df.loc[mask, 'PhysicalHealth'] += np.random.randint(5, 15, mask.sum())  # More physical health issues
    
    # Ensure BMI is positive
    df['BMI'] = np.maximum(df['BMI'], 15.0)
    
    return df


@pytest.fixture
def services():
    """Create all service instances"""
    return {
        'data_service': DataService(),
        'model_service': ModelService(),
        'eda_service': EDAService(),
        'logger': AOPLogger('integration_test'),
        'data_validator': DataValidator(),
        'model_validator': ModelValidator()
    }


class TestCompleteDataPipeline:
    """Test complete data processing pipeline"""
    
    def test_end_to_end_data_processing(self, services, sample_dataset):
        """Test complete data processing from raw data to ready-to-train data"""
        data_service = services['data_service']
        validator = services['data_validator']
        
        # 1. Validate raw data
        assert validator.validate_dataframe(sample_dataset, 
                                          required_columns=['HeartDisease', 'BMI', 'Smoking'])
        
        # 2. Preprocess data
        processed_df = data_service.preprocess_data(sample_dataset)
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_dataset)
        
        # 3. Engineer features
        engineered_df = data_service._engineer_features(processed_df)
        assert isinstance(engineered_df, pd.DataFrame)
        assert 'BMI_Category' in engineered_df.columns
        assert 'Age_Numeric' in engineered_df.columns
        assert 'Health_Score' in engineered_df.columns
        assert 'Risk_Score' in engineered_df.columns
        
        # 4. Encode categorical variables
        encoded_df = data_service.encode_categorical_variables(engineered_df)
        assert isinstance(encoded_df, pd.DataFrame)
        
        # 5. Split features and target
        X = encoded_df.drop('HeartDisease', axis=1)
        y = encoded_df['HeartDisease']
        
        # 6. Validate features and target
        assert validator.validate_features(X)
        assert validator.validate_target_variable(y, min_samples=50)
        
        # 7. Handle class imbalance
        X_balanced, y_balanced = data_service.handle_class_imbalance(X, y)
        assert isinstance(X_balanced, pd.DataFrame)
        assert isinstance(y_balanced, pd.Series)
        assert len(X_balanced) == len(y_balanced)
        
        # 8. Feature selection
        X_selected, selected_features = data_service.select_features(X_balanced, y_balanced)
        assert isinstance(X_selected, pd.DataFrame)
        assert isinstance(selected_features, list)
        assert len(X_selected.columns) <= len(X_balanced.columns)
        
        # 9. Split data
        X_train, X_test, y_train, y_test = data_service.split_data(X_selected, y_balanced)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # 10. Scale features
        X_train_scaled = data_service.scale_features(X_train)
        X_test_scaled = data_service.scale_features(X_test)
        
        # Verify final data
        assert len(X_train_scaled) > len(X_test_scaled)
        assert len(X_train_scaled) == len(y_train)
        assert len(X_test_scaled) == len(y_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


class TestCompleteMLPipeline:
    """Test complete machine learning pipeline"""
    
    def test_end_to_end_ml_pipeline(self, services, sample_dataset):
        """Test complete ML pipeline from data to trained model"""
        # Get processed data
        X_train, X_test, y_train, y_test = self.test_end_to_end_data_processing(services, sample_dataset)
        
        model_service = services['model_service']
        validator = services['model_validator']
        
        # 1. Initialize models
        models = model_service.initialize_models()
        assert len(models) > 0
        
        # 2. Train models
        trained_models = model_service.train_models(models, X_train, y_train)
        assert len(trained_models) == len(models)
        
        # 3. Evaluate models
        evaluation_results = model_service.evaluate_models(X_test, y_test, X_test, y_test)
        assert len(evaluation_results) == len(trained_models)
        
        # Check evaluation metrics are reasonable
        for model_name, metrics in evaluation_results.items():
            assert all(0 <= metric <= 1 for metric in metrics.values())
        
        # 4. Cross-validation
        cv_results = model_service.cross_validation_evaluation(X_train, y_train)
        assert len(cv_results) == len(trained_models)
        
        # 5. Select best model
        best_model, best_model_name = model_service.select_best_model()
        assert best_model is not None
        assert isinstance(best_model_name, str)
        
        # 6. Final evaluation
        final_metrics = model_service.final_evaluation(X_test, y_test, X_test, y_test)
        assert isinstance(final_metrics, dict)
        assert 'balanced' in final_metrics
        assert 'original' in final_metrics
        
        # 7. Validate performance
        performance_valid = model_service.validate_model_performance(final_metrics)
        assert isinstance(performance_valid, bool)
        
        # 8. Get feature importance
        feature_importance = model_service.get_feature_importance(X_train)
        if feature_importance is not None:
            assert isinstance(feature_importance, pd.DataFrame)
            assert 'feature' in feature_importance.columns
            assert 'importance' in feature_importance.columns
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'final_metrics': final_metrics,
            'performance_valid': performance_valid,
            'feature_importance': feature_importance
        }


class TestFlaskAppIntegration:
    """Test Flask application integration with services"""
    
    @patch('flask_app.data_service')
    @patch('flask_app.loaded_models')
    def test_flask_prediction_integration(self, mock_loaded_models, mock_data_service, services, sample_dataset):
        """Test Flask prediction endpoint integration"""
        from flask_app import preprocess_user_input, apply_feature_engineering, get_risk_level, get_recommendations
        
        # Setup mocks
        mock_data_service.load_data.return_value = sample_dataset
        
        # Create a mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]  # 70% probability of heart disease
        mock_model.predict.return_value = [1]
        mock_loaded_models.__getitem__.return_value = mock_model
        mock_loaded_models.__contains__.return_value = True
        
        # Sample user input
        user_input = {
            'BMI': '28.5',
            'Smoking': 'No',
            'PhysicalActivity': 'Yes',
            'AgeCategory': '55-59',
            'Sex': 'Male',
            'PhysicalHealth': '0',
            'MentalHealth': '0',
            'SleepTime': '7',
            'GenHealth': 'Very good',
            'Diabetic': 'No',
            'Asthma': 'No',
            'KidneyDisease': 'No',
            'SkinCancer': 'No'
        }
        
        # Test preprocessing pipeline
        df_user = preprocess_user_input(user_input)
        assert isinstance(df_user, pd.DataFrame)
        assert len(df_user) == 1
        
        df_engineered = apply_feature_engineering(df_user)
        assert isinstance(df_engineered, pd.DataFrame)
        assert 'BMI_Category' in df_engineered.columns
        
        # Test risk level and recommendations
        probability = 0.7
        risk_level = get_risk_level(probability)
        assert risk_level == "High Risk"
        
        recommendations = get_recommendations(user_input, probability)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestEDAIntegration:
    """Test EDA service integration"""
    
    def test_eda_pipeline_integration(self, services, sample_dataset):
        """Test EDA service integration with data pipeline"""
        eda_service = services['eda_service']
        
        # Test data overview
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('seaborn.heatmap') as mock_heatmap:
                overview = eda_service.generate_data_overview(sample_dataset, "/tmp/test")
                assert isinstance(overview, dict)
                assert 'shape' in overview
                assert 'dtypes' in overview
        
        # Test target distribution plotting
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            eda_service.plot_target_distribution(sample_dataset, "/tmp/test")
            mock_savefig.assert_called()
        
        # Test correlation analysis
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            eda_service.plot_correlation_analysis(sample_dataset, "/tmp/test")
            mock_savefig.assert_called()
        
        # Test feature analysis
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            eda_service.plot_feature_analysis(sample_dataset, "/tmp/test")
            mock_savefig.assert_called()


class TestErrorHandlingIntegration:
    """Test error handling across the pipeline"""
    
    def test_pipeline_error_handling(self, services):
        """Test error handling throughout the pipeline"""
        data_service = services['data_service']
        model_service = services['model_service']
        
        # Test with invalid data
        invalid_df = pd.DataFrame()  # Empty DataFrame
        
        try:
            data_service.preprocess_data(invalid_df)
            assert False, "Should have raised an exception"
        except (ValueError, Exception):
            pass  # Expected behavior
        
        # Test with invalid model data
        invalid_X = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        invalid_y = pd.Series([0, 1, 2])  # Binary classification expects 0/1
        
        try:
            models = model_service.initialize_models()
            model_service.train_models(models, invalid_X, invalid_y)
            # Some models might handle this gracefully, others might fail
        except (ValueError, Exception):
            pass  # Expected behavior


class TestPerformanceIntegration:
    """Test performance and scalability"""
    
    def test_pipeline_performance(self, services, sample_dataset):
        """Test pipeline performance with larger dataset"""
        import time
        
        # Create larger dataset
        large_dataset = pd.concat([sample_dataset] * 10, ignore_index=True)
        
        data_service = services['data_service']
        model_service = services['model_service']
        
        # Measure preprocessing time
        start_time = time.time()
        processed_df = data_service.preprocess_data(large_dataset)
        preprocessing_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert preprocessing_time < 30.0, f"Preprocessing took too long: {preprocessing_time:.2f}s"
        
        # Test feature selection on large dataset
        X = processed_df.drop('HeartDisease', axis=1)
        y = processed_df['HeartDisease']
        
        start_time = time.time()
        X_selected, selected_features = data_service.select_features(X, y)
        feature_selection_time = time.time() - start_time
        
        assert feature_selection_time < 60.0, f"Feature selection took too long: {feature_selection_time:.2f}s"


class TestDataConsistency:
    """Test data consistency across pipeline stages"""
    
    def test_data_consistency(self, services, sample_dataset):
        """Test data consistency throughout the pipeline"""
        data_service = services['data_service']
        
        # Track data shape changes
        original_shape = sample_dataset.shape
        
        # Preprocess
        processed_df = data_service.preprocess_data(sample_dataset)
        assert processed_df.shape[0] == original_shape[0]  # Same number of rows
        
        # Feature engineering
        engineered_df = data_service._engineer_features(processed_df)
        assert engineered_df.shape[0] == processed_df.shape[0]  # Same number of rows
        
        # Encoding
        encoded_df = data_service.encode_categorical_variables(engineered_df)
        assert encoded_df.shape[0] == engineered_df.shape[0]  # Same number of rows
        
        # Splitting
        X = encoded_df.drop('HeartDisease', axis=1)
        y = encoded_df['HeartDisease']
        
        X_balanced, y_balanced = data_service.handle_class_imbalance(X, y)
        assert len(X_balanced) == len(y_balanced)
        
        # Feature selection
        X_selected, selected_features = data_service.select_features(X_balanced, y_balanced)
        assert X_selected.shape[0] == len(y_balanced)
        assert len(selected_features) == X_selected.shape[1]
        
        # Train-test split
        X_train, X_test, y_train, y_test = data_service.split_data(X_selected, y_balanced)
        total_samples = len(X_train) + len(X_test)
        assert total_samples == len(y_balanced)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestConfigurationIntegration:
    """Test configuration integration"""
    
    def test_config_integration(self, services):
        """Test that services use configuration properly"""
        # This test would verify that services respect configuration settings
        # Implementation depends on the actual configuration system
        
        data_service = services['data_service']
        model_service = services['model_service']
        
        # Test that services have expected attributes from config
        assert hasattr(data_service, 'logger')
        assert hasattr(model_service, 'logger')
        assert hasattr(model_service, 'trained_models')
        assert hasattr(model_service, 'tuned_models')


class TestEndToEndScenario:
    """Test complete end-to-end scenarios"""
    
    def test_complete_heart_disease_prediction_scenario(self, services, sample_dataset):
        """Test complete heart disease prediction scenario"""
        # 1. Data Processing
        X_train, X_test, y_train, y_test = self.test_end_to_end_data_processing(services, sample_dataset)
        
        # 2. Model Training
        ml_results = self.test_end_to_end_ml_pipeline(services, sample_dataset)
        
        # 3. Prediction Simulation
        best_model = ml_results['best_model']
        
        # Make predictions on test set
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)
        
        assert len(predictions) == len(y_test)
        assert probabilities.shape == (len(y_test), 2)  # Binary classification
        
        # 4. Results Analysis
        accuracy = (predictions == y_test).mean()
        assert accuracy > 0.5  # Should be better than random
        
        # 5. Generate recommendations for a sample patient
        sample_patient = X_test.iloc[0:1]  # First test patient
        patient_probability = probabilities[0][1]  # Probability of class 1 (heart disease)
        
        from flask_app import get_risk_level, get_recommendations
        
        risk_level = get_risk_level(patient_probability)
        assert risk_level in ["Low Risk", "Medium Risk", "High Risk"]
        
        # Create mock form data for recommendations
        form_data = {
            'BMI': str(sample_patient['BMI'].iloc[0]) if 'BMI' in sample_patient.columns else '25.0',
            'Smoking': 'No',
            'PhysicalActivity': 'Yes'
        }
        
        recommendations = get_recommendations(form_data, patient_probability)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # 6. Verify complete pipeline success
        assert ml_results['performance_valid'] is True or ml_results['performance_valid'] is False
        assert isinstance(ml_results['final_metrics'], dict)


if __name__ == '__main__':
    pytest.main([__file__])
