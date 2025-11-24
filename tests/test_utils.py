"""
Unit tests for utility functions (logging and error handling)
"""
import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from utils.logging_utils import AOPLogger, DataLogger, ModelLogger
from utils.error_handling import ErrorHandler, DataValidator, ModelValidator, DataValidationError, ModelValidationError


class TestAOPLogger:
    """Test AOP Logger functionality"""
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        logger = AOPLogger('test_logger')
        
        assert logger is not None
        assert hasattr(logger, 'logger')
        assert logger.logger.name == 'test_logger'
    
    def test_logger_info(self):
        """Test info logging"""
        logger = AOPLogger('test_logger')
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Test message")
            mock_info.assert_called_once_with("Test message")
    
    def test_logger_error(self):
        """Test error logging"""
        logger = AOPLogger('test_logger')
        
        with patch.object(logger.logger, 'error') as mock_error:
            logger.error("Test error")
            mock_error.assert_called_once_with("Test error")
    
    def test_logger_warning(self):
        """Test warning logging"""
        logger = AOPLogger('test_logger')
        
        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.warning("Test warning")
            mock_warning.assert_called_once_with("Test warning")
    
    def test_logger_debug(self):
        """Test debug logging"""
        logger = AOPLogger('test_logger')
        
        with patch.object(logger.logger, 'debug') as mock_debug:
            logger.debug("Test debug")
            mock_debug.assert_called_once_with("Test debug")
    
    def test_logger_critical(self):
        """Test critical logging"""
        logger = AOPLogger('test_logger')
        
        with patch.object(logger.logger, 'critical') as mock_critical:
            logger.critical("Test critical")
            mock_critical.assert_called_once_with("Test critical")
    
    def test_log_decorator(self):
        """Test logging decorator"""
        logger = AOPLogger('test_logger')
        
        @logger.log
        def test_function(x, y):
            return x + y
        
        with patch.object(logger.logger, 'info') as mock_info:
            result = test_function(2, 3)
            
            assert result == 5
            # Check that logging was called for function start and end
            assert mock_info.call_count >= 1


class TestDataLogger:
    """Test Data Logger functionality"""
    
    def test_data_logger_initialization(self):
        """Test data logger initialization"""
        logger = DataLogger()
        
        assert logger is not None
        assert hasattr(logger, 'logger')
    
    def test_log_data_loading(self):
        """Test data loading log"""
        logger = DataLogger()
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_data_loading("test_url.csv", (100, 10))
            mock_info.assert_called_once_with("Data loaded from test_url.csv, shape: (100, 10)")
    
    def test_log_feature_engineering(self):
        """Test feature engineering log"""
        logger = DataLogger()
        
        with patch.object(logger.logger, 'info') as mock_info:
            features = ['BMI', 'Age', 'Sex']
            logger.log_feature_engineering(features)
            mock_info.assert_called_once_with("Features engineered: ['BMI', 'Age', 'Sex']")
    
    def test_log_class_balance(self):
        """Test class balance log"""
        logger = DataLogger()
        
        with patch.object(logger.logger, 'info') as mock_info:
            balance = {0: 800, 1: 200}
            logger.log_class_balance(balance)
            mock_info.assert_called_once_with("Class balance: {0: 800, 1: 200}")


class TestModelLogger:
    """Test Model Logger functionality"""
    
    def test_model_logger_initialization(self):
        """Test model logger initialization"""
        logger = ModelLogger()
        
        assert logger is not None
        assert hasattr(logger, 'logger')
    
    def test_log_model_training(self):
        """Test model training log"""
        logger = ModelLogger()
        
        with patch.object(logger.logger, 'info') as mock_info:
            name = "Random Forest"
            params = {"n_estimators": 100, "max_depth": 10}
            logger.log_model_training(name, params)
            mock_info.assert_called_once_with("Model Random Forest trained with params: {'n_estimators': 100, 'max_depth': 10}")
    
    def test_log_model_evaluation(self):
        """Test model evaluation log"""
        logger = ModelLogger()
        
        with patch.object(logger.logger, 'info') as mock_info:
            name = "Random Forest"
            metrics = {"accuracy": 0.8, "precision": 0.75, "recall": 0.7}
            logger.log_model_evaluation(name, metrics)
            mock_info.assert_called_once_with("Model Random Forest evaluation: {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.7}")
    
    def test_log_cross_validation(self):
        """Test cross validation log"""
        logger = ModelLogger()
        
        with patch.object(logger.logger, 'info') as mock_info:
            name = "Random Forest"
            results = {"accuracy": {"mean": 0.78, "std": 0.02}}
            logger.log_cross_validation(name, results)
            mock_info.assert_called_once_with("Model Random Forest CV results: {'accuracy': {'mean': 0.78, 'std': 0.02}}")


class TestErrorHandler:
    """Test Error Handler functionality"""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization"""
        handler = ErrorHandler()
        
        assert handler is not None
        assert hasattr(handler, 'logger')
    
    def test_handle_data_errors_empty_data(self):
        """Test handling empty data error"""
        handler = ErrorHandler()
        
        @handler.handle_data_errors
        def test_function():
            raise pd.errors.EmptyDataError("Empty data")
        
        with pytest.raises(pd.errors.EmptyDataError):
            test_function()
    
    def test_handle_data_errors_generic_error(self):
        """Test handling generic data error"""
        handler = ErrorHandler()
        
        @handler.handle_data_errors
        def test_function():
            raise ValueError("Generic error")
        
        with pytest.raises(ValueError):
            test_function()
    
    def test_handle_model_errors_value_error(self):
        """Test handling model value error"""
        handler = ErrorHandler()
        
        @handler.handle_model_errors
        def test_function():
            raise ValueError("Model error")
        
        with pytest.raises(ValueError):
            test_function()
    
    def test_handle_model_errors_success(self):
        """Test successful model operation"""
        handler = ErrorHandler()
        
        @handler.handle_model_errors
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        assert result == 5
    
    def test_handle_file_errors_not_found(self):
        """Test handling file not found error"""
        handler = ErrorHandler()
        
        @handler.handle_file_errors
        def test_function():
            raise FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            test_function()
    
    def test_handle_file_errors_permission_error(self):
        """Test handling permission error"""
        handler = ErrorHandler()
        
        @handler.handle_file_errors
        def test_function():
            raise PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            test_function()


class TestDataValidator:
    """Test Data Validator functionality"""
    
    def test_data_validator_initialization(self):
        """Test data validator initialization"""
        validator = DataValidator()
        
        assert validator is not None
        assert hasattr(validator, 'logger')
    
    def test_validate_dataframe_valid(self):
        """Test validation of valid DataFrame"""
        validator = DataValidator()
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        
        result = validator.validate_dataframe(df)
        assert result is True
    
    def test_validate_dataframe_empty(self):
        """Test validation of empty DataFrame"""
        validator = DataValidator()
        df = pd.DataFrame()
        
        with pytest.raises(DataValidationError, match="DataFrame is empty"):
            validator.validate_dataframe(df)
    
    def test_validate_dataframe_with_required_columns(self):
        """Test validation with required columns"""
        validator = DataValidator()
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        required_columns = ['col1', 'col2']
        
        result = validator.validate_dataframe(df, required_columns)
        assert result is True
    
    def test_validate_dataframe_missing_required_columns(self):
        """Test validation with missing required columns"""
        validator = DataValidator()
        df = pd.DataFrame({'col1': [1, 2, 3]})
        required_columns = ['col1', 'col2', 'col3']
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validator.validate_dataframe(df, required_columns)
    
    def test_validate_dataframe_with_null_values(self):
        """Test validation with null values"""
        validator = DataValidator()
        df = pd.DataFrame({'col1': [1, None, 3], 'col2': [4, 5, 6]})
        
        # Should handle null values gracefully
        result = validator.validate_dataframe(df)
        assert result is True
    
    def test_validate_target_variable_valid(self):
        """Test validation of valid target variable"""
        validator = DataValidator()
        y = pd.Series([0, 1, 0, 1, 0])
        
        result = validator.validate_target_variable(y)
        assert result is True
    
    def test_validate_target_variable_insufficient_samples(self):
        """Test validation with insufficient samples"""
        validator = DataValidator()
        y = pd.Series([0, 1])  # Only 2 samples
        
        with pytest.raises(DataValidationError, match="Insufficient samples"):
            validator.validate_target_variable(y, min_samples=5)
    
    def test_validate_target_variable_numpy_array(self):
        """Test validation with numpy array"""
        validator = DataValidator()
        y = np.array([0, 1, 0, 1, 0])
        
        result = validator.validate_target_variable(y)
        assert result is True
    
    def test_validate_target_variable_single_class(self):
        """Test validation with single class"""
        validator = DataValidator()
        y = pd.Series([0, 0, 0, 0, 0])  # Only one class
        
        with pytest.raises(DataValidationError, match="Target variable must have at least 2 classes"):
            validator.validate_target_variable(y)
    
    def test_validate_features_dataframe(self):
        """Test validation of feature DataFrame"""
        validator = DataValidator()
        X = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        
        result = validator.validate_features(X)
        assert result is True
    
    def test_validate_features_numpy_array(self):
        """Test validation of feature numpy array"""
        validator = DataValidator()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        result = validator.validate_features(X)
        assert result is True
    
    def test_validate_features_with_inf_values(self):
        """Test validation with infinite values"""
        validator = DataValidator()
        X = pd.DataFrame({'col1': [1, np.inf, 3], 'col2': [4, 5, 6]})
        
        with pytest.raises(DataValidationError, match="Infinite values found"):
            validator.validate_features(X)
    
    def test_validate_features_with_nan_values(self):
        """Test validation with NaN values"""
        validator = DataValidator()
        X = pd.DataFrame({'col1': [1, np.nan, 3], 'col2': [4, 5, 6]})
        
        with pytest.raises(DataValidationError, match="NaN values found"):
            validator.validate_features(X)


class TestModelValidator:
    """Test Model Validator functionality"""
    
    def test_model_validator_initialization(self):
        """Test model validator initialization"""
        validator = ModelValidator()
        
        assert validator is not None
        assert hasattr(validator, 'logger')
    
    def test_validate_model_performance_good(self):
        """Test validation of good model performance"""
        validator = ModelValidator()
        metrics = {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.7}
        thresholds = {'accuracy': 0.7, 'precision': 0.7, 'recall': 0.6}
        
        result = validator.validate_model_performance(metrics, thresholds)
        assert result is True
    
    def test_validate_model_performance_poor(self):
        """Test validation of poor model performance"""
        validator = ModelValidator()
        metrics = {'accuracy': 0.6, 'precision': 0.65, 'recall': 0.5}
        thresholds = {'accuracy': 0.7, 'precision': 0.7, 'recall': 0.6}
        
        with pytest.raises(ModelValidationError, match="Model performance below threshold"):
            validator.validate_model_performance(metrics, thresholds)
    
    def test_validate_model_performance_missing_metric(self):
        """Test validation with missing metric"""
        validator = ModelValidator()
        metrics = {'accuracy': 0.8}  # Missing precision and recall
        thresholds = {'accuracy': 0.7, 'precision': 0.7, 'recall': 0.6}
        
        # Should handle missing metrics gracefully
        result = validator.validate_model_performance(metrics, thresholds)
        assert result is True
    
    def test_validate_cross_validation_good(self):
        """Test validation of good cross-validation results"""
        validator = ModelValidator()
        cv_scores = {
            'accuracy': {'mean': 0.78, 'std': 0.02, 'scores': [0.8, 0.76, 0.78]},
            'precision': {'mean': 0.75, 'std': 0.03, 'scores': [0.77, 0.73, 0.75]}
        }
        
        result = validator.validate_cross_validation(cv_scores, min_cv_score=0.7)
        assert result is True
    
    def test_validate_cross_validation_poor(self):
        """Test validation of poor cross-validation results"""
        validator = ModelValidator()
        cv_scores = {
            'accuracy': {'mean': 0.45, 'std': 0.02, 'scores': [0.47, 0.43, 0.45]},
            'precision': {'mean': 0.55, 'std': 0.03, 'scores': [0.57, 0.53, 0.55]}
        }
        
        with pytest.raises(ModelValidationError, match="Cross-validation score below threshold"):
            validator.validate_cross_validation(cv_scores, min_cv_score=0.7)


class TestErrorHandlingIntegration:
    """Test error handling integration"""
    
    def test_combined_error_handling(self):
        """Test combined error handling decorators"""
        handler = ErrorHandler()
        
        @handler.handle_data_errors
        @handler.handle_model_errors
        def complex_function(data_type, operation):
            if data_type == "empty":
                raise pd.errors.EmptyDataError("Empty data")
            elif operation == "model_error":
                raise ValueError("Model error")
            else:
                return "success"
        
        # Test data error
        with pytest.raises(pd.errors.EmptyDataError):
            complex_function("empty", "train")
        
        # Test model error
        with pytest.raises(ValueError):
            complex_function("valid", "model_error")
        
        # Test success
        result = complex_function("valid", "train")
        assert result == "success"


class TestLoggingIntegration:
    """Test logging integration"""
    
    def test_logging_with_error_handling(self):
        """Test logging integration with error handling"""
        handler = ErrorHandler()
        
        @handler.handle_data_errors
        def function_with_logging():
            raise ValueError("Test error")
        
        # Should handle error and log it
        with pytest.raises(ValueError):
            function_with_logging()


if __name__ == '__main__':
    pytest.main([__file__])
