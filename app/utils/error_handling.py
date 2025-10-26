"""
Error handling and validation utilities for Heart Disease Prediction application
"""

import logging
from typing import Any, Callable, Dict, Optional, Union
from functools import wraps
import pandas as pd
import numpy as np
from app.utils.logging_utils import aop_logger


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class ModelValidationError(Exception):
    """Custom exception for model validation errors"""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


class ErrorHandler:
    """Centralized error handling class"""
    
    def __init__(self):
        self.logger = aop_logger.logger
    
    def handle_data_errors(self, func: Callable) -> Callable:
        """Decorator to handle data-related errors"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except pd.errors.EmptyDataError:
                self.logger.error("Empty dataset encountered")
                raise DataValidationError("Dataset is empty")
            except pd.errors.ParserError as e:
                self.logger.error(f"Data parsing error: {str(e)}")
                raise DataValidationError(f"Failed to parse data: {str(e)}")
            except KeyError as e:
                self.logger.error(f"Missing required column: {str(e)}")
                raise DataValidationError(f"Required column missing: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected data error: {str(e)}")
                raise DataValidationError(f"Data processing failed: {str(e)}")
        
        return wrapper
    
    def handle_model_errors(self, func: Callable) -> Callable:
        """Decorator to handle model-related errors"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                self.logger.error(f"Model parameter error: {str(e)}")
                raise ModelValidationError(f"Invalid model parameters: {str(e)}")
            except RuntimeError as e:
                self.logger.error(f"Model runtime error: {str(e)}")
                raise ModelValidationError(f"Model execution failed: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected model error: {str(e)}")
                raise ModelValidationError(f"Model operation failed: {str(e)}")
        
        return wrapper
    
    def handle_file_errors(self, func: Callable) -> Callable:
        """Decorator to handle file-related errors"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                self.logger.error(f"File not found: {str(e)}")
                raise FileNotFoundError(f"Required file not found: {str(e)}")
            except PermissionError as e:
                self.logger.error(f"Permission denied: {str(e)}")
                raise PermissionError(f"Access denied: {str(e)}")
            except Exception as e:
                self.logger.error(f"File operation error: {str(e)}")
                raise Exception(f"File operation failed: {str(e)}")
        
        return wrapper


class DataValidator:
    """Data validation utilities"""
    
    def __init__(self):
        self.logger = aop_logger.logger
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: list = None) -> bool:
        """Validate DataFrame structure and content"""
        try:
            # Check if DataFrame is empty
            if df.empty:
                raise DataValidationError("DataFrame is empty")
            
            # Check required columns
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise DataValidationError(f"Missing required columns: {missing_columns}")
            
            # Check for all NaN columns
            all_nan_columns = df.columns[df.isnull().all()].tolist()
            if all_nan_columns:
                self.logger.warning(f"Columns with all NaN values: {all_nan_columns}")
            
            # Check data types
            self.logger.info(f"DataFrame shape: {df.shape}")
            self.logger.info(f"DataFrame dtypes: {df.dtypes.value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"DataFrame validation failed: {str(e)}")
            raise DataValidationError(f"Data validation failed: {str(e)}")
    
    def validate_target_variable(self, y: Union[pd.Series, np.ndarray], min_samples: int = 10) -> bool:
        """Validate target variable"""
        try:
            if len(y) < min_samples:
                raise DataValidationError(f"Insufficient samples: {len(y)} < {min_samples}")
            
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                raise DataValidationError("Target variable must have at least 2 classes")
            
            # Check class balance
            class_counts = pd.Series(y).value_counts()
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            
            imbalance_ratio = max_class_count / min_class_count
            if imbalance_ratio > 10:
                self.logger.warning(f"High class imbalance ratio: {imbalance_ratio:.2f}")
            
            self.logger.info(f"Target variable classes: {unique_classes}")
            self.logger.info(f"Class distribution: {class_counts.to_dict()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Target variable validation failed: {str(e)}")
            raise DataValidationError(f"Target validation failed: {str(e)}")
    
    def validate_features(self, X: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate feature matrix"""
        try:
            if isinstance(X, pd.DataFrame):
                # Check for infinite values
                inf_columns = X.columns[X.isin([np.inf, -np.inf]).any()].tolist()
                if inf_columns:
                    self.logger.warning(f"Columns with infinite values: {inf_columns}")
                
                # Check for constant columns
                constant_columns = X.columns[X.nunique() <= 1].tolist()
                if constant_columns:
                    self.logger.warning(f"Constant columns: {constant_columns}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Feature validation failed: {str(e)}")
            raise DataValidationError(f"Feature validation failed: {str(e)}")


class ModelValidator:
    """Model validation utilities"""
    
    def __init__(self):
        self.logger = aop_logger.logger
    
    def validate_model_performance(self, metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
        """Validate model performance against thresholds"""
        try:
            for metric, threshold in thresholds.items():
                if metric in metrics:
                    if metrics[metric] < threshold:
                        self.logger.warning(f"{metric} below threshold: {metrics[metric]:.3f} < {threshold}")
                        return False
                    else:
                        self.logger.info(f"{metric} meets threshold: {metrics[metric]:.3f} >= {threshold}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model performance validation failed: {str(e)}")
            raise ModelValidationError(f"Performance validation failed: {str(e)}")
    
    def validate_cross_validation(self, cv_scores: Dict[str, Any], min_cv_score: float = 0.5) -> bool:
        """Validate cross-validation results"""
        try:
            for metric, scores in cv_scores.items():
                mean_score = scores['mean']
                if mean_score < min_cv_score:
                    self.logger.warning(f"CV {metric} below minimum: {mean_score:.3f} < {min_cv_score}")
                    return False
                else:
                    self.logger.info(f"CV {metric} acceptable: {mean_score:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cross-validation validation failed: {str(e)}")
            raise ModelValidationError(f"CV validation failed: {str(e)}")


# Global error handler and validators
error_handler = ErrorHandler()
data_validator = DataValidator()
model_validator = ModelValidator()
