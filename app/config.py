"""
Configuration management for Heart Disease Prediction application
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from configs folder
config_path = Path(__file__).parent.parent / 'configs' / 'config.env'
load_dotenv(config_path)

def get_env(key, default):
    """Get environment variable, trying both upper and lower case"""
    return os.getenv(key.upper()) or os.getenv(key.lower()) or default

class DataConfig:
    """Data configuration settings"""
    def __init__(self):
        # Store values in private variables
        self._data_url = get_env("DATA_URL", "https://github.com/shreyasdeodhare/Datasets101/raw/main/heart_2020_cleaned.csv")
        self._dataset_path = get_env("DATASET_PATH", "./data/heart_disease.csv")
        self._output_dir = get_env("OUTPUT_DIR", "./outputs")
        self._model_dir = get_env("MODEL_DIR", "./models")

    # Lowercase properties
    @property
    def data_url(self): return self._data_url
    @property
    def dataset_path(self): return self._dataset_path
    @property
    def output_dir(self): return self._output_dir
    @property
    def model_dir(self): return self._model_dir

    # Uppercase properties
    @property
    def DATA_URL(self): return self._data_url
    @property
    def DATASET_PATH(self): return self._dataset_path
    @property
    def OUTPUT_DIR(self): return self._output_dir
    @property
    def MODEL_DIR(self): return self._model_dir

class ModelConfig:
    """Model configuration settings"""
    def __init__(self):
        self._test_size = float(get_env("TEST_SIZE", "0.2"))
        self._random_state = int(get_env("RANDOM_STATE", "42"))
        self._cv_folds = int(get_env("CV_FOLDS", "3"))
        self._n_iter_random_search = int(get_env("N_ITER_RANDOM_SEARCH", "5"))
        self._smote_random_state = int(get_env("SMOTE_RANDOM_STATE", "42"))
        self._smote_k_neighbors = int(get_env("SMOTE_K_NEIGHBORS", "5"))
        self._n_features_select = int(get_env("N_FEATURES_SELECT", "15"))
        self._correlation_threshold = float(get_env("CORRELATION_THRESHOLD", "0.8"))

    # Lowercase properties
    @property
    def test_size(self): return self._test_size
    @property
    def random_state(self): return self._random_state
    @property
    def cv_folds(self): return self._cv_folds
    @property
    def n_iter_random_search(self): return self._n_iter_random_search
    @property
    def smote_random_state(self): return self._smote_random_state
    @property
    def smote_k_neighbors(self): return self._smote_k_neighbors
    @property
    def n_features_select(self): return self._n_features_select
    @property
    def correlation_threshold(self): return self._correlation_threshold

    # Uppercase properties
    @property
    def TEST_SIZE(self): return self._test_size
    @property
    def RANDOM_STATE(self): return self._random_state
    @property
    def CV_FOLDS(self): return self._cv_folds
    @property
    def N_ITER_RANDOM_SEARCH(self): return self._n_iter_random_search
    @property
    def SMOTE_RANDOM_STATE(self): return self._smote_random_state
    @property
    def SMOTE_K_NEIGHBORS(self): return self._smote_k_neighbors
    @property
    def N_FEATURES_SELECT(self): return self._n_features_select
    @property
    def CORRELATION_THRESHOLD(self): return self._correlation_threshold

class LoggingConfig:
    """Logging configuration settings"""
    def __init__(self):
        self._log_level = get_env("LOG_LEVEL", "INFO")
        self._log_file = get_env("LOG_FILE", "./logs/heart_disease_prediction.log")
        self._log_format = get_env("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Lowercase properties
    @property
    def log_level(self): return self._log_level
    @property
    def log_file(self): return self._log_file
    @property
    def log_format(self): return self._log_format

    # Uppercase properties
    @property
    def LOG_LEVEL(self): return self._log_level
    @property
    def LOG_FILE(self): return self._log_file
    @property
    def LOG_FORMAT(self): return self._log_format

class VisualizationConfig:
    """Visualization configuration settings"""
    def __init__(self):
        self._figure_size_large = eval(get_env("FIGURE_SIZE_LARGE", "(18, 12)"))
        self._figure_size_medium = eval(get_env("FIGURE_SIZE_MEDIUM", "(12, 8)"))
        self._figure_size_small = eval(get_env("FIGURE_SIZE_SMALL", "(10, 6)"))
        self._dpi = int(get_env("DPI", "300"))
        self._style = get_env("STYLE", "seaborn-v0_8")

    # Lowercase properties
    @property
    def figure_size_large(self): return self._figure_size_large
    @property
    def figure_size_medium(self): return self._figure_size_medium
    @property
    def figure_size_small(self): return self._figure_size_small
    @property
    def dpi(self): return self._dpi
    @property
    def style(self): return self._style

    # Uppercase properties
    @property
    def FIGURE_SIZE_LARGE(self): return self._figure_size_large
    @property
    def FIGURE_SIZE_MEDIUM(self): return self._figure_size_medium
    @property
    def FIGURE_SIZE_SMALL(self): return self._figure_size_small
    @property
    def DPI(self): return self._dpi
    @property
    def STYLE(self): return self._style

class ModelParamsConfig:
    """Model parameters configuration"""
    def __init__(self):
        self._rf_n_estimators = int(get_env("RF_N_ESTIMATORS", "100"))
        self._xgb_n_estimators = int(get_env("XGB_N_ESTIMATORS", "100"))
        self._lgb_n_estimators = int(get_env("LGB_N_ESTIMATORS", "100"))
        self._max_depth = int(get_env("MAX_DEPTH", "10"))
        self._learning_rate = float(get_env("LEARNING_RATE", "0.1"))

    # Lowercase properties
    @property
    def rf_n_estimators(self): return self._rf_n_estimators
    @property
    def xgb_n_estimators(self): return self._xgb_n_estimators
    @property
    def lgb_n_estimators(self): return self._lgb_n_estimators
    @property
    def max_depth(self): return self._max_depth
    @property
    def learning_rate(self): return self._learning_rate

    # Uppercase properties
    @property
    def RF_N_ESTIMATORS(self): return self._rf_n_estimators
    @property
    def XGB_N_ESTIMATORS(self): return self._xgb_n_estimators
    @property
    def LGB_N_ESTIMATORS(self): return self._lgb_n_estimators
    @property
    def MAX_DEPTH(self): return self._max_depth
    @property
    def LEARNING_RATE(self): return self._learning_rate

class PerformanceConfig:
    """Performance thresholds configuration"""
    def __init__(self):
        self._min_accuracy_threshold = float(get_env("MIN_ACCURACY_THRESHOLD", "0.7"))
        self._min_f1_threshold = float(get_env("MIN_F1_THRESHOLD", "0.6"))
        self._min_auc_threshold = float(get_env("MIN_AUC_THRESHOLD", "0.7"))

    # Lowercase properties
    @property
    def min_accuracy_threshold(self): return self._min_accuracy_threshold
    @property
    def min_f1_threshold(self): return self._min_f1_threshold
    @property
    def min_auc_threshold(self): return self._min_auc_threshold

    # Uppercase properties
    @property
    def MIN_ACCURACY_THRESHOLD(self): return self._min_accuracy_threshold
    @property
    def MIN_F1_THRESHOLD(self): return self._min_f1_threshold
    @property
    def MIN_AUC_THRESHOLD(self): return self._min_auc_threshold

class TrainingConfig:
    """Training options configuration"""
    def __init__(self):
        self._enable_hyperparameter_tuning = get_env("ENABLE_HYPERPARAMETER_TUNING", "false").lower() == "true"

    # Lowercase property
    @property
    def enable_hyperparameter_tuning(self): return self._enable_hyperparameter_tuning

    # Uppercase property
    @property
    def ENABLE_HYPERPARAMETER_TUNING(self): return self._enable_hyperparameter_tuning

class Config:
    """Main configuration class"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.logging = LoggingConfig()
        self.visualization = VisualizationConfig()
        self.model_params = ModelParamsConfig()
        self.performance = PerformanceConfig()
        self.training = TrainingConfig()
        self.treshold=self.performance

    # Add uppercase access for backward compatibility
    @property
    def DATA(self): return self.data
    @property
    def MODEL(self): return self.model
    @property
    def LOGGING(self): return self.logging
    @property
    def VISUALIZATION(self): return self.visualization
    @property
    def MODEL_PARAMS(self): return self.model_params
    @property
    def PERFORMANCE(self): return self.performance
    @property
    def TRAINING(self): return self.training

# Create global config instance
config = Config()