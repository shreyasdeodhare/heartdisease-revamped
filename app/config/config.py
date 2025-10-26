# app/config/config.py
import os
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
import os
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs', 'config.env')
load_dotenv(config_path)

@dataclass
class DataConfig:
    """Data configuration settings"""
    data_url: str = os.getenv('DATA_URL', 'https://github.com/shreyasdeodhare/Datasets101/raw/main/heart_2020_cleaned.csv')
    dataset_path: str = os.getenv('DATASET_PATH', './data/heart_disease.csv')
    output_dir: str = os.getenv('OUTPUT_DIR', './outputs')
    model_dir: str = os.getenv('MODEL_DIR', './models')

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    file: str = os.getenv('LOG_FILE', './logs/heart_disease_prediction.log')
    format: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class ModelConfig:
    """Model configuration settings"""
    test_size: float = float(os.getenv('TEST_SIZE', '0.2'))
    random_state: int = int(os.getenv('RANDOM_STATE', '42'))
    cv_folds: int = int(os.getenv('CV_FOLDS', '5'))
    n_iter_random_search: int = int(os.getenv('N_ITER_RANDOM_SEARCH', '20'))
    smote_random_state: int = int(os.getenv('SMOTE_RANDOM_STATE', '42'))
    smote_k_neighbors: int = int(os.getenv('SMOTE_K_NEIGHBORS', '5'))
    n_features_select: int = int(os.getenv('N_FEATURES_SELECT', '15'))
    correlation_threshold: float = float(os.getenv('CORRELATION_THRESHOLD', '0.8'))

@dataclass
class VisualizationConfig:
    """Visualization configuration settings"""
    figure_size_large = eval(os.getenv("FIGURE_SIZE_LARGE", "(18, 12)"))
    figure_size_medium = eval(os.getenv("FIGURE_SIZE_MEDIUM", "(12, 8)"))
    figure_size_small = eval(os.getenv("FIGURE_SIZE_SMALL", "(10, 6)"))
    dpi= int(os.getenv("DPI", "300"))
    style = os.getenv("STYLE", "seaborn-v0_8")

@dataclass
class ModelParameters:
    """Model parameter configuration"""
    rf_n_estimators: int = int(os.getenv('RF_N_ESTIMATORS', '100'))
    xgb_n_estimators: int = int(os.getenv('XGB_N_ESTIMATORS', '100'))
    lgb_n_estimators: int = int(os.getenv('LGB_N_ESTIMATORS', '100'))
    max_depth: int = int(os.getenv('MAX_DEPTH', '10'))
    learning_rate: float = float(os.getenv('LEARNING_RATE', '0.1'))

@dataclass
class PerformanceThresholds:
    """Performance threshold configuration"""
    min_accuracy_threshold: float = float(os.getenv('MIN_ACCURACY_THRESHOLD', '0.7'))
    min_f1_threshold: float = float(os.getenv('MIN_F1_THRESHOLD', '0.6'))
    min_auc_threshold: float = float(os.getenv('MIN_AUC_THRESHOLD', '0.7'))

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    enable_hyperparameter_tuning: bool = os.getenv('ENABLE_HYPERPARAMETER_TUNING', 'false').lower() == 'true'
class LoggingConfig:
    level = "INFO"
    file = "app/logs/app.log"
    max_size = 5 * 1024 * 1024  # 5MB
    backup_count = 3
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Add this line


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.model = ModelConfig()
        self.visualization = VisualizationConfig()  # Initialize VisualizationConfig
        self.model_params = ModelParameters()
        self.thresholds = PerformanceThresholds()
        self.training = TrainingConfig()
    
    def create_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.data.output_dir,
            self.data.model_dir,
            os.path.dirname(self.logging.file) if self.logging.file else None,
            os.path.dirname(self.data.dataset_path) if self.data.dataset_path else None
        ]
        
        for directory in directories:
            if directory:  # Only create if directory path is not empty or None
                os.makedirs(directory, exist_ok=True)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'data': {k: v for k, v in self.data.__dict__.items() if not k.startswith('_')},
            'model': {k: v for k, v in self.model.__dict__.items() if not k.startswith('_')},
            'logging': {k: v for k, v in self.logging.__dict__.items() if not k.startswith('_')},
            'visualization': {k: v for k, v in self.visualization.__dict__.items() if not k.startswith('_')},
            'model_params': {k: v for k, v in self.model_params.__dict__.items() if not k.startswith('_')},
            'thresholds': {k: v for k, v in self.thresholds.__dict__.items() if not k.startswith('_')}
        }

# At the end of config.py
config = Config()

# Explicitly expose visualization
visualization = config.visualization