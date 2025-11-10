import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Callable, Any
from app.config import config

class AOPLogger:
    def __init__(self, name='heart_disease_prediction'):
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        log_level = config.logging.log_level
        self.logger.setLevel(getattr(logging, log_level.upper()))

        if not self.logger.handlers:
            log_format = config.logging.log_format
            formatter = logging.Formatter(log_format)

            log_file = config.logging.log_file
            file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            self.logger.info(f"Starting {func.__name__}")
            try:
                result = func(*args, **kwargs)
                self.logger.info(f"Completed {func.__name__} successfully")
                return result
            except Exception as e:
                self.logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
        
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
        
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)


    log_function_call = log


class DataLogger:
    def __init__(self):
        self.logger = AOPLogger('data_service').logger

    def log_data_loading(self, url, shape):
        self.logger.info(f"Data loaded from {url}, shape: {shape}")

    def log_feature_engineering(self, features):
        self.logger.info(f"Features engineered: {features}")

    def log_class_balance(self, balance):
        self.logger.info(f"Class balance: {balance}")


class ModelLogger:
    def __init__(self):
        self.logger = AOPLogger('model_service').logger

    def log_model_training(self, name, params):
        self.logger.info(f"Model {name} trained with params: {params}")

    def log_model_evaluation(self, name, metrics):
        self.logger.info(f"Model {name} evaluation: {metrics}")

    def log_cross_validation(self, name, results):
        self.logger.info(f"Model {name} CV results: {results}")


aop_logger = AOPLogger()
data_logger = DataLogger()
model_logger = ModelLogger()
