"""
Model training and evaluation service for Heart Disease Prediction application
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_auc_score, roc_curve)
import xgboost as xgb
import lightgbm as lgb

from app.config import config
from app.utils.logging_utils import model_logger, aop_logger
from app.utils.error_handling import error_handler, model_validator


class ModelService:
    """Service for model training and evaluation"""
    
    def __init__(self):
        self.logger = aop_logger.logger
        self.trained_models = {}
        self.tuned_models = {}
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = None
        
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models for training"""
        self.logger.info("Initializing models")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=config.model_params.rf_n_estimators, random_state=config.model.random_state, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=config.model.random_state, eval_metric='logloss', n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=config.model.random_state, max_iter=1000, n_jobs=-1)
        }
        
        # Class weights for imbalanced data (will be calculated dynamically)
        class_weights = {
            'Random Forest': RandomForestClassifier(n_estimators=config.model_params.rf_n_estimators, class_weight='balanced', random_state=config.model.random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=config.model.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=config.model.random_state, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=config.model.random_state, verbose=-1, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(random_state=config.model.random_state, max_iter=1000, class_weight='balanced'),
            'SVM': SVC(random_state=config.model.random_state, probability=True, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(random_state=config.model.random_state, class_weight='balanced')
        }
        
        self.logger.info(f"Initialized {len(models)} models")
        return models, class_weights
    
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def train_models(self, models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all models"""
        self.logger.info(f"Starting model training with {len(models)} models...")
        self.logger.info(f"Training data shape: {X_train.shape}")
        
        for i, (name, model) in enumerate(models.items()):
            self.logger.info(f"Training model {i+1}/{len(models)}: {name}")
            self.logger.info(f"Model type: {type(model).__name__}")
            
            import time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.trained_models[name] = model
            self.logger.info(f"{name} training completed in {training_time:.2f} seconds")
            model_logger.log_model_training(name, {'training_time': training_time})
        
        self.logger.info(f"All {len(self.trained_models)} models trained successfully")
        return self.trained_models
    
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series, X_orig_test: pd.DataFrame, y_orig_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        self.logger.info("Evaluating models")
        
        model_results = {}
        
        for name, model in self.trained_models.items():
            self.logger.info(f"Evaluating {name}...")
            
            # Predictions
            y_pred_test = model.predict(X_test)
            y_pred_orig = model.predict(X_orig_test)
            
            # Calculate metrics
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test),
                'recall': recall_score(y_test, y_pred_test),
                'f1': f1_score(y_test, y_pred_test)
            }
            
            orig_metrics = {
                'accuracy': accuracy_score(y_orig_test, y_pred_orig),
                'precision': precision_score(y_orig_test, y_pred_orig),
                'recall': recall_score(y_orig_test, y_pred_orig),
                'f1': f1_score(y_orig_test, y_pred_orig)
            }
            
            model_results[name] = {
                'test': test_metrics,
                'original': orig_metrics
            }
            
            model_logger.log_model_evaluation(name, test_metrics)
        
        return model_results
    
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter tuning for top models"""
        self.logger.info("Starting hyperparameter tuning")
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'num_leaves': [31, 50, 100, 200],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        # Focus on top 2 models for tuning
        top_models = ['Random Forest', 'XGBoost']
        tuning_results = {}
        
        for model_name in top_models:
            if model_name in param_grids and model_name in self.trained_models:
                self.logger.info(f"Tuning {model_name}...")
                
                random_search = RandomizedSearchCV(
                    self.trained_models[model_name],
                    param_grids[model_name],
                    n_iter=config.model.n_iter_random_search,
                    cv=2,
                    scoring='f1',
                    random_state=config.model.random_state,
                    n_jobs=-1
                )
                
                random_search.fit(X_train, y_train)
                self.tuned_models[model_name] = random_search.best_estimator_
                tuning_results[model_name] = {
                    'best_params': random_search.best_params_,
                    'best_score': random_search.best_score_
                }
                
                self.logger.info(f"{model_name} best parameters: {random_search.best_params_}")
                self.logger.info(f"{model_name} best CV score: {random_search.best_score_:.3f}")
        
        return tuning_results
    
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def cross_validation_evaluation(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Perform cross-validation evaluation"""
        self.logger.info("Performing cross-validation evaluation")
        
        cv_strategy = StratifiedKFold(n_splits=config.model.cv_folds, shuffle=True, random_state=config.model.random_state)
        all_models = {**self.trained_models, **self.tuned_models}
        
        for model_name, model in all_models.items():
            self.logger.info(f"Cross-validation for {model_name}...")
            
            # Cross-validation scores
            cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
            cv_precision = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='precision')
            cv_recall = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='recall')
            cv_f1 = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1')
            cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc')
            
            self.cv_results[model_name] = {
                'accuracy': {'mean': cv_accuracy.mean(), 'std': cv_accuracy.std()},
                'precision': {'mean': cv_precision.mean(), 'std': cv_precision.std()},
                'recall': {'mean': cv_recall.mean(), 'std': cv_recall.std()},
                'f1': {'mean': cv_f1.mean(), 'std': cv_f1.std()},
                'roc_auc': {'mean': cv_roc_auc.mean(), 'std': cv_roc_auc.std()}
            }
            
            model_logger.log_cross_validation(model_name, self.cv_results[model_name])
        
        return self.cv_results
    
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def select_best_model(self) -> Tuple[Any, str]:
        """Select the best model based on cross-validation results"""
        self.logger.info("Selecting best model")
        
        if not self.cv_results:
            raise ValueError("No cross-validation results available")
        
        # Find best model based on F1 score
        best_model_name = max(self.cv_results.keys(), key=lambda x: self.cv_results[x]['f1']['mean'])
        
        # Get the best model (tuned if available, otherwise trained)
        if best_model_name in self.tuned_models:
            self.best_model = self.tuned_models[best_model_name]
        else:
            self.best_model = self.trained_models[best_model_name]
        
        self.best_model_name = best_model_name
        
        self.logger.info(f"Best model selected: {best_model_name}")
        self.logger.info(f"Best CV F1: {self.cv_results[best_model_name]['f1']['mean']:.3f}")
        
        return self.best_model, self.best_model_name
    
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def final_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series, X_orig_test: pd.DataFrame, y_orig_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Perform final evaluation on best model"""
        self.logger.info("Performing final evaluation")
        
        if not self.best_model:
            raise ValueError("No best model selected")
        
        # Final predictions
        y_pred_final = self.best_model.predict(X_test)
        y_pred_orig_final = self.best_model.predict(X_orig_test)
        
        # Calculate comprehensive metrics
        final_metrics = {
            'Balanced Test Set': {
                'accuracy': accuracy_score(y_test, y_pred_final),
                'precision': precision_score(y_test, y_pred_final),
                'recall': recall_score(y_test, y_pred_final),
                'f1': f1_score(y_test, y_pred_final)
            },
            'Original Test Set': {
                'accuracy': accuracy_score(y_orig_test, y_pred_orig_final),
                'precision': precision_score(y_orig_test, y_pred_orig_final),
                'recall': recall_score(y_orig_test, y_pred_orig_final),
                'f1': f1_score(y_orig_test, y_pred_orig_final)
            }
        }
        
        # ROC curve analysis
        if hasattr(self.best_model, 'predict_proba'):
            y_proba_balanced = self.best_model.predict_proba(X_test)[:, 1]
            y_proba_original = self.best_model.predict_proba(X_orig_test)[:, 1]
            
            fpr_balanced, tpr_balanced, _ = roc_curve(y_test, y_proba_balanced)
            fpr_original, tpr_original, _ = roc_curve(y_orig_test, y_proba_original)
            
            auc_balanced = roc_auc_score(y_test, y_proba_balanced)
            auc_original = roc_auc_score(y_orig_test, y_proba_original)
            
            final_metrics['Balanced Test Set']['auc'] = auc_balanced
            final_metrics['Original Test Set']['auc'] = auc_original
        
        self.logger.info("Final evaluation completed")
        return final_metrics
    
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def get_feature_importance(self, X_train: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get feature importance from best model"""
        if not self.best_model or not hasattr(self.best_model, 'feature_importances_'):
            return None
        
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def save_models(self, output_dir: str) -> None:
        """Save all models and artifacts"""
        self.logger.info("Saving models")
        
        import os
        from datetime import datetime
        import joblib
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model with feature names
        if self.best_model:
            # Get feature names if available
            feature_names = []
            if hasattr(self, 'X_train_processed') and hasattr(self.X_train_processed, 'columns'):
                feature_names = self.X_train_processed.columns.tolist()
            
            # Import the model class to ensure it's available during unpickling
            from sklearn.ensemble import RandomForestClassifier
            
            # Create a new instance of the model with the same parameters
            # This ensures we're using the standard scikit-learn class
            if hasattr(self.best_model, 'get_params'):
                model_params = self.best_model.get_params()
                model = RandomForestClassifier(**model_params)
                
                # If the model is already fitted, we need to retrain it
                if hasattr(self.best_model, 'estimators_') and hasattr(self, 'X_train_processed') and hasattr(self, 'y_train'):
                    model.fit(self.X_train_processed, self.y_train)
                
                # Create a dictionary to store model and feature names
                model_data = {
                    'model': model,
                    'feature_names': feature_names
                }
            else:
                model_data = {
                    'model': self.best_model,
                    'feature_names': feature_names
                }
            
            # Generate timestamp for the model file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"heart_disease_model_{model.__class__.__name__.lower()}_{timestamp}.pkl"
            model_path = os.path.join(output_dir, model_name)
            
            # Save using joblib which is better for scikit-learn models
            joblib.dump(model_data, model_path, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"Best model saved to {model_path}")
            
            # Create a symlink for latest model
            latest_path = os.path.join(output_dir, 'heart_disease_model_latest.pkl')
            try:
                if os.path.exists(latest_path):
                    os.remove(latest_path)
                os.symlink(model_name, latest_path)
                self.logger.info(f"Symlink created at {latest_path}")
            except OSError as e:
                self.logger.warning(f"Could not create symlink: {str(e)}")
        
        # Save all trained models
        with open(os.path.join(output_dir, 'all_models.pkl'), 'wb') as f:
            pickle.dump(self.trained_models, f)
        
        # Save tuned models
        if self.tuned_models:
            with open(os.path.join(output_dir, 'tuned_models.pkl'), 'wb') as f:
                pickle.dump(self.tuned_models, f)
        
        # Save cross-validation results
        with open(os.path.join(output_dir, 'cv_results.pkl'), 'wb') as f:
            pickle.dump(self.cv_results, f)
        
        self.logger.info(f"Models saved to {output_dir}")
    
    @error_handler.handle_model_errors
    @aop_logger.log_function_call
    def validate_model_performance(self, final_metrics: Dict[str, Dict[str, float]]) -> bool:
        """Validate model performance against thresholds"""
        self.logger.info("Validating model performance")
        
        thresholds = {
            'accuracy': config.thresholds.min_accuracy_threshold,
            'f1': config.thresholds.min_f1_threshold,
            'auc': config.thresholds.min_auc_threshold
        }
        
        for dataset_name, metrics in final_metrics.items():
            for metric, threshold in thresholds.items():
                if metric in metrics:
                    if metrics[metric] < threshold:
                        self.logger.warning(f"{dataset_name} {metric} below threshold: {metrics[metric]:.3f} < {threshold}")
                        return False
                    else:
                        self.logger.info(f"{dataset_name} {metric} meets threshold: {metrics[metric]:.3f} >= {threshold}")
        
        return True
