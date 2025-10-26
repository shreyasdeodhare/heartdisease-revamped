"""
Main application entry point for Heart Disease Prediction
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any
from datetime import datetime

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import config
from app.services.data_service import DataService
from app.services.eda_service import EDAService
from app.services.model_service import ModelService
from app.utils.logging_utils import aop_logger
from app.utils.error_handling import error_handler

class HeartDiseasePredictionApp:
    """Main application class for Heart Disease Prediction"""
    
    def __init__(self):
        """Initialize the application and its components"""
        print("Initializing Heart Disease Prediction Application...")
        self.logger = aop_logger
        self.data_service = DataService()
        self.eda_service = EDAService()
        self.model_service = ModelService()
        
        # Create necessary directories
        os.makedirs(config.data.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.data.MODEL_DIR, exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        self.logger.info("Application initialized successfully")



    @error_handler.handle_data_errors
    @aop_logger.log
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete machine learning pipeline"""
        self.logger.info("Starting complete ML pipeline")
        
        try:
            # Step 1-8: Data loading, preprocessing, and preparation
            self.logger.info("=== STEP 1-8: DATA PROCESSING ===")
            df = self.data_service.load_data()
            df_processed = self.data_service.preprocess_data(df)
            df_encoded = self.data_service.encode_categorical_variables(df_processed)
            
            # Generate EDA and visualizations
            self.eda_service.generate_comprehensive_report(df_encoded, config.data.OUTPUT_DIR)
            
            # Prepare features and target
            X = df_encoded.drop('HeartDisease', axis=1)
            y = df_encoded['HeartDisease']
            
            # Data validation
            from app.utils.error_handling import data_validator
            data_validator.validate_dataframe(X)
            data_validator.validate_target_variable(y)
            
            # Feature scaling
            X_scaled = self.data_service.scale_features(X)
            
            # Handle class imbalance
            X_balanced, y_balanced = self.data_service.handle_class_imbalance(X_scaled, y)
            
            # Feature selection
            X_selected, selected_features = self.data_service.select_features(X_balanced, y_balanced)
            
            # Split data
            X_train, X_test, y_train, y_test = self.data_service.split_data(X_selected, y_balanced)

            # Create test set from original data
            _, X_orig_test_full, _, y_orig_test = self.data_service.split_data(X_scaled, y)
            X_orig_test = X_orig_test_full[selected_features]
            
            # Step 9-16: Model training and evaluation
            self.logger.info("=== STEP 9-16: MODEL TRAINING & EVALUATION ===")
            models, class_weights = self.model_service.initialize_models()
            self.model_service.train_models(models, X_train, y_train)
            
            # Evaluate models
            model_results = self.model_service.evaluate_models(X_test, y_test, X_orig_test, y_orig_test)
            
            # Hyperparameter tuning if enabled
            if config.training.enable_hyperparameter_tuning:
                self.model_service.hyperparameter_tuning(X_train, y_train)
            
            # Cross-validation
            cv_results = self.model_service.cross_validation_evaluation(X_train, y_train)
            
            # Select best model
            best_model, best_model_name = self.model_service.select_best_model()
            
            # Final evaluation
            final_metrics = self.model_service.final_evaluation(X_test, y_test, X_orig_test, y_orig_test)
            
            # Feature importance
            feature_importance = self.model_service.get_feature_importance(X_train)
            
            # Performance validation
            performance_valid = self.model_service.validate_model_performance(final_metrics)
            
            # Step 17: Save models and artifacts
            self.logger.info("=== STEP 17: SAVING MODELS ===")
            # Save all models
            self.model_service.save_models(config.data.MODEL_DIR)
            
            # Save the best model with metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"heart_disease_model_{best_model_name.lower()}_{timestamp}.pkl"
            model_path = os.path.join(config.data.MODEL_DIR, model_filename)
            
            # Prepare model data
            model_data = {
                'model': best_model,
                'model_name': best_model_name,
                'feature_names': X_train.columns.tolist(),
                'timestamp': timestamp,
                'metrics': final_metrics,
                'feature_importance': feature_importance.to_dict() if feature_importance is not None else None
            }
            
            # Save the model
            joblib.dump(model_data, model_path)
            
            # Create a symlink to the latest model
            latest_model_path = os.path.join(config.data.MODEL_DIR, "heart_disease_model_latest.pkl")
            if os.path.exists(latest_model_path):
                os.remove(latest_model_path)
            import shutil
            shutil.copy2(model_path, latest_model_path)            
            self.logger.info(f"Best model saved to {model_path}")
            self.logger.info(f"Symlink created at {latest_model_path}")
            
            # Save preprocessing artifacts
            self.data_service.save_preprocessing_artifacts(config.data.MODEL_DIR)
            
            # Generate final visualizations
            self._generate_final_visualizations(cv_results, final_metrics, feature_importance)
            
            # Generate summary report
            summary = self._generate_summary_report(
                df, best_model_name, final_metrics, cv_results, 
                performance_valid, selected_features, model_path
            )
            
            self.logger.info("Complete ML pipeline finished successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _generate_final_visualizations(self, cv_results: Dict, final_metrics: Dict, feature_importance: pd.DataFrame) -> None:
        """Generate final visualizations"""
        try:
            # Model comparison
            comparison_data = []
            for model_name, results in cv_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy']['mean'],
                    'Precision': results['precision']['mean'],
                    'Recall': results['recall']['mean'],
                    'F1-Score': results['f1']['mean'],
                    'ROC-AUC': results['roc_auc']['mean']
                })
            
            comparison_df = pd.DataFrame(comparison_data).sort_values('F1-Score', ascending=False)
            self.eda_service.plot_model_comparison(comparison_df, config.data.OUTPUT_DIR)
            
            # Feature importance
            if feature_importance is not None:
                self.eda_service.plot_feature_importance(feature_importance, config.data.OUTPUT_DIR)
            
            self.logger.info("Final visualizations generated")
            
        except Exception as e:
            self.logger.error(f"Failed to generate final visualizations: {str(e)}")

    def _generate_summary_report(self, df: pd.DataFrame, best_model_name: str, 
                              final_metrics: Dict, cv_results: Dict, 
                              performance_valid: bool, selected_features: list,
                              model_path: str) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        
        summary = {
            'dataset_info': {
                'shape': df.shape,
                'target_distribution': df['HeartDisease'].value_counts().to_dict()
            },
            'best_model': best_model_name,
            'model_path': model_path,
            'final_metrics': final_metrics,
            'cv_results': cv_results,
            'performance_valid': performance_valid,
            'selected_features': selected_features,
            'pipeline_status': 'SUCCESS',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary to file
        summary_file = os.path.join(config.data.output_dir, 'pipeline_summary.json')
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary report saved to {summary_file}")
        return summary

def main():
    """Main entry point"""
    print("Starting Heart Disease Prediction Application...")
    try:
        # Initialize application
        app = HeartDiseasePredictionApp()
        print("Application initialized successfully")
        
        # Run complete pipeline
        print("Starting complete pipeline...")
        summary = app.run_complete_pipeline()
        
        # Print final summary
        print("\n" + "="*60)
        print("HEART DISEASE PREDICTION - PIPELINE COMPLETED")
        print("="*60)
        print(f"Best Model: {summary['best_model']}")
        print(f"Model saved to: {summary['model_path']}")
        print(f"Performance Valid: {summary['performance_valid']}")
        print(f"Selected Features: {len(summary['selected_features'])}")
        
        print("\nFinal Metrics:")
        for dataset_name, metrics in summary['final_metrics'].items():
            print(f"\n{dataset_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
        
        print(f"\nPipeline Status: {summary['pipeline_status']}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Application failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())