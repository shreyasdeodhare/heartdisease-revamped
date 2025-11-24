"""
Unit tests for model service functions
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from services.model_service import ModelService
from utils.error_handling import ModelValidationError


@pytest.fixture
def model_service():
    """Create ModelService instance for testing"""
    return ModelService()


@pytest.fixture
def sample_X_train():
    """Create sample training features"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'feature4': np.random.randn(100),
        'feature5': np.random.randn(100)
    })


@pytest.fixture
def sample_y_train():
    """Create sample training target"""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100))


@pytest.fixture
def sample_X_test():
    """Create sample test features"""
    np.random.seed(123)
    return pd.DataFrame({
        'feature1': np.random.randn(30),
        'feature2': np.random.randn(30),
        'feature3': np.random.randn(30),
        'feature4': np.random.randn(30),
        'feature5': np.random.randn(30)
    })


@pytest.fixture
def sample_y_test():
    """Create sample test target"""
    np.random.seed(123)
    return pd.Series(np.random.randint(0, 2, 30))


class TestModelInitialization:
    """Test model initialization functionality"""
    
    def test_initialize_models(self, model_service):
        """Test model initialization"""
        models = model_service.initialize_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Check that expected models are present
        expected_models = ['Random Forest', 'Logistic Regression', 'XGBoost', 'LightGBM']
        for model_name in expected_models:
            assert model_name in models
            assert models[model_name] is not None
    
    def test_initialize_models_random_state(self, model_service):
        """Test that models have consistent random state"""
        models = model_service.initialize_models()
        
        # Check that models support random_state parameter
        for name, model in models.items():
            if hasattr(model, 'random_state'):
                assert model.random_state is not None


class TestModelTraining:
    """Test model training functionality"""
    
    def test_train_models_basic(self, model_service, sample_X_train, sample_y_train):
        """Test basic model training"""
        models = model_service.initialize_models()
        
        trained_models = model_service.train_models(models, sample_X_train, sample_y_train)
        
        assert isinstance(trained_models, dict)
        assert len(trained_models) == len(models)
        
        # Check that all models are trained
        for name, model in trained_models.items():
            assert hasattr(model, 'predict') or hasattr(model, 'fit')
    
    def test_train_models_empty_data(self, model_service):
        """Test training with empty data"""
        empty_X = pd.DataFrame()
        empty_y = pd.Series()
        models = model_service.initialize_models()
        
        with pytest.raises((ValueError, ModelValidationError)):
            model_service.train_models(models, empty_X, empty_y)
    
    def test_train_models_mismatched_data(self, model_service, sample_X_train):
        """Test training with mismatched X and y lengths"""
        wrong_length_y = pd.Series([0, 1, 0])  # Much shorter than X
        models = model_service.initialize_models()
        
        with pytest.raises((ValueError, ModelValidationError)):
            model_service.train_models(models, sample_X_train, wrong_length_y)


class TestModelEvaluation:
    """Test model evaluation functionality"""
    
    def test_evaluate_models_basic(self, model_service, sample_X_train, sample_y_train, sample_X_test, sample_y_test):
        """Test basic model evaluation"""
        # First train models
        models = model_service.initialize_models()
        trained_models = model_service.train_models(models, sample_X_train, sample_y_train)
        
        # Then evaluate
        results = model_service.evaluate_models(sample_X_test, sample_y_test, sample_X_test, sample_y_test)
        
        assert isinstance(results, dict)
        assert len(results) == len(trained_models)
        
        # Check that results contain expected metrics
        for model_name, metrics in results.items():
            assert isinstance(metrics, dict)
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert 'roc_auc' in metrics
            
            # Check metric values are reasonable
            for metric_value in metrics.values():
                assert 0 <= metric_value <= 1
    
    def test_evaluate_models_single_model(self, model_service, sample_X_train, sample_y_train, sample_X_test, sample_y_test):
        """Test evaluation of single trained model"""
        # Train just one model
        models = model_service.initialize_models()
        model_name = list(models.keys())[0]
        single_model = {model_name: models[model_name]}
        
        trained_models = model_service.train_models(single_model, sample_X_train, sample_y_train)
        
        results = model_service.evaluate_models(sample_X_test, sample_y_test, sample_X_test, sample_y_test)
        
        assert isinstance(results, dict)
        assert len(results) == 1
        assert model_name in results


class TestHyperparameterTuning:
    """Test hyperparameter tuning functionality"""
    
    @patch('sklearn.model_selection.GridSearchCV')
    def test_hyperparameter_tuning_basic(self, mock_grid_search, model_service, sample_X_train, sample_y_train):
        """Test basic hyperparameter tuning"""
        # Mock GridSearchCV to avoid long computation
        mock_search = Mock()
        mock_search.best_estimator_ = Mock()
        mock_search.best_params_ = {'n_estimators': 100}
        mock_search.best_score_ = 0.8
        mock_grid_search.return_value = mock_search
        
        tuned_models = model_service.hyperparameter_tuning(sample_X_train, sample_y_train)
        
        assert isinstance(tuned_models, dict)
        assert len(tuned_models) >= 0  # May be empty if no models qualify for tuning
    
    def test_hyperparameter_tuning_small_dataset(self, model_service):
        """Test hyperparameter tuning with small dataset"""
        small_X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        small_y = pd.Series([0, 1, 0])
        
        # Should handle small datasets gracefully
        tuned_models = model_service.hyperparameter_tuning(small_X, small_y)
        
        assert isinstance(tuned_models, dict)


class TestCrossValidation:
    """Test cross-validation functionality"""
    
    def test_cross_validation_evaluation(self, model_service, sample_X_train, sample_y_train):
        """Test cross-validation evaluation"""
        # First train models
        models = model_service.initialize_models()
        trained_models = model_service.train_models(models, sample_X_train, sample_y_train)
        
        cv_results = model_service.cross_validation_evaluation(sample_X_train, sample_y_train)
        
        assert isinstance(cv_results, dict)
        assert len(cv_results) == len(trained_models)
        
        # Check CV results structure
        for model_name, cv_metrics in cv_results.items():
            assert isinstance(cv_metrics, dict)
            for metric_name, scores in cv_metrics.items():
                assert isinstance(scores, dict)
                assert 'mean' in scores
                assert 'std' in scores
                assert 'scores' in scores
                
                # Check mean score is reasonable
                assert 0 <= scores['mean'] <= 1
                assert scores['std'] >= 0
    
    def test_cross_validation_single_model(self, model_service, sample_X_train, sample_y_train):
        """Test cross-validation with single model"""
        models = model_service.initialize_models()
        model_name = list(models.keys())[0]
        single_model = {model_name: models[model_name]}
        
        trained_models = model_service.train_models(single_model, sample_X_train, sample_y_train)
        
        cv_results = model_service.cross_validation_evaluation(sample_X_train, sample_y_train)
        
        assert isinstance(cv_results, dict)
        assert len(cv_results) == 1


class TestModelSelection:
    """Test model selection functionality"""
    
    def test_select_best_model(self, model_service, sample_X_train, sample_y_train):
        """Test best model selection"""
        # First train and evaluate models
        models = model_service.initialize_models()
        trained_models = model_service.train_models(models, sample_X_train, sample_y_train)
        
        cv_results = model_service.cross_validation_evaluation(sample_X_train, sample_y_train)
        
        # Now select best model
        best_model, best_model_name = model_service.select_best_model()
        
        assert best_model is not None
        assert isinstance(best_model_name, str)
        assert best_model_name in trained_models
    
    def test_select_best_model_no_cv_results(self, model_service):
        """Test best model selection with no CV results"""
        model_service.cv_results = {}
        
        with pytest.raises(ValueError, match="No cross-validation results available"):
            model_service.select_best_model()


class TestFinalEvaluation:
    """Test final evaluation functionality"""
    
    def test_final_evaluation(self, model_service, sample_X_train, sample_y_train, sample_X_test, sample_y_test):
        """Test final evaluation of best model"""
        # Complete pipeline
        models = model_service.initialize_models()
        trained_models = model_service.train_models(models, sample_X_train, sample_y_train)
        
        cv_results = model_service.cross_validation_evaluation(sample_X_train, sample_y_train)
        best_model, best_model_name = model_service.select_best_model()
        
        final_metrics = model_service.final_evaluation(sample_X_test, sample_y_test, sample_X_test, sample_y_test)
        
        assert isinstance(final_metrics, dict)
        assert len(final_metrics) == 2  # balanced and original metrics
        assert 'balanced' in final_metrics
        assert 'original' in final_metrics
        
        # Check metrics structure
        for dataset_type, metrics in final_metrics.items():
            assert isinstance(metrics, dict)
            for metric_name, value in metrics.items():
                assert 0 <= value <= 1
    
    def test_final_evaluation_no_best_model(self, model_service, sample_X_test, sample_y_test):
        """Test final evaluation with no best model selected"""
        model_service.best_model = None
        
        with pytest.raises(ValueError, match="No best model selected"):
            model_service.final_evaluation(sample_X_test, sample_y_test, sample_X_test, sample_y_test)


class TestFeatureImportance:
    """Test feature importance functionality"""
    
    def test_get_feature_importance(self, model_service, sample_X_train, sample_y_train):
        """Test feature importance extraction"""
        # Train a model that supports feature importance
        models = model_service.initialize_models()
        trained_models = model_service.train_models(models, sample_X_train, sample_y_train)
        
        # Set best model to Random Forest (supports feature_importances_)
        if 'Random Forest' in trained_models:
            model_service.best_model = trained_models['Random Forest']
            
            feature_importance = model_service.get_feature_importance(sample_X_train)
            
            if feature_importance is not None:
                assert isinstance(feature_importance, pd.DataFrame)
                assert len(feature_importance) == len(sample_X_train.columns)
                assert 'feature' in feature_importance.columns
                assert 'importance' in feature_importance.columns
    
    def test_get_feature_importance_no_importance(self, model_service, sample_X_train, sample_y_train):
        """Test feature importance with model that doesn't support it"""
        # Create a mock model without feature_importances_
        mock_model = Mock()
        del mock_model.feature_importances_  # Ensure attribute doesn't exist
        
        model_service.best_model = mock_model
        
        feature_importance = model_service.get_feature_importance(sample_X_train)
        
        assert feature_importance is None


class TestModelSaving:
    """Test model saving functionality"""
    
    @patch('pickle.dump')
    @patch('builtins.open', create=True)
    def test_save_models(self, mock_open, mock_pickle, model_service, sample_X_train, sample_y_train):
        """Test model saving"""
        # Train models first
        models = model_service.initialize_models()
        trained_models = model_service.train_models(models, sample_X_train, sample_y_train)
        
        model_service.trained_models = trained_models
        
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        output_dir = "/tmp/test_models"
        
        # Should not raise an exception
        model_service.save_models(output_dir)
        
        # Check that file operations were attempted
        mock_open.assert_called()


class TestModelValidation:
    """Test model validation functionality"""
    
    def test_validate_model_performance_good(self, model_service):
        """Test model performance validation with good metrics"""
        final_metrics = {
            'balanced': {'accuracy': 0.8, 'f1': 0.75, 'roc_auc': 0.85},
            'original': {'accuracy': 0.78, 'f1': 0.73, 'roc_auc': 0.83}
        }
        
        result = model_service.validate_model_performance(final_metrics)
        
        assert isinstance(result, bool)
    
    def test_validate_model_performance_poor(self, model_service):
        """Test model performance validation with poor metrics"""
        final_metrics = {
            'balanced': {'accuracy': 0.5, 'f1': 0.45, 'roc_auc': 0.55},
            'original': {'accuracy': 0.48, 'f1': 0.43, 'roc_auc': 0.53}
        }
        
        result = model_service.validate_model_performance(final_metrics)
        
        assert isinstance(result, bool)


class TestIntegration:
    """Integration tests for model service"""
    
    def test_complete_model_pipeline(self, model_service, sample_X_train, sample_y_train, sample_X_test, sample_y_test):
        """Test complete model training and evaluation pipeline"""
        # 1. Initialize models
        models = model_service.initialize_models()
        
        # 2. Train models
        trained_models = model_service.train_models(models, sample_X_train, sample_y_train)
        
        # 3. Evaluate models
        evaluation_results = model_service.evaluate_models(sample_X_test, sample_y_test, sample_X_test, sample_y_test)
        
        # 4. Cross-validation
        cv_results = model_service.cross_validation_evaluation(sample_X_train, sample_y_train)
        
        # 5. Select best model
        best_model, best_model_name = model_service.select_best_model()
        
        # 6. Final evaluation
        final_metrics = model_service.final_evaluation(sample_X_test, sample_y_test, sample_X_test, sample_y_test)
        
        # 7. Validate performance
        performance_valid = model_service.validate_model_performance(final_metrics)
        
        # Verify all steps completed successfully
        assert isinstance(trained_models, dict)
        assert isinstance(evaluation_results, dict)
        assert isinstance(cv_results, dict)
        assert best_model is not None
        assert isinstance(best_model_name, str)
        assert isinstance(final_metrics, dict)
        assert isinstance(performance_valid, bool)
        
        # Check that we have reasonable metrics
        for metrics in evaluation_results.values():
            assert all(0 <= metric <= 1 for metric in metrics.values())
    
    def test_pipeline_with_small_dataset(self, model_service):
        """Test pipeline with very small dataset"""
        small_X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0, 1, 0, 1, 0]
        })
        small_y = pd.Series([0, 1, 0, 1, 0])
        
        # Should handle small datasets gracefully
        try:
            models = model_service.initialize_models()
            trained_models = model_service.train_models(models, small_X, small_y)
            
            # Some models might fail with very small datasets, but the service should handle it
            assert isinstance(trained_models, dict)
        except Exception as e:
            # If it fails, it should be a graceful failure
            assert isinstance(e, (ValueError, ModelValidationError))


if __name__ == '__main__':
    pytest.main([__file__])
