"""
Unit tests for data service functions
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from services.data_service import DataService
from utils.error_handling import DataValidationError


@pytest.fixture
def data_service():
    """Create DataService instance for testing"""
    return DataService()


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'HeartDisease': [0, 1, 0, 1, 0],
        'BMI': [25.5, 30.0, 22.0, 28.5, 35.0],
        'Smoking': ['Yes', 'No', 'No', 'Yes', 'No'],
        'AlcoholDrinking': ['No', 'No', 'Yes', 'No', 'No'],
        'PhysicalActivity': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'AgeCategory': ['25-29', '55-59', '25-29', '55-59', '65-69'],
        'Sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'PhysicalHealth': [0, 5, 0, 3, 10],
        'MentalHealth': [0, 3, 0, 5, 2],
        'SleepTime': [7, 6, 8, 5, 7],
        'GenHealth': ['Very good', 'Fair', 'Excellent', 'Good', 'Fair'],
        'Diabetic': ['No', 'Yes', 'No', 'No', 'Yes'],
        'Asthma': ['No', 'No', 'Yes', 'No', 'No'],
        'KidneyDisease': ['No', 'No', 'No', 'Yes', 'No'],
        'SkinCancer': ['No', 'Yes', 'No', 'No', 'No']
    })


@pytest.fixture
def dataframe_with_missing():
    """Create DataFrame with missing values"""
    return pd.DataFrame({
        'HeartDisease': [0, 1, 0, 1, np.nan],
        'BMI': [25.5, np.nan, 22.0, 28.5, 35.0],
        'Smoking': ['Yes', 'No', np.nan, 'Yes', 'No'],
        'PhysicalHealth': [0, 5, 0, 3, 10]
    })


class TestDataServiceLoadData:
    """Test data loading functionality"""
    
    @patch('pandas.read_csv')
    def test_load_data_from_url(self, mock_read_csv, data_service):
        """Test loading data from URL"""
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df
        
        result = data_service.load_data('test_url.csv')
        
        assert isinstance(result, pd.DataFrame)
        mock_read_csv.assert_called_once_with('test_url.csv')
    
    @patch('pandas.read_csv')
    def test_load_data_default_url(self, mock_read_csv, data_service):
        """Test loading data with default URL"""
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df
        
        result = data_service.load_data()
        
        assert isinstance(result, pd.DataFrame)
        mock_read_csv.assert_called_once()
    
    @patch('pandas.read_csv')
    def test_load_data_error_handling(self, mock_read_csv, data_service):
        """Test error handling in data loading"""
        mock_read_csv.side_effect = Exception("File not found")
        
        with pytest.raises(Exception):
            data_service.load_data('nonexistent.csv')


class TestDataServicePreprocess:
    """Test data preprocessing functionality"""
    
    def test_preprocess_data_basic(self, data_service, sample_dataframe):
        """Test basic data preprocessing"""
        result = data_service.preprocess_data(sample_dataframe)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        # Should not modify the original DataFrame
        assert result is not sample_dataframe
    
    def test_preprocess_data_empty_dataframe(self, data_service):
        """Test preprocessing empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, DataValidationError)):
            data_service.preprocess_data(empty_df)


class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    def test_engineer_features_bmi_categories(self, data_service, sample_dataframe):
        """Test BMI category creation"""
        result = data_service._engineer_features(sample_dataframe)
        
        assert isinstance(result, pd.DataFrame)
        assert 'BMI_Category' in result.columns
        
        # Check BMI categories are correct
        bmi_categories = result['BMI_Category'].tolist()
        assert 'Normal weight' in bmi_categories
        assert 'Overweight' in bmi_categories
        assert 'Obese' in bmi_categories
    
    def test_engineer_features_age_numeric(self, data_service, sample_dataframe):
        """Test age numeric conversion"""
        result = data_service._engineer_features(sample_dataframe)
        
        assert 'Age_Numeric' in result.columns
        assert result['Age_Numeric'].dtype in ['int64', 'float64']
        
        # Check age conversion is reasonable
        age_values = result['Age_Numeric'].tolist()
        assert all(20 <= age <= 70 for age in age_values if pd.notna(age))
    
    def test_engineer_features_health_scores(self, data_service, sample_dataframe):
        """Test health score calculations"""
        result = data_service._engineer_features(sample_dataframe)
        
        assert 'Health_Score' in result.columns
        assert 'Risk_Score' in result.columns
        
        # Check scores are in expected range
        health_scores = result['Health_Score']
        risk_scores = result['Risk_Score']
        
        assert all(0 <= score <= 1 for score in health_scores if pd.notna(score))
        assert all(0 <= score <= 1 for score in risk_scores if pd.notna(score))


class TestCategoricalEncoding:
    """Test categorical variable encoding"""
    
    def test_encode_categorical_variables(self, data_service, sample_dataframe):
        """Test categorical variable encoding"""
        # First engineer features to create BMI categories
        df_engineered = data_service._engineer_features(sample_dataframe)
        
        result = data_service.encode_categorical_variables(df_engineered)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df_engineered)
        
        # Check that categorical columns are encoded
        categorical_cols = ['Smoking', 'AlcoholDrinking', 'PhysicalActivity', 'Sex', 'GenHealth']
        for col in categorical_cols:
            if col in result.columns:
                assert result[col].dtype in ['int64', 'float64', 'uint8']
    
    def test_encode_categorical_variables_preserves_numerical(self, data_service, sample_dataframe):
        """Test that numerical columns are preserved during encoding"""
        df_engineered = data_service._engineer_features(sample_dataframe)
        
        result = data_service.encode_categorical_variables(df_engineered)
        
        # Check that numerical columns remain unchanged
        numerical_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Age_Numeric']
        for col in numerical_cols:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col])


class TestFeatureScaling:
    """Test feature scaling functionality"""
    
    def test_scale_features(self, data_service):
        """Test feature scaling"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        result = data_service.scale_features(X)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)
        assert len(result.columns) == len(X.columns)
        
        # Check that values are scaled between 0 and 1
        for col in result.columns:
            assert result[col].min() >= 0
            assert result[col].max() <= 1
    
    def test_scale_features_stores_scaler(self, data_service):
        """Test that scaler is stored for later use"""
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        
        data_service.scale_features(X)
        
        assert data_service.scaler is not None


class TestClassImbalanceHandling:
    """Test class imbalance handling"""
    
    def test_handle_class_imbalance_basic(self, data_service):
        """Test basic class imbalance handling"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Balanced classes
        
        X_balanced, y_balanced = data_service.handle_class_imbalance(X, y)
        
        assert isinstance(X_balanced, pd.DataFrame)
        assert isinstance(y_balanced, pd.Series)
        assert len(X_balanced) == len(y_balanced)
    
    def test_handle_class_imbalance_imbalanced(self, data_service):
        """Test class imbalance handling with imbalanced data"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # Imbalanced classes
        
        X_balanced, y_balanced = data_service.handle_class_imbalance(X, y)
        
        assert isinstance(X_balanced, pd.DataFrame)
        assert isinstance(y_balanced, pd.Series)
        assert len(X_balanced) == len(y_balanced)
        
        # Check that classes are more balanced
        class_counts = y_balanced.value_counts()
        assert abs(class_counts[0] - class_counts[1]) <= 2  # Allow small difference


class TestFeatureSelection:
    """Test feature selection functionality"""
    
    def test_select_features_basic(self, data_service):
        """Test basic feature selection"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'noise': [np.random.randn() for _ in range(10)]
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        X_selected, selected_features = data_service.select_features(X, y)
        
        assert isinstance(X_selected, pd.DataFrame)
        assert isinstance(selected_features, list)
        assert len(X_selected.columns) <= len(X.columns)  # Should reduce or maintain features
        assert len(selected_features) == len(X_selected.columns)
    
    def test_select_features_large_dataset(self, data_service):
        """Test feature selection with large dataset (subset selection)"""
        # Create larger dataset
        n_samples = 100000
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        X_selected, selected_features = data_service.select_features(X, y)
        
        assert isinstance(X_selected, pd.DataFrame)
        assert isinstance(selected_features, list)
        # Should use subset for large datasets
        assert len(X_selected) == n_samples


class TestDataSplitting:
    """Test data splitting functionality"""
    
    def test_split_data_basic(self, data_service):
        """Test basic data splitting"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        X_train, X_test, y_train, y_test = data_service.split_data(X, y)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Check split proportions
        total_samples = len(X)
        train_samples = len(X_train)
        test_samples = len(X_test)
        
        assert train_samples + test_samples == total_samples
        assert train_samples > test_samples  # Train should be larger
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestDataSummary:
    """Test data summary functionality"""
    
    def test_get_data_summary(self, data_service, sample_dataframe):
        """Test data summary generation"""
        summary = data_service.get_data_summary(sample_dataframe)
        
        assert isinstance(summary, dict)
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'missing_values' in summary
        assert 'memory_usage' in summary
        
        assert summary['shape'] == sample_dataframe.shape
        assert len(summary['columns']) == len(sample_dataframe.columns)


class TestMissingValueHandling:
    """Test missing value handling"""
    
    def test_handle_missing_values(self, data_service, dataframe_with_missing):
        """Test missing value handling"""
        result = data_service._handle_missing_values(dataframe_with_missing)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(dataframe_with_missing)
        
        # Check that missing values are handled
        # (The exact handling depends on implementation)
        missing_after = result.isnull().sum()
        
        # Either no missing values or handled appropriately
        assert missing_after.sum() <= dataframe_with_missing.isnull().sum()


class TestIntegration:
    """Integration tests for data service"""
    
    def test_complete_preprocessing_pipeline(self, data_service, sample_dataframe):
        """Test complete preprocessing pipeline"""
        # Step 1: Preprocess
        df_processed = data_service.preprocess_data(sample_dataframe)
        
        # Step 2: Engineer features
        df_engineered = data_service._engineer_features(df_processed)
        
        # Step 3: Encode categorical variables
        df_encoded = data_service.encode_categorical_variables(df_engineered)
        
        # Step 4: Split features and target
        X = df_encoded.drop('HeartDisease', axis=1)
        y = df_encoded['HeartDisease']
        
        # Step 5: Handle class imbalance
        X_balanced, y_balanced = data_service.handle_class_imbalance(X, y)
        
        # Step 6: Split data
        X_train, X_test, y_train, y_test = data_service.split_data(X_balanced, y_balanced)
        
        # Verify all steps completed successfully
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Check data integrity
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train.columns) == len(X_test.columns)


if __name__ == '__main__':
    pytest.main([__file__])
