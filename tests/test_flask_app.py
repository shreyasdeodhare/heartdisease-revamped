"""
Unit tests for Flask application routes and API endpoints
"""
import pytest
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from flask_app import app, validate_and_coerce_input, preprocess_user_input, apply_feature_engineering, get_risk_level, safe_float, safe_int, get_recommendations


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        'BMI': '25.5',
        'PhysicalHealth': '0',
        'MentalHealth': '0',
        'SleepTime': '7',
        'AgeCategory': '55-59',
        'Sex': 'Male',
        'Smoking': 'No',
        'AlcoholDrinking': 'No',
        'PhysicalActivity': 'Yes',
        'GenHealth': 'Very good',
        'Diabetic': 'No',
        'Asthma': 'No',
        'KidneyDisease': 'No',
        'SkinCancer': 'No'
    }


@pytest.fixture
def mock_models():
    """Mock loaded models for testing"""
    mock_model = Mock()
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    mock_model.predict.return_value = [1]
    
    return {
        'best_model': mock_model,
        'feature_names': ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Age_Numeric', 'Health_Score', 'Risk_Score']
    }


class TestFlaskRoutes:
    """Test Flask application routes"""
    
    def test_index_route(self, client):
        """Test index route returns correct template"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'index' in response.data or b'Heart' in response.data
    
    def test_predict_page_route(self, client):
        """Test predict page route"""
        response = client.get('/predict')
        assert response.status_code == 200
        assert b'predict' in response.data
    
    def test_analysis_page_route(self, client):
        """Test analysis page route"""
        response = client.get('/analysis')
        assert response.status_code == 200
        assert b'analysis' in response.data
    
    def test_about_page_route(self, client):
        """Test about page route"""
        response = client.get('/about')
        assert response.status_code == 200
        assert b'about' in response.data


class TestAPIEndpoints:
    """Test API endpoints"""
    
    @patch('flask_app.loaded_models', {'best_model': None})
    def test_predict_endpoint_no_model(self, client, sample_user_data):
        """Test predict endpoint when no model is loaded"""
        response = client.post('/api/predict', 
                              data=json.dumps(sample_user_data),
                              content_type='application/json')
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
    
    @patch('flask_app.loaded_models', {'best_model': Mock()})
    @patch('flask_app.preprocess_user_input')
    @patch('flask_app.apply_feature_engineering')
    def test_predict_endpoint_success(self, client, sample_user_data, mock_models):
        """Test successful prediction endpoint"""
        with patch('flask_app.loaded_models', mock_models):
            with patch('flask_app.preprocess_user_input') as mock_preprocess:
                with patch('flask_app.apply_feature_engineering') as mock_feature:
                    mock_preprocess.return_value = pd.DataFrame([sample_user_data])
                    mock_feature.return_value = pd.DataFrame([[25.5, 0, 0, 7, 57, 0.8, 0.2]], 
                                                             columns=['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Age_Numeric', 'Health_Score', 'Risk_Score'])
                    
                    response = client.post('/api/predict',
                                          data=json.dumps(sample_user_data),
                                          content_type='application/json')
                    
                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert 'prediction' in data
                    assert 'probability' in data
                    assert 'risk_level' in data
                    assert 'recommendations' in data
    
    def test_predict_endpoint_invalid_json(self, client):
        """Test predict endpoint with invalid JSON"""
        response = client.post('/api/predict',
                              data='invalid json',
                              content_type='application/json')
        assert response.status_code == 400
    
    @patch('flask_app.data_service.load_data')
    def test_analysis_data_endpoint(self, client):
        """Test analysis data endpoint"""
        mock_df = pd.DataFrame({
            'HeartDisease': [0, 1, 0, 1],
            'BMI': [25.0, 30.0, 22.0, 28.0],
            'AgeCategory': ['25-29', '55-59', '25-29', '55-59']
        })
        
        with patch('flask_app.data_service.load_data', return_value=mock_df):
            response = client.get('/api/analysis-data')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'dataset_info' in data
            assert 'risk_factors' in data
    
    def test_health_tips_endpoint(self, client):
        """Test health tips endpoint"""
        response = client.get('/api/health-tips')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'general' in data
        assert isinstance(data['general'], list)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_validate_and_coerce_input_numeric_fields(self):
        """Test numeric field validation and coercion"""
        data = {
            'BMI': '25.5',
            'PhysicalHealth': '5',
            'MentalHealth': '3',
            'SleepTime': '7',
            'Age_Numeric': '45',
            'Health_Score': '0.8',
            'Risk_Score': '0.2'
        }
        
        result = validate_and_coerce_input(data)
        
        assert result['BMI'] == 25.5
        assert result['PhysicalHealth'] == 5.0
        assert result['MentalHealth'] == 3.0
        assert result['SleepTime'] == 7.0
        assert result['Age_Numeric'] == 45.0
        assert result['Health_Score'] == 0.8
        assert result['Risk_Score'] == 0.2
    
    def test_validate_and_coerce_input_invalid_numeric(self):
        """Test handling of invalid numeric values"""
        data = {
            'BMI': 'invalid',
            'PhysicalHealth': '5'
        }
        
        result = validate_and_coerce_input(data)
        
        assert result['BMI'] == 0.0  # Should default to 0.0
        assert result['PhysicalHealth'] == 5.0
    
    def test_validate_and_coerce_input_binary_fields(self):
        """Test binary field validation"""
        data = {
            'Smoking': 'Yes',
            'AlcoholDrinking': 'No',
            'PhysicalActivity': 'yes'  # Should be normalized
        }
        
        result = validate_and_coerce_input(data)
        
        assert result['Smoking'] == 'Yes'
        assert result['AlcoholDrinking'] == 'No'
        assert result['PhysicalActivity'] == 'Yes'  # Should be normalized
    
    def test_preprocess_user_input_dict(self, sample_user_data):
        """Test preprocessing user input from dictionary"""
        result = preprocess_user_input(sample_user_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'BMI' in result.columns
    
    def test_preprocess_user_input_dataframe(self, sample_user_data):
        """Test preprocessing user input from DataFrame"""
        df = pd.DataFrame([sample_user_data])
        result = preprocess_user_input(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        # Should be a copy, not the same object
        assert result is not df
    
    def test_apply_feature_engineering(self):
        """Test feature engineering application"""
        df = pd.DataFrame({
            'BMI': [25.5, 30.0],
            'AgeCategory': ['25-29', '55-59'],
            'PhysicalHealth': [0, 5],
            'MentalHealth': [0, 3],
            'SleepTime': [7, 6]
        })
        
        result = apply_feature_engineering(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'BMI' in result.columns
        # Check if BMI categories are created
        assert 'BMI_Category' in result.columns
    
    def test_get_risk_level(self):
        """Test risk level determination"""
        assert get_risk_level(0.1) == "Low Risk"
        assert get_risk_level(0.3) == "Low Risk"
        assert get_risk_level(0.4) == "Medium Risk"
        assert get_risk_level(0.6) == "Medium Risk"
        assert get_risk_level(0.7) == "High Risk"
        assert get_risk_level(0.9) == "High Risk"
    
    def test_safe_float(self):
        """Test safe float conversion"""
        assert safe_float("25.5") == 25.5
        assert safe_float(25) == 25.0
        assert safe_float(None) == 0.0
        assert safe_float("invalid") == 0.0
        assert safe_float("invalid", default=1.0) == 1.0
    
    def test_safe_int(self):
        """Test safe int conversion"""
        assert safe_int("25") == 25
        assert safe_int(25.5) == 25
        assert safe_int(None) == 0
        assert safe_int("invalid") == 0
        assert safe_int("invalid", default=1) == 1
    
    def test_get_recommendations(self):
        """Test recommendations generation"""
        form_data = {
            'BMI': '30.0',
            'Smoking': 'Yes',
            'PhysicalActivity': 'No'
        }
        probability = 0.8
        
        recommendations = get_recommendations(form_data, probability)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should contain recommendations for high risk
        assert any('weight' in rec.lower() or 'smoking' in rec.lower() or 'exercise' in rec.lower() 
                  for rec in recommendations)
    
    def test_get_recommendations_low_risk(self):
        """Test recommendations for low risk"""
        form_data = {'BMI': '22.0', 'Smoking': 'No', 'PhysicalActivity': 'Yes'}
        probability = 0.1
        
        recommendations = get_recommendations(form_data, probability)
        
        assert isinstance(recommendations, list)
        # Should have maintenance recommendations
        assert len(recommendations) > 0


class TestErrorHandling:
    """Test error handling in Flask app"""
    
    def test_predict_endpoint_missing_fields(self, client):
        """Test predict endpoint with missing required fields"""
        incomplete_data = {'BMI': '25.5'}  # Missing many required fields
        
        response = client.post('/api/predict',
                              data=json.dumps(incomplete_data),
                              content_type='application/json')
        
        # Should handle gracefully, either with error or by filling defaults
        assert response.status_code in [200, 400]
    
    @patch('flask_app.preprocess_user_input')
    def test_preprocessing_error_handling(self, client, sample_user_data):
        """Test error handling in preprocessing"""
        with patch('flask_app.preprocess_user_input', side_effect=Exception("Preprocessing error")):
            response = client.post('/api/predict',
                                  data=json.dumps(sample_user_data),
                                  content_type='application/json')
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data


if __name__ == '__main__':
    pytest.main([__file__])
