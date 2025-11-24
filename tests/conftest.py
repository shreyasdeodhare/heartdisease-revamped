import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

# Set up test environment
os.environ['TESTING'] = 'True'
os.environ['LOG_LEVEL'] = 'DEBUG'


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory"""
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture(scope="session")
def mock_data():
    """Generate mock data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'HeartDisease': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'BMI': np.random.normal(28.5, 6.0, n_samples),
        'Smoking': np.random.choice(['Yes', 'No'], n_samples),
        'PhysicalActivity': np.random.choice(['Yes', 'No'], n_samples),
        'AgeCategory': np.random.choice(['25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'], n_samples),
        'Sex': np.random.choice(['Male', 'Female'], n_samples),
        'PhysicalHealth': np.random.randint(0, 30, n_samples),
        'MentalHealth': np.random.randint(0, 30, n_samples),
        'SleepTime': np.random.randint(1, 12, n_samples),
        'GenHealth': np.random.choice(['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], n_samples),
        'Diabetic': np.random.choice(['Yes', 'No'], n_samples),
        'Asthma': np.random.choice(['Yes', 'No'], n_samples),
        'KidneyDisease': np.random.choice(['Yes', 'No'], n_samples),
        'SkinCancer': np.random.choice(['Yes', 'No'], n_samples)
    })


@pytest.fixture
def sample_user_input():
    """Sample user input for prediction testing"""
    return {
        'BMI': '28.5',
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


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test file location
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        elif "test_flask_app" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_data_service" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_model_service" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_utils" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
