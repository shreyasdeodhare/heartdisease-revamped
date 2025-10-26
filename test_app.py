"""
Test script to verify the Heart Disease Prediction application works correctly
"""

import sys
import os

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_imports():
    """Test if all imports work correctly"""
    try:
        print("Testing imports...")
        
        # Test configuration
        from app.config import config
        print("✓ Configuration imported successfully")
        
        # Test services
        from app.services.data_service import DataService
        from app.services.eda_service import EDAService
        from app.services.model_service import ModelService
        print("✓ Services imported successfully")
        
        # Test utilities
        from app.utils.logging_utils import aop_logger
        from app.utils.error_handling import error_handler
        print("✓ Utilities imported successfully")
        
        # Test main application
        from app.main import HeartDiseasePredictionApp
        print("✓ Main application imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    try:
        print("\nTesting configuration...")
        
        from app.config import config
        
        # Test data config
        assert config.data.data_url is not None
        print("✓ Data configuration loaded")
        
        # Test model config
        assert config.model.test_size > 0
        print("✓ Model configuration loaded")
        
        # Test logging config
        assert config.logging.log_level is not None
        print("✓ Logging configuration loaded")
        
        print("✅ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    try:
        print("\nTesting data loading...")
        
        from app.services.data_service import DataService
        
        data_service = DataService()
        
        # Test data loading (this will actually download data)
        print("Loading data from URL...")
        df = data_service.load_data()
        
        assert df is not None
        assert len(df) > 0
        print(f"✓ Data loaded successfully: {df.shape}")
        
        # Test data preprocessing
        print("Testing data preprocessing...")
        df_processed = data_service.preprocess_data(df)
        
        assert df_processed is not None
        assert len(df_processed) > 0
        print(f"✓ Data preprocessing successful: {df_processed.shape}")
        
        print("✅ Data loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_services_initialization():
    """Test service initialization"""
    try:
        print("\nTesting service initialization...")
        
        from app.services.data_service import DataService
        from app.services.eda_service import EDAService
        from app.services.model_service import ModelService
        
        # Test data service
        data_service = DataService()
        assert data_service is not None
        print("✓ DataService initialized")
        
        # Test EDA service
        eda_service = EDAService()
        assert eda_service is not None
        print("✓ EDAService initialized")
        
        # Test model service
        model_service = ModelService()
        assert model_service is not None
        print("✓ ModelService initialized")
        
        print("✅ Service initialization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Service initialization error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("HEART DISEASE PREDICTION - APPLICATION TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Service Initialization Test", test_services_initialization),
        ("Data Loading Test", test_data_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
        print("\nTo run the complete pipeline:")
        print("python app/main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
