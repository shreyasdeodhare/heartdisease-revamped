"""
Test script to verify Flask frontend-backend connection
"""

import requests
import json

def test_prediction_api():
    """Test the prediction API endpoint"""
    
    # Test data
    test_data = {
        "BMI": 28.5,
        "Smoking": "No",
        "AlcoholDrinking": "No",
        "Stroke": "No",
        "PhysicalHealth": 5,
        "MentalHealth": 3,
        "DiffWalking": "No",
        "Sex": "Male",
        "AgeCategory": "45-49",
        "Race": "White",
        "Diabetic": "No",
        "PhysicalActivity": "Yes",
        "GenHealth": "Good",
        "SleepTime": 7.5,
        "Asthma": "No",
        "KidneyDisease": "No",
        "SkinCancer": "No"
    }
    
    print("🧪 Testing Heart Disease Prediction API")
    print("=" * 50)
    
    try:
        # Make request to prediction API
        response = requests.post(
            'http://localhost:5000/api/predict',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"🎯 Prediction: {'High Risk' if result['prediction'] == 1 else 'Low Risk'}")
            print(f"📊 Disease Probability: {result['probability']['disease']:.2%}")
            print(f"⚠️  Risk Level: {result['risk_level']}")
            print(f"💡 Recommendations: {len(result['recommendations'])} suggestions")
            
            print("\n📋 Full Response:")
            print(json.dumps(result, indent=2))
            
            return True
        elif response.status_code == 503:
            result = response.json()
            print("⚠️  Model not trained!")
            print(f"📄 Message: {result.get('message', 'No trained model available')}")
            print("💡 Please run: python app/main.py")
            return False
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask app is not running")
        print("💡 Start the Flask app with: python run_flask.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analysis_api():
    """Test the analysis API endpoint"""
    
    print("\n🧪 Testing Analysis API")
    print("=" * 30)
    
    try:
        response = requests.get('http://localhost:5000/api/analysis')
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis data loaded successfully!")
            print(f"📊 Dataset samples: {result['dataset_info']['total_samples']}")
            print(f"🔢 Features: {result['dataset_info']['features']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_tips_api():
    """Test the health tips API endpoint"""
    
    print("\n🧪 Testing Health Tips API")
    print("=" * 30)
    
    try:
        response = requests.get('http://localhost:5000/api/health-tips')
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health tips loaded successfully!")
            print(f"💡 General tips: {len(result['general'])}")
            print(f"❤️  Heart health tips: {len(result['heart_health'])}")
            print(f"🛡️  Prevention tips: {len(result['prevention'])}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 HeartGuard AI - Frontend-Backend Connection Test")
    print("=" * 60)
    
    tests = [
        ("Prediction API", test_prediction_api),
        ("Analysis API", test_analysis_api),
        ("Health Tips API", test_health_tips_api)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} - PASSED")
        else:
            print(f"❌ {test_name} - FAILED")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Frontend is properly connected to backend.")
        print("\n💡 You can now:")
        print("   - Fill out the prediction form at http://localhost:5000/predict")
        print("   - View analysis at http://localhost:5000/analysis")
        print("   - Get personalized heart disease risk assessments")
    else:
        print("⚠️  Some tests failed. Please check the Flask app is running.")
        print("   Start with: python run_flask.py")

if __name__ == "__main__":
    main()
