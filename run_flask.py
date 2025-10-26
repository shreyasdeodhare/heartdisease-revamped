"""
Run script for HeartGuard AI Flask application
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r flask_requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'static/css',
        'static/js',
        'templates',
        'models',
        'outputs',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main function to run the Flask app"""
    print("🚀 Starting HeartGuard AI Flask Application")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        sys.exit(1)
    
    print("✅ All dependencies found!")
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check if models exist
    models_dir = './models'
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        print("\n⚠️  No trained models found!")
        print("   Please run the training pipeline first:")
        print("   python app/main.py")
        print("\n   Or run the Flask app in demo mode...")
    
    # Start Flask app
    print("\n🌐 Starting Flask application...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from flask_app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Flask application stopped.")
    except Exception as e:
        print(f"\n❌ Error starting Flask app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
