"""
Install all dependencies for HeartGuard AI project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages"""
    requirements_files = [
        'requirements.txt',  # Main project dependencies
        'flask_requirements.txt'  # Flask frontend dependencies
    ]
    
    print("🚀 Installing HeartGuard AI Dependencies")
    print("=" * 50)
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"\n📦 Installing from {req_file}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file])
                print(f"✅ {req_file} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error installing {req_file}: {e}")
                return False
        else:
            print(f"⚠️  {req_file} not found, skipping...")
    
    print("\n🎉 All dependencies installed successfully!")
    return True

def check_installation():
    """Check if key packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'flask', 
        'xgboost', 'lightgbm', 'imbalanced-learn'
    ]
    
    print("\n🔍 Checking installation...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

if __name__ == "__main__":
    print("HeartGuard AI - Dependency Installation")
    print("=" * 50)
    
    # Install dependencies
    if install_requirements():
        # Check installation
        if check_installation():
            print("\n🎉 Installation completed successfully!")
            print("\nNext steps:")
            print("1. Train the models: python app/main.py")
            print("2. Start Flask frontend: python run_flask.py")
        else:
            print("\n❌ Installation check failed. Please install missing packages manually.")
    else:
        print("\n❌ Installation failed. Please check the error messages above.")
