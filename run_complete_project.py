"""
Complete HeartGuard AI Project Runner
This script runs the entire project with proper workflow
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def print_step(step_num, description):
    """Print formatted step"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def check_file_exists(file_path, description):
    """Check if file exists"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def run_command(command, description, wait=True):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    print(f"Command: {command}")
    
    try:
        if wait:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ {description} completed successfully")
            return True
        else:
            # Run in background
            subprocess.Popen(command, shell=True)
            print(f"✅ {description} started in background")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    """Main project runner"""
    print_header("HeartGuard AI - Complete Project Runner")
    
    # Step 1: Check project structure
    print_step(1, "Checking Project Structure")
    
    required_files = [
        ("app/main.py", "Main training script"),
        ("flask_app.py", "Flask frontend"),
        ("run_flask.py", "Flask runner"),
        ("requirements.txt", "Main dependencies"),
        ("flask_requirements.txt", "Flask dependencies")
    ]
    
    all_files_exist = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some required files are missing. Please check the project structure.")
        return False
    
    print("\n✅ All required files found!")
    
    # Step 2: Install dependencies
    print_step(2, "Installing Dependencies")
    
    print("Installing main project dependencies...")
    if not run_command("pip install -r requirements.txt", "Main dependencies installation"):
        print("⚠️  Main dependencies installation had issues, continuing...")
    
    print("\nInstalling Flask dependencies...")
    if not run_command("pip install -r flask_requirements.txt", "Flask dependencies installation"):
        print("⚠️  Flask dependencies installation had issues, continuing...")
    
    # Step 3: Train the ML models
    print_step(3, "Training Machine Learning Models")
    
    print("This step will:")
    print("- Load and preprocess the heart disease dataset")
    print("- Train multiple ML models (Random Forest, XGBoost, LightGBM)")
    print("- Perform hyperparameter tuning and cross-validation")
    print("- Save the best model and preprocessing artifacts")
    print("- Generate visualizations and analysis reports")
    
    user_input = input("\n🤔 Do you want to train the models now? (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        print("\n🔄 Training models... (This may take 5-10 minutes)")
        if run_command("python app/main.py", "Model training"):
            print("✅ Models trained successfully!")
            
            # Check if model files were created
            model_files = [
                "models/best_model.pkl",
                "models/scaler.pkl", 
                "models/label_encoders.pkl"
            ]
            
            models_created = True
            for model_file in model_files:
                if not check_file_exists(model_file, "Model file"):
                    models_created = False
            
            if not models_created:
                print("⚠️  Some model files were not created. Training may have failed.")
        else:
            print("❌ Model training failed. You can still run the Flask frontend in demo mode.")
    else:
        print("⏭️  Skipping model training. You can train later with: python app/main.py")
    
    # Step 4: Start Flask frontend
    print_step(4, "Starting Flask Frontend")
    
    print("Starting the Flask web application...")
    print("The application will be available at: http://localhost:5000")
    
    user_input = input("\n🤔 Do you want to start the Flask frontend now? (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        print("\n🌐 Starting Flask application...")
        print("Press Ctrl+C to stop the server")
        print("\n" + "=" * 60)
        print("🚀 Flask server starting...")
        print("=" * 60)
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start Flask app
        try:
            run_command("python run_flask.py", "Flask application", wait=True)
        except KeyboardInterrupt:
            print("\n👋 Flask application stopped.")
    else:
        print("⏭️  Skipping Flask startup. You can start later with: python run_flask.py")
    
    # Step 5: Test the connection
    print_step(5, "Testing Frontend-Backend Connection")
    
    user_input = input("\n🤔 Do you want to test the API connection? (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        print("\n🧪 Testing API connection...")
        if run_command("python test_prediction.py", "API connection test"):
            print("✅ API connection test completed!")
        else:
            print("⚠️  API connection test failed. Check if Flask is running.")
    
    # Final instructions
    print_header("Project Setup Complete!")
    
    print("🎉 HeartGuard AI project is ready!")
    print("\n📋 Available Commands:")
    print("• Train models: python app/main.py")
    print("• Start Flask: python run_flask.py")
    print("• Test API: python test_prediction.py")
    print("• Install deps: python install_dependencies.py")
    
    print("\n🌐 Access the application:")
    print("• Homepage: http://localhost:5000")
    print("• Prediction: http://localhost:5000/predict")
    print("• Analysis: http://localhost:5000/analysis")
    print("• About: http://localhost:5000/about")
    
    print("\n💡 Tips:")
    print("• Make sure to train models first for accurate predictions")
    print("• The Flask frontend provides a beautiful, modern UI")
    print("• All predictions are based on trained ML models")
    print("• Check the logs for any issues")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Project setup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
