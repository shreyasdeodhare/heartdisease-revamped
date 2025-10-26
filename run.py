"""
Simple run script for Heart Disease Prediction application
"""

import sys
import os

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def main():
    """Run the Heart Disease Prediction application"""
    try:
        print("Starting Heart Disease Prediction Application...")
        print("="*60)
        
        # Import and run the main application
        from app.main import main as app_main
        app_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
